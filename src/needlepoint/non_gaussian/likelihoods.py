"""Gaussian expectations and observation moments for three observation models.

The exponential model is never capped. Nonfinite rates raise an error so that
the optimizer can reject the step without silently changing the likelihood.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from .data import fingerprint


def vector(value, name, *, size=None, device='cpu'):
    array = np.asarray(value, dtype=float)
    if size is not None:
        array = np.broadcast_to(array, (size,))
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite vector')
    return torch.tensor(array.copy(), dtype=torch.float64, device=device)


def checked_exp(value):
    result = value.exp()
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite exponential rate; reduce the optimization step or rescale the model')
    return result


def lognormal_moments(mean, variance):
    """Stable lognormal moments, including tiny/large variances and zero."""
    rate = checked_exp(mean + variance / 2)
    # exp(2m+2v)*(1-exp(-v)) avoids an overflow times underflow product.
    log_fraction = (-torch.expm1(-variance)).log()
    rate_variance = checked_exp(2 * mean + 2 * variance + log_fraction)
    return rate, rate_variance


@dataclass
class ExpectedLikelihood:
    value: torch.Tensor
    score: torch.Tensor
    information: torch.Tensor


def _moments(mean, variance):
    if mean.shape != variance.shape or not torch.isfinite(mean).all():
        raise ValueError('Mean and variance must be matching finite vectors')
    if not torch.isfinite(variance).all() or (variance < 0).any():
        raise ValueError('Latent variance must be finite and nonnegative')


def inverse_mills(a):
    """Stable phi(a)/Phi(a), including very negative arguments."""
    result = torch.empty_like(a)
    negative = a < 0
    result[negative] = math.sqrt(2 / math.pi) / torch.special.erfcx(-a[negative] / math.sqrt(2))
    positive = a[~negative]
    result[~negative] = torch.exp(-.5 * positive.square() - .5 * math.log(2 * math.pi)
                                 - torch.special.log_ndtr(positive))
    return result


def censored_information(a):
    """Stable negative second derivative of log Phi, without tail cancellation."""
    r = inverse_mills(a)
    value = torch.empty_like(a)
    tail = a < -32
    t = a[tail].reciprocal().square()
    value[tail] = 1 - t + 6*t**2 - 50*t**3 + 518*t**4 - 6354*t**5
    value[~tail] = r[~tail] * (a[~tail] + r[~tail])
    return value


class TobitLikelihood:
    name = 'tobit'

    def __init__(self, values, is_censored, *, threshold, noise_sd, quadrature_order=64,
                 device='cpu', measurement_scale='declared_gaussian'):
        self.values = vector(values, 'values', device=device)
        n = len(self.values)
        if n == 0:
            raise ValueError('At least one acquired Tobit observation is required')
        flags = np.asarray(is_censored)
        if flags.dtype != bool or flags.shape != (n,):
            raise ValueError('is_censored must be an explicit boolean vector')
        self.is_censored = torch.tensor(flags.copy(), device=device)
        self.threshold = vector(threshold, 'threshold', size=n, device=device)
        self.noise_sd = float(noise_sd)
        if not math.isfinite(self.noise_sd) or self.noise_sd <= 0:
            raise ValueError('noise_sd must be finite and positive')
        if not isinstance(quadrature_order, int) or not 2 <= quadrature_order <= 256:
            raise ValueError('quadrature_order must be between 2 and 256')
        if torch.any(self.values[self.is_censored] != 0):
            raise ValueError('Censored values must use the declared zero code')
        if torch.any(self.values[~self.is_censored] < self.threshold[~self.is_censored]):
            raise ValueError('Uncensored observations must be at or above their threshold')
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_order)
        self.nodes = vector(nodes, 'nodes', device=device)
        self.weights = vector(weights / np.sqrt(np.pi), 'weights', device=device)
        self.measurement_scale = str(measurement_scale)
        self.size = n

    def expected(self, mean, variance):
        _moments(mean, variance)
        sigma = self.noise_sd
        value = -.5 * math.log(2 * math.pi) - math.log(sigma) - (
            (self.values - mean).square() + variance) / (2 * sigma**2)
        score = (self.values - mean) / sigma**2
        information = torch.full_like(mean, 1 / sigma**2)
        mask = self.is_censored
        if mask.any():
            # v=0 has the scalar limiting expectation. The optimizer itself
            # always has positive covariance; the clamp avoids a 0/0 derivative.
            value = value.clone(); score = score.clone(); information = information.clone()
            for selected in torch.nonzero(mask).flatten().split(4096):
                sd = variance[selected].clamp_min(torch.finfo(mean.dtype).tiny).sqrt()
                a = (self.threshold[selected, None] - mean[selected, None]
                     - math.sqrt(2) * sd[:, None] * self.nodes) / sigma
                value[selected] = torch.special.log_ndtr(a) @ self.weights
                score[selected] = -(inverse_mills(a) @ self.weights) / sigma
                information[selected] = (censored_information(a) @ self.weights) / sigma**2
        return ExpectedLikelihood(value, score, information)

    def predict(self, mean, variance):
        """Moments of Y=Z*1{Z>=b}, with measurement noise included."""
        _moments(mean, variance)
        sigma2 = variance + self.noise_sd**2
        sd = sigma2.sqrt()
        if self.threshold.numel() == mean.numel():
            threshold = self.threshold
        elif torch.all(self.threshold == self.threshold[0]):
            threshold = self.threshold[0]
        else:
            raise ValueError('Spatially varying thresholds require explicit prediction thresholds')
        a = (threshold - mean) / sd
        detected = torch.special.ndtr(-a)
        density = torch.exp(-a.square() / 2) / math.sqrt(2 * math.pi)
        observed = mean * detected + sd * density
        second = (mean.square() + sigma2) * detected + sd * (mean + threshold) * density
        return dict(observed_mean=observed, observed_variance=(second - observed.square()).clamp_min(0),
                    censoring_probability=torch.special.ndtr(a), pre_censoring_variance=sigma2)

    def fingerprint(self):
        return fingerprint(self.values.cpu().numpy(), self.is_censored.cpu().numpy(),
            self.threshold.cpu().numpy(), metadata=dict(likelihood=self.name,
                noise_sd=self.noise_sd, quadrature_order=len(self.nodes),
                measurement_scale=self.measurement_scale))


class PoissonLikelihood:
    name = 'poisson'

    def __init__(self, counts, *, exposure=1.0, exposure_semantics='unit', device='cpu'):
        self.counts = vector(counts, 'counts', device=device)
        if len(self.counts) == 0:
            raise ValueError('At least one acquired Poisson observation is required')
        if (self.counts < 0).any() or torch.any(self.counts != self.counts.round()):
            raise ValueError('Poisson observations must be nonnegative raw integer counts')
        self.exposure = vector(exposure, 'exposure', size=len(self.counts), device=device)
        if (self.exposure < 0).any() or torch.any((self.exposure == 0) & (self.counts > 0)):
            raise ValueError('Exposure must be nonnegative; positive counts require positive exposure')
        if exposure_semantics == 'unit' and not torch.all(self.exposure == 1):
            raise ValueError('Nonunit exposure requires explicit exposure_semantics')
        self.exposure_semantics = str(exposure_semantics)
        self.size = len(self.counts)

    def expected(self, mean, variance):
        _moments(mean, variance)
        positive = self.exposure > 0
        rate = torch.zeros_like(mean)
        rate[positive] = checked_exp(self.exposure[positive].log() + mean[positive] + variance[positive] / 2)
        value = -rate - torch.lgamma(self.counts + 1)
        value = value.clone()
        value[positive] += self.counts[positive] * (self.exposure[positive].log() + mean[positive])
        return ExpectedLikelihood(value, self.counts - rate, rate)

    def predict(self, mean, variance, *, exposure=1.0):
        _moments(mean, variance)
        rate, rate_variance = lognormal_moments(mean, variance)
        exposure = torch.as_tensor(exposure, dtype=mean.dtype, device=mean.device)
        if not torch.isfinite(exposure).all() or (exposure < 0).any():
            raise ValueError('Prediction exposure must be finite and nonnegative')
        return dict(rate_mean=rate, rate_variance=rate_variance,
                    observed_mean=exposure * rate,
                    observed_variance=exposure * rate + exposure.square() * rate_variance)

    def fingerprint(self):
        return fingerprint(self.counts.cpu().numpy(), self.exposure.cpu().numpy(),
            metadata=dict(likelihood=self.name, exposure_semantics=self.exposure_semantics))


class LGCPIntegral:
    """Negative expected spatial integral; events enter a separate linear score."""
    name = 'lgcp'

    def __init__(self, areas, *, effort=1.0, device='cpu'):
        self.areas = vector(areas, 'areas', device=device)
        self.effort = vector(effort, 'effort', size=len(self.areas), device=device)
        if (self.areas < 0).any() or self.areas.sum() <= 0 or (self.effort < 0).any():
            raise ValueError('LGCP requires nonnegative effort/areas and positive acquired area')
        self.size = len(self.areas)

    def expected(self, mean, variance):
        _moments(mean, variance)
        positive = (self.areas > 0) & (self.effort > 0)
        rate = torch.zeros_like(mean)
        rate[positive] = checked_exp(self.areas[positive].log() + self.effort[positive].log()
                                     + mean[positive] + variance[positive] / 2)
        return ExpectedLikelihood(-rate, -rate, rate)

    def predict(self, mean, variance, *, effort=1.0):
        _moments(mean, variance)
        effort = torch.as_tensor(effort, dtype=mean.dtype, device=mean.device)
        if not torch.isfinite(effort).all() or (effort < 0).any():
            raise ValueError('Prediction effort must be finite and nonnegative')
        rate, rate_variance = lognormal_moments(mean, variance)
        return dict(rate_mean=effort * rate, rate_variance=effort.square() * rate_variance)

    def fingerprint(self):
        return fingerprint(self.areas.cpu().numpy(), self.effort.cpu().numpy(),
                           metadata=dict(likelihood=self.name))
