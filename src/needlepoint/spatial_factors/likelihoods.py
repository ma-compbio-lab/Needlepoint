"""Observation adapters evaluated *after* shared factors have been mixed.

Predictors have shape [sample, location, feature]; log_prob must return that
same shape. Unlike the per-feature GP adapters, the input is a mixture mean,
not a log rate. No exponential or likelihood term is silently clipped.
"""
import math

import numpy as np
import torch
from torch import nn


class FactorObservation(nn.Module):
    """Subclass to add likelihoods without changing spatial inference.

    Implement validate (NumPy data blocks), log_prob (differentiable Torch),
    and moments (predictive first two moments). Feature-specific parameters
    must be indexed by ``features``. Config enables built-in checkpoint reload;
    custom adapters can instead be supplied explicitly when loading.
    """

    name = "custom"
    uses_exposure = False
    requires_nonnegative = False
    requires_censoring = False

    def validate(self, values, *, exposure, is_censored=None, threshold=None):
        if not np.isfinite(values).all():
            raise ValueError("Observations must be finite")

    def log_prob(self, values, predictor, *, features, exposure=1.0,
                 is_censored=None, threshold=None):
        raise NotImplementedError

    def moments(self, mean, variance, *, features, exposure=1.0, threshold=None,
                samples=None):
        raise NotImplementedError

    def config(self):
        return {"name": self.name}


class PoissonObservation(FactorObservation):
    name = "poisson"
    uses_exposure = True
    requires_nonnegative = True

    def validate(self, values, *, exposure, **kwargs):
        super().validate(values, exposure=exposure)
        if np.any(values < 0) or np.any(values != np.rint(values)):
            raise ValueError("Count likelihoods require nonnegative raw integer counts")
        if np.any((exposure == 0) & (values > 0)):
            raise ValueError("Positive counts require positive exposure")

    def log_prob(self, values, predictor, *, features, exposure=1.0, **kwargs):
        exposure = torch.as_tensor(exposure, dtype=predictor.dtype, device=predictor.device)
        positive = exposure > 0
        # Mask before taking log: a zero-exposure/zero-count entry is exactly
        # deterministic and contributes zero, with zero predictor gradient.
        safe_exposure = torch.where(positive, exposure, 1.0)
        return torch.where(positive,
            values * (predictor.log() + safe_exposure.log())
            - exposure * predictor - torch.lgamma(values + 1), 0.0)

    def moments(self, mean, variance, *, features, exposure=1.0, **kwargs):
        exposure = torch.as_tensor(exposure, dtype=mean.dtype, device=mean.device)
        mu = exposure * mean
        return mu, mu + exposure.square() * variance


class NegativeBinomialObservation(PoissonObservation):
    """NB2: variance conditional on mean mu is mu + mu**2 / dispersion."""

    name = "negative-binomial"

    def __init__(self, n_features, dispersion=10.0, *, learn_dispersion=True):
        super().__init__()
        value = np.broadcast_to(np.asarray(dispersion, float), (n_features,)).copy()
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("dispersion must be finite and positive")
        self.log_dispersion = nn.Parameter(torch.tensor(value).log(),
                                          requires_grad=learn_dispersion)

    def log_prob(self, values, predictor, *, features, exposure=1.0, **kwargs):
        exposure = torch.as_tensor(exposure, dtype=predictor.dtype, device=predictor.device)
        theta = self.log_dispersion[features].exp()
        positive = exposure > 0
        safe_exposure = torch.where(positive, exposure, 1.0)
        log_ratio = predictor.log() + safe_exposure.log() - theta.log()
        # log(sigmoid) remains stable for both tiny and large means.
        result = (torch.lgamma(values + theta) - torch.lgamma(theta)
                  - torch.lgamma(values + 1)
                  - theta * torch.nn.functional.softplus(log_ratio)
                  - values * torch.nn.functional.softplus(-log_ratio))
        return torch.where(positive, result, 0.0)

    def moments(self, mean, variance, *, features, exposure=1.0, **kwargs):
        exposure = torch.as_tensor(exposure, dtype=mean.dtype, device=mean.device)
        mu, latent_var = exposure * mean, exposure.square() * variance
        theta = self.log_dispersion[features].exp()
        return mu, mu + latent_var + (latent_var + mu.square()) / theta

    def config(self):
        return dict(name=self.name, dispersion=self.log_dispersion.detach().exp().cpu().tolist(),
                    learn_dispersion=self.log_dispersion.requires_grad)


class GaussianObservation(FactorObservation):
    name = "gaussian"

    def __init__(self, n_features, noise_sd=1.0, *, learn_noise=True):
        super().__init__()
        value = np.broadcast_to(np.asarray(noise_sd, float), (n_features,)).copy()
        if not np.isfinite(value).all() or np.any(value <= 0):
            raise ValueError("noise_sd must be finite and positive")
        self.log_noise_sd = nn.Parameter(torch.tensor(value).log(), requires_grad=learn_noise)

    def log_prob(self, values, predictor, *, features, **kwargs):
        log_sd = self.log_noise_sd[features]
        return (-0.5 * math.log(2 * math.pi) - log_sd
                - 0.5 * ((values - predictor) * (-log_sd).exp()).square())

    def moments(self, mean, variance, *, features, **kwargs):
        return mean, variance + (2 * self.log_noise_sd[features]).exp()

    def config(self):
        return dict(name=self.name, noise_sd=self.log_noise_sd.detach().exp().cpu().tolist(),
                    learn_noise=self.log_noise_sd.requires_grad)


class TobitObservation(GaussianObservation):
    """Left censoring with explicit flags, thresholds and observed zero codes.

    Observed Y is Z if Z >= threshold and zero otherwise. This is the same
    observation convention as Needlepoint's per-feature Tobit model.
    """

    name = "tobit"
    requires_censoring = True

    def validate(self, values, *, exposure, is_censored=None, threshold=None):
        super().validate(values, exposure=exposure)
        if is_censored is None or threshold is None:
            raise ValueError("Tobit requires explicit is_censored and threshold")
        if np.any(values[is_censored] != 0):
            raise ValueError("Censored values must use the declared zero code")
        if np.any(values[~is_censored] < threshold[~is_censored]):
            raise ValueError("Uncensored observations must be at or above threshold")

    def log_prob(self, values, predictor, *, features, is_censored=None,
                 threshold=None, **kwargs):
        uncensored = super().log_prob(values, predictor, features=features)
        a = (threshold - predictor) * (-self.log_noise_sd[features]).exp()
        return torch.where(is_censored, torch.special.log_ndtr(a), uncensored)

    def moments(self, mean, variance, *, features, threshold=None, samples=None, exposure=1.0):
        if threshold is None or samples is None:
            raise ValueError("Tobit predictive moments require threshold and posterior samples")
        sd = self.log_noise_sd[features].exp()
        a = (samples - threshold) / sd
        prob = torch.special.ndtr(a)
        density = torch.exp(-0.5 * a.square()) / math.sqrt(2 * math.pi)
        first = (samples * prob + sd * density).mean(0)
        second = ((samples.square() + sd.square()) * prob
                  + sd * (samples + threshold) * density).mean(0)
        return first, (second - first.square()).clamp_min(0)


def make_observation(value, n_features):
    if isinstance(value, FactorObservation):
        for name in ("log_noise_sd", "log_dispersion"):
            parameter = getattr(value, name, None)
            if parameter is not None and parameter.shape != (n_features,):
                raise ValueError(f"{name} must match the number of features")
        return value
    options = {"name": value} if isinstance(value, str) else dict(value)
    name = options.pop("name")
    factories = {"poisson": PoissonObservation, "negative-binomial": NegativeBinomialObservation,
                 "gaussian": GaussianObservation, "tobit": TobitObservation}
    if name not in factories:
        raise ValueError(f"Unknown factor likelihood {name!r}; supply a FactorObservation adapter")
    return (PoissonObservation(**options) if name == "poisson"
            else factories[name](n_features, **options))
