"""Shared latent factors reusing Needlepoint's structured GP implementations."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from ..gpytorch_global_local import _BatchedCachedVNNGP, load_or_build_vnngp_geometry
from ..gpytorch_joint_global_local import _validate_joint_settings
from ..gpytorch_vski import _BatchedGridSVGP, _coordinate_grid_bounds
from ..non_gaussian.likelihoods import lognormal_moments


@dataclass
class SpatialFactorPrediction:
    """Pointwise variational moments, in caller location and feature order.

    Components describe the *Gaussian* spatial factors before exponentiation.
    Factor moments and reconstruction include the exponentiation uncertainty.
    These are mean-field variational moments, not calibrated credible bounds.
    """

    factor_mean: np.ndarray
    factor_variance: np.ndarray
    predictor_mean: np.ndarray
    predictor_variance: np.ndarray
    observation_mean: np.ndarray
    observation_variance: np.ndarray
    log_factor_mean: np.ndarray
    log_factor_variance: np.ndarray
    global_mean: np.ndarray
    global_variance: np.ndarray
    local_mean: np.ndarray
    local_variance: np.ndarray


def factor_moments(mean, variance, nonnegative):
    return lognormal_moments(mean, variance) if nonnegative else (mean, variance)


def _finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {name}; rescale observations or reduce learning rates")
    return value


def _selection(value, size, name):
    result = np.arange(size) if value is None else np.asarray(value)
    if (result.ndim != 1 or not len(result) or result.dtype.kind not in "iu"
            or np.any(result < 0) or np.any(result >= size)):
        raise ValueError(f"{name} must be a nonempty vector of valid indices")
    return result


class GlobalLocalFactorModel(nn.Module):
    """Independent Global V-SKI + local VNNGP priors for each spatial factor.

    q(global), q(local), and q(nonspatial) are mean-field. The likelihood joins
    all factors through a shared feature loading matrix. Nonspatial factors
    have factor-specific Gaussian priors and no spatial interpolation. Prior
    location and scale are learned by default, following the NSFH author code.
    """

    def __init__(self, coordinates, n_features, settings, gp_settings, observation):
        import gpytorch

        super().__init__()
        _validate_joint_settings(gp_settings)
        if gp_settings.global_variational_covariance != "meanfield":
            raise ValueError("Spatial factor fitting currently requires meanfield GP covariance")
        if gp_settings.dtype not in {"float32", "float64"}:
            raise ValueError("Spatial factors require float32 or float64")
        if any(gp_settings.global_grid_phase):
            raise ValueError("Spatial factor fitting does not support a shifted global grid")
        coordinates = np.asarray(coordinates, float)
        if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2
                or not np.isfinite(coordinates).all()):
            raise ValueError("coordinates must be a finite N x 2 matrix, N >= 2")
        self.settings, self.gp_settings = settings, gp_settings
        self.n_features = int(n_features)
        self.n_spatial = settings.n_spatial_factors
        self.n_factors = self.n_spatial + settings.n_nonspatial_factors
        self.n_training = len(coordinates)
        if observation.requires_nonnegative and not settings.nonnegative:
            raise ValueError("Count observations require nonnegative factors and loadings")
        order, dag, self.geometry_metadata = load_or_build_vnngp_geometry(
            coordinates, gp_settings.local_neighbors,
            candidate_multiplier=gp_settings.predecessor_candidate_multiplier,
            query_batch_size=gp_settings.neighbor_query_batch_size,
            cache_dir=gp_settings.geometry_cache_dir, cache_mode=gp_settings.geometry_cache_mode)
        self.dtype = torch.float64 if gp_settings.dtype == "float64" else torch.float32
        self.device = torch.device(gp_settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.register_buffer("native_coordinates", torch.tensor(coordinates, dtype=torch.float64, device=self.device))
        self.register_buffer("coordinates", torch.tensor(coordinates, dtype=self.dtype, device=self.device))
        self.register_buffer("order", torch.tensor(order, dtype=torch.long, device=self.device))
        self.register_buffer("inverse_order", torch.argsort(self.order))
        ordered = self.coordinates[self.order]
        self.grid_bounds = _coordinate_grid_bounds(coordinates)
        self.global_model = _BatchedGridSVGP.build(
            gpytorch, torch, self.n_spatial, gp_settings.global_grid_size,
            self.grid_bounds, "meanfield", kl_backend="structured_kron",
            prior_jitter=gp_settings.global_prior_jitter,
            min_length_scale=gp_settings.global_min_length_scale,
            grid_eigensolver=gp_settings.global_grid_eigensolver,
            reuse_identical_axes=gp_settings.global_reuse_identical_axes).to(
                device=self.device, dtype=self.dtype)
        self.local_model = _BatchedCachedVNNGP.build(
            gpytorch, torch, ordered, torch.tensor(dag, dtype=torch.long, device=self.device),
            self.n_spatial, gp_settings.local_neighbors, gp_settings.batch_size_spots,
            gp_settings.local_min_length_scale, gp_settings.local_max_length_scale).to(
                device=self.device, dtype=self.dtype)
        for prefix, model, initial_length, initial_scale in (
            ("global", self.global_model, gp_settings.global_initial_length_scale, 0.5),
            ("local", self.local_model, gp_settings.local_initial_length_scale, 0.25),
        ):
            parameters = gp_settings.fixed_kernel_parameters
            model.covar_module.base_kernel.lengthscale = (
                initial_length if parameters is None else parameters[f"{prefix}_length"])
            model.covar_module.outputscale = (
                initial_scale if parameters is None else parameters[f"{prefix}_scale"])
            model.covar_module.requires_grad_(settings.learn_kernel and parameters is None)
        if gp_settings.fixed_local_length_scale:
            self.local_model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
        shape = (self.n_training, settings.n_nonspatial_factors)
        self.nonspatial_prior_mean = nn.Parameter(torch.zeros(
            settings.n_nonspatial_factors, dtype=self.dtype, device=self.device),
            requires_grad=settings.learn_nonspatial_prior)
        self.nonspatial_prior_log_sd = nn.Parameter(torch.full(
            (settings.n_nonspatial_factors,), np.log(settings.nonspatial_prior_sd),
            dtype=self.dtype, device=self.device), requires_grad=settings.learn_nonspatial_prior)
        self.nonspatial_mean = nn.Parameter(torch.zeros(shape, dtype=self.dtype, device=self.device))
        self.nonspatial_log_sd = nn.Parameter(torch.full(shape, -3.0, dtype=self.dtype, device=self.device))
        self.raw_loadings = nn.Parameter(torch.zeros(
            (self.n_features, self.n_factors), dtype=self.dtype, device=self.device))
        self.observation = observation.to(device=self.device, dtype=self.dtype)
        # Initialize explicitly: GridInterpolation's default initialization
        # otherwise overwrites data-derived values on its first __call__.
        with torch.no_grad():
            self.global_model.mean_module.constant.zero_()
            for model in (self.global_model, self.local_model):
                strategy = model.variational_strategy
                strategy._variational_distribution.variational_mean.zero_()
                strategy._variational_distribution._variational_stddev.fill_(0.05)
                strategy.variational_params_initialized.fill_(1)
        if settings.variational_scale_parameterization == "softplus":
            from .parameterizations import parameterize_variational_scale
            for model in (self.global_model, self.local_model):
                parameterize_variational_scale(model.variational_strategy._variational_distribution)

    def loadings(self, features=slice(None)):
        raw = self.raw_loadings[features]
        return torch.nn.functional.softplus(raw) if self.settings.nonnegative else raw

    def training_moments(self, indices):
        """Indices are always in original observation order, including repeats."""
        distribution = self.global_model(self.coordinates[indices])
        gm, gv = distribution.mean.T, distribution.variance.T
        lm, lv = self.local_model.variational_strategy.training_marginals(self.inverse_order[indices])
        lm, lv = lm.T, lv.T
        mean = torch.cat((gm + lm, self.nonspatial_mean[indices]), -1)
        variance = torch.cat((gv + lv, (2 * self.nonspatial_log_sd[indices]).exp()), -1)
        return mean, variance, (gm, gv, lm, lv)

    def query_moments(self, coordinates):
        x = torch.as_tensor(coordinates, dtype=self.dtype, device=self.device)
        if x.ndim != 2 or x.shape[1] != 2 or not torch.isfinite(x).all():
            raise ValueError("Query coordinates must be finite Q x 2")
        for dim, (low, high) in enumerate(self.grid_bounds):
            if torch.any((x[:, dim] < low) | (x[:, dim] > high)):
                raise ValueError("Query coordinates must lie within the fitted coordinate bounds")
        strategy = self.local_model.variational_strategy
        if strategy.nn_util.train_n is None:
            # The cached predecessor DAG bypasses GPyTorch's normal neighbor
            # initialization. Its separate prediction index must be populated.
            strategy.nn_util.set_nn_idx(self.coordinates[self.order].float())
        g, local = self.global_model(x), self.local_model(x)
        gm, gv, lm, lv = g.mean.T, g.variance.T, local.mean.T, local.variance.T
        # New observations have no inferred nonspatial state. Integrate its
        # prior instead of interpolating fitted nonspatial observation effects.
        shape = (len(x), self.settings.n_nonspatial_factors)
        mean = torch.cat((gm + lm, self.nonspatial_prior_mean.expand(shape)), -1)
        variance = torch.cat((gv + lv, (2 * self.nonspatial_prior_log_sd).exp().expand(shape)), -1)
        return mean, variance, (gm, gv, lm, lv)

    def sample_factors(self, mean, variance, n_samples, generator):
        if torch.any(variance < 0):
            raise FloatingPointError("Negative factor variance")
        noise = torch.randn((n_samples, *mean.shape), device=mean.device,
                            dtype=mean.dtype, generator=generator)
        sample = mean + variance.sqrt() * noise
        return _finite("factor samples", sample.exp() if self.settings.nonnegative else sample)

    def kl_terms(self, indices, generator=None):
        global_kl = self.global_model.variational_strategy.kl_divergence().sum()
        local_kl = self.local_model.variational_strategy.local_kl(
            mode=self.gp_settings.local_kl_mode, batch_size=self.gp_settings.local_kl_batch_size,
            generator=generator).sum()
        h_mean, h_log_sd = self.nonspatial_mean[indices], self.nonspatial_log_sd[indices]
        prior_var = (2 * self.nonspatial_prior_log_sd).exp()
        h_kl = 0.5 * (((h_mean - self.nonspatial_prior_mean).square() + (2 * h_log_sd).exp()) / prior_var
                      - 1 + 2 * self.nonspatial_prior_log_sd - 2 * h_log_sd).sum()
        return global_kl, local_kl, h_kl

    @torch.no_grad()
    def predict(self, coordinates=None, *, indices=None, exposure=1.0,
                threshold=None, features=None, n_samples=256, random_state=0):
        """Pointwise moments; training indices use exact VNNGP q marginals.

        Default predicts at all training observations in original order. For
        new coordinates, nonspatial factors are integrated under their prior.
        Tobit moments integrate the conditional censored normal analytically
        and the factor posterior by Monte Carlo (``n_samples``).
        Output arrays are materialized; select features or pass location chunks
        when the observation-by-feature prediction would be too large.
        """
        from .fit import AlignedArray

        if coordinates is not None and indices is not None:
            raise ValueError("Supply coordinates or training indices, not both")
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        if coordinates is None:
            indices = _selection(indices, self.n_training, "indices")
            n_rows = len(indices)
        else:
            coordinates = np.asarray(coordinates, float)
            if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
                raise ValueError("Query coordinates must be finite Q x 2")
            n_rows = len(coordinates)
        if n_rows == 0:
            raise ValueError("At least one prediction location is required")
        features = _selection(features, self.n_features, "features")
        exp_array = AlignedArray(exposure, (n_rows, len(features)), "exposure", nonnegative=True)
        if not self.observation.uses_exposure and np.any(exp_array.value != 1):
            raise ValueError("This likelihood does not use exposure")
        thresholds = (None if threshold is None else AlignedArray(
            threshold, (n_rows, len(features)), "threshold"))
        if self.observation.requires_censoring and thresholds is None:
            raise ValueError("Tobit prediction requires threshold")
        generator = torch.Generator(device=self.device).manual_seed(random_state)
        was_training = self.training
        self.eval()
        chunks = {key: [] for key in SpatialFactorPrediction.__dataclass_fields__}
        feature_tensor = torch.tensor(features, dtype=torch.long, device=self.device)
        weights = self.loadings(feature_tensor)
        try:
            for start in range(0, n_rows, self.gp_settings.eval_batch_size_spots):
                rows = np.arange(start, min(n_rows, start + self.gp_settings.eval_batch_size_spots))
                if coordinates is None:
                    idx = torch.tensor(indices[rows], dtype=torch.long, device=self.device)
                    mean, variance, parts = self.training_moments(idx)
                else:
                    mean, variance, parts = self.query_moments(coordinates[rows])
                fm, fv = factor_moments(mean, variance, self.settings.nonnegative)
                mu, var = fm @ weights.T, fv @ weights.square().T
                exp_batch = exp_array.tensor(rows, slice(None), self)
                threshold_batch = None if thresholds is None else thresholds.tensor(rows, slice(None), self)
                samples = None
                if self.observation.requires_censoring:
                    samples = self.sample_factors(mean, variance, n_samples, generator) @ weights.T
                om, ov = self.observation.moments(mu, var, features=feature_tensor,
                    exposure=exp_batch, threshold=threshold_batch, samples=samples)
                values = (fm, fv, mu, var, om, ov, mean, variance, *parts)
                for key, value in zip(chunks, values):
                    chunks[key].append(_finite(key, value).cpu().numpy())
        finally:
            self.train(was_training)
        return SpatialFactorPrediction(**{key: np.concatenate(value) for key, value in chunks.items()})
