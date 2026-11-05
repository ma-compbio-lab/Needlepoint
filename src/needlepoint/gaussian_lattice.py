"""Experimental Gaussian conditioning on an acquired rectangular lattice.

The global RBF plus radial Matérn-3/2 covariance is evaluated on the native
pixels. FFT products exploit a block Toeplitz matrix, without interpolating or
making a diagonal/factorized variational posterior. CG/SLQ and spatial diagonal
probing are numerical approximations. Probing is checked against direct solves
and refined when necessary. This path
is opt-in, Gaussian-only, and requires complete rectangular prediction support.
"""

from __future__ import annotations

import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from needlepoint.storage import GPStorageSettings, encode_gp_maps


@dataclass
class GaussianLatticeSettings(GPStorageSettings):
    output_dtype: str = "float32"
    retain_predictive_variance: bool = True
    n_epochs: int = 100
    learning_rate: float = 0.05
    global_min_length_scale: float = 40.0
    global_initial_length_scale: float = 80.0
    local_min_length_scale: float = 8.0
    local_initial_length_scale: float = 16.0
    local_max_length_scale: float = 32.0
    peak_chunk_size: int = 2
    eval_batch_size_spots: int = 4096
    cg_tolerance: float = 1.0e-4
    max_cg_iterations: int = 1000
    trace_samples: int = 16
    lanczos_iterations: int = 64
    variance_probe_stride: int = 32
    variance_max_probe_stride: int = 128
    variance_probe_batch_size: int = 64
    variance_check_spots: int = 128
    variance_check_relative_tolerance: float = 0.01
    max_cholesky_size: int = 0
    dtype: str = "float32"
    device: str | None = None
    random_state: int = 0

    def validate(self):
        positive = (self.learning_rate, self.cg_tolerance, self.global_min_length_scale,
                    self.global_initial_length_scale, self.local_min_length_scale,
                    self.local_initial_length_scale, self.local_max_length_scale,
                    self.variance_check_relative_tolerance)
        if not np.isfinite(positive).all() or min(positive) <= 0:
            raise ValueError("Rates, tolerance and length scale bounds must be positive")
        if self.global_initial_length_scale <= self.global_min_length_scale:
            raise ValueError("Global initialization must be above its lower bound")
        if not self.local_min_length_scale < self.local_initial_length_scale < self.local_max_length_scale:
            raise ValueError("Local initialization must be inside its bounds")
        if min(self.n_epochs, self.peak_chunk_size, self.eval_batch_size_spots,
               self.max_cg_iterations, self.trace_samples, self.lanczos_iterations,
               self.variance_probe_stride, self.variance_probe_batch_size,
               self.variance_check_spots) <= 0:
            raise ValueError("Epochs, batch sizes, probe counts and ranks must be positive")
        if self.dtype not in {"float32", "float64"} or self.max_cholesky_size < 0:
            raise ValueError("Invalid dtype or Cholesky threshold")
        if self.retain_component_maps or self.retain_latent_maps:
            raise ValueError("This backend exports the coupled total Gaussian field only")
        if self.variance_max_probe_stride < self.variance_probe_stride:
            raise ValueError("Maximum probing stride must be at least the initial stride")


def lattice_geometry(coords):
    """Validate full native support and return ascending axes and row indices."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError("Prediction coordinates must be finite n x 2")
    axes = [np.unique(coords[:, dim]) for dim in range(2)]
    for axis in axes:
        if len(axis) < 2 or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1.e-5, atol=1.e-6):
            raise ValueError("Prediction support must be a uniform rectangular lattice")
    nx, ny = map(len, axes)
    ix, iy = [np.searchsorted(axis, coords[:, dim]) for dim, axis in enumerate(axes)]
    rows = iy * nx + ix
    if len(rows) != nx * ny or len(np.unique(rows)) != nx * ny:
        raise ValueError("Prediction support must contain every lattice pixel exactly once")
    return axes, rows


def _fft_operator_class():
    import torch
    from linear_operator.operators import LinearOperator
    from scipy.fft import next_fast_len

    class BlockToeplitz2D(LinearOperator):
        """Exact native-grid matrix products via symmetric circulant embedding."""

        def __init__(self, first_column):
            super().__init__(first_column)
            self.first_column = first_column
            self.ny, self.nx = first_column.shape[-2:]

        def _size(self):
            n = self.ny * self.nx
            return torch.Size((*self.first_column.shape[:-2], n, n))

        def _transpose_nonbatch(self):
            return self

        def _diagonal(self):
            return self.first_column[..., 0, 0, None].expand(*self.batch_shape, self.ny * self.nx)

        def _get_indices(self, row_index, col_index, *batch_indices):
            dy = (row_index // self.nx - col_index // self.nx).abs()
            dx = (row_index % self.nx - col_index % self.nx).abs()
            return self.first_column[(*batch_indices, dy, dx)]

        def _matmul(self, rhs):
            ny, nx = self.ny, self.nx
            shape = (next_fast_len(2 * ny - 1), next_fast_len(2 * nx - 1))
            iy = torch.arange(shape[0], device=rhs.device)
            ix = torch.arange(shape[1], device=rhs.device)
            iy, ix = torch.minimum(iy, shape[0] - iy), torch.minimum(ix, shape[1] - ix)
            valid = (iy < ny).unsqueeze(-1) & (ix < nx).unsqueeze(0)
            iy, ix = iy.clamp_max(ny - 1), ix.clamp_max(nx - 1)
            embedding = self.first_column.index_select(-2, iy).index_select(-1, ix)
            embedding = embedding * valid.to(embedding)
            padded = torch.nn.functional.pad(
                rhs.reshape(*rhs.shape[:-2], ny, nx, rhs.shape[-1]),
                (0, 0, 0, shape[1] - nx, 0, shape[0] - ny),
            )
            spectrum = torch.fft.rfft2(embedding, s=shape)
            transformed = torch.fft.rfft2(padded, s=shape, dim=(-3, -2))
            result = torch.fft.irfft2(spectrum.unsqueeze(-1) * transformed,
                                     s=shape, dim=(-3, -2))
            return result[..., :ny, :nx, :].reshape(*result.shape[:-3], ny * nx, rhs.shape[-1])

    return BlockToeplitz2D


def _build_model(x_train, y_train, full_coords, settings):
    import torch
    import gpytorch
    from linear_operator.operators import InterpolatedLinearOperator
    from gpytorch.models.exact_prediction_strategies import InterpolatedPredictionStrategy

    axes, _ = lattice_geometry(full_coords.detach().cpu().numpy())
    nx, ny = map(len, axes)
    batch_shape = torch.Size([y_train.shape[0]])
    operator_class = _fft_operator_class()
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        batch_shape=batch_shape, noise_constraint=gpytorch.constraints.GreaterThan(1.e-5)
    ).to(x_train)

    class LatticeKernel(gpytorch.kernels.Kernel):
        def __init__(self):
            super().__init__(batch_shape=batch_shape)
            self.global_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=batch_shape,
                    lengthscale_constraint=gpytorch.constraints.GreaterThan(settings.global_min_length_scale)),
                batch_shape=batch_shape)
            self.local_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=batch_shape,
                    lengthscale_constraint=gpytorch.constraints.Interval(settings.local_min_length_scale,
                                                                        settings.local_max_length_scale)),
                batch_shape=batch_shape)
            self.register_buffer("origin", full_coords.new_tensor([axis[0] for axis in axes]))
            self.register_buffer("pitch", full_coords.new_tensor([axis[1] - axis[0] for axis in axes]))
            xx, yy = torch.meshgrid(torch.arange(nx, device=full_coords.device),
                                    torch.arange(ny, device=full_coords.device), indexing="xy")
            offsets = torch.stack((xx.flatten(), yy.flatten()), -1).to(full_coords) * self.pitch
            self.register_buffer("offsets", offsets)

        def indices(self, x):
            positions = (x - self.origin) / self.pitch
            rounded = positions.round()
            if not torch.allclose(positions, rounded, rtol=0, atol=2.e-4):
                raise ValueError("All training and prediction sites must lie on the native lattice")
            ix, iy = rounded.long().unbind(-1)
            if ((ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)).any():
                raise ValueError("Coordinates fall outside the native lattice")
            return (iy * nx + ix).unsqueeze(-1)

        def forward(self, x1, x2, diag=False, **params):
            zero = self.offsets[:1] * 0
            column = (self.global_kernel(zero, self.offsets).to_dense()
                      + self.local_kernel(zero, self.offsets).to_dense()).reshape(*batch_shape, ny, nx)
            base = operator_class(column)
            left, right = self.indices(x1), self.indices(x2)
            left = left.expand(*batch_shape, *left.shape[-2:])
            right = right.expand(*batch_shape, *right.shape[-2:])
            # One-hot selection is exact at acquired pixels; there is no cubic interpolation.
            cov = InterpolatedLinearOperator(base, left, torch.ones_like(left, dtype=x1.dtype),
                                             right, torch.ones_like(right, dtype=x2.dtype))
            return cov.diagonal() if diag else cov

        def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
            return InterpolatedPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)

    class LatticeGP(gpytorch.models.ExactGP):
        def __init__(self):
            super().__init__(x_train, y_train, likelihood)
            self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
            self.covar_module = LatticeKernel()

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    model = LatticeGP().to(x_train)
    model.covar_module.global_kernel.base_kernel.lengthscale = settings.global_initial_length_scale
    model.covar_module.local_kernel.base_kernel.lengthscale = settings.local_initial_length_scale
    model.covar_module.global_kernel.outputscale = 0.5
    model.covar_module.local_kernel.outputscale = 0.5
    likelihood.noise = 0.5
    return model, likelihood


def _numerical_context(settings):
    import gpytorch
    stack = ExitStack()
    contexts = (
        gpytorch.settings.max_cholesky_size(settings.max_cholesky_size),
        gpytorch.settings.cg_tolerance(settings.cg_tolerance),
        gpytorch.settings.eval_cg_tolerance(settings.cg_tolerance),
        gpytorch.settings.max_cg_iterations(settings.max_cg_iterations),
        gpytorch.settings.max_preconditioner_size(0),
        gpytorch.settings.num_trace_samples(settings.trace_samples),
        gpytorch.settings.max_lanczos_quadrature_iterations(settings.lanczos_iterations),
        gpytorch.settings.fast_pred_var(False),
    )
    for context in contexts:
        stack.enter_context(context)
    return stack


def _posterior_variance(model, likelihood, full_coords, training_rows, settings):
    """Estimate diag(C) by spatial coloring and independently check sampled sites.

    For a probe z supported on one spatial color, (Cz)[i] estimates C[i,i]
    at sites of that color. Only distant same-color covariances contaminate it.
    Increasing the stride separates those sites. This is an explicit numerical
    approximation, not a variational distribution or smoothing of an SD image.
    """
    import torch
    axes, lattice_rows = lattice_geometry(full_coords.detach().cpu().numpy())
    nx, ny = map(len, axes)
    ix, iy = lattice_rows % nx, lattice_rows // nx
    train_rows = np.asarray(training_rows, dtype=np.int64)
    training_coords = full_coords[train_rows]
    numerical = replace(settings, cg_tolerance=min(settings.cg_tolerance, 1.e-5))
    kernel = model.covar_module
    rng = np.random.default_rng(settings.random_state + 1947)
    n_check = settings.variance_check_spots
    groups = (np.arange(len(full_coords)), train_rows,
              np.setdiff1d(np.arange(len(full_coords)), train_rows),
              np.flatnonzero((ix == 0) | (ix == nx - 1) | (iy == 0) | (iy == ny - 1)))
    selected = [rng.choice(group, min(max(n_check // 4, 1), len(group)), replace=False)
                for group in groups if len(group)]
    with torch.no_grad(), _numerical_context(numerical):
        a = likelihood(model.forward(training_coords)).lazy_covariance_matrix.evaluate_kernel()
        full_k = kernel(full_coords).evaluate_kernel()
        cross_op = kernel(full_coords, training_coords).evaluate_kernel()
        stride = settings.variance_probe_stride
        attempts = []
        reference = None
        while stride <= settings.variance_max_probe_stride:
            colors = (iy % stride) * stride + (ix % stride)
            present = np.unique(colors)
            variance = np.empty((len(full_coords), *kernel.batch_shape), dtype=np.float32)
            for first in range(0, len(present), settings.variance_probe_batch_size):
                batch_colors = present[first:first + settings.variance_probe_batch_size]
                z = torch.as_tensor(colors[:, None] == batch_colors, dtype=full_coords.dtype,
                                    device=full_coords.device)
                kz = full_k @ z
                solved = a.solve(kz[:, train_rows, :])
                cz = kz - cross_op @ solved
                rows = np.flatnonzero(np.isin(colors, batch_colors))
                columns = np.searchsorted(batch_colors, colors[rows])
                variance[rows] = cz[:, rows, columns].T.cpu().numpy()
            if reference is None:
                # Also check high-uncertainty pixels so small spatial gaps are
                # not missed by the uniform/observed/withheld/boundary samples.
                selected.extend(np.argsort(variance[:, j])[-max(n_check // 4, 1):]
                                for j in range(variance.shape[1]))
                check_rows = np.unique(np.concatenate(selected))
                reference_parts, residuals = [], []
                prior_diag = kernel.global_kernel.outputscale + kernel.local_kernel.outputscale
                for first in range(0, len(check_rows), 32):
                    rows = check_rows[first:first + 32]
                    cross = (kernel.global_kernel(training_coords, full_coords[rows]).to_dense()
                             + kernel.local_kernel(training_coords, full_coords[rows]).to_dense())
                    solved = a.solve(cross)
                    residuals.append(((a @ solved - cross).norm(dim=-2)
                                      / cross.norm(dim=-2).clamp_min(1.e-20)).cpu().numpy())
                    reference_parts.append((prior_diag[:, None] - (cross * solved).sum(-2)).T.cpu().numpy())
                reference = np.concatenate(reference_parts)
                if not np.isfinite(reference).all() or np.any(reference <= 0):
                    raise FloatingPointError("Direct posterior variance reference is nonpositive/nonfinite")
                max_residual = float(np.max(np.concatenate(residuals, axis=-1)))
                if max_residual > max(1.e-4, numerical.cg_tolerance * 10):
                    raise FloatingPointError(f"Variance reference CG residual is too large: {max_residual:g}")
            relative = np.abs(variance[check_rows] - reference) / reference
            maximum = float(np.max(relative))
            attempts.append({"stride": int(stride), "n_probes": len(present),
                             "checked_maximum_relative_error": maximum})
            if np.isfinite(variance).all() and np.all(variance > 0) and maximum <= settings.variance_check_relative_tolerance:
                return variance, {
                    "method": "spatially colored posterior covariance probing",
                    "stride_pixels": int(stride), "n_probes": len(present),
                    "n_direct_check_pixels": len(check_rows),
                    "direct_check_max_cg_relative_residual": max_residual,
                    "checked_relative_error_quantiles": np.quantile(relative, [.5, .95, 1], axis=0).tolist(),
                    "check_row_indices": check_rows.tolist(),
                    "attempts": attempts,
                    "limitation": "Sampled checks bound observed numerical errors, not the unchecked maximum or statistical calibration.",
                }
            stride *= 2
    raise RuntimeError(f"Posterior variance probing failed its direct-solve checks: {attempts}")


def fit_gaussian_lattice_spatial_maps(expression, coords, feature_names, output_dir, *,
                                     settings=None, prediction_coords=None,
                                     prediction_spot_indices=None):
    """Condition a coupled additive Gaussian GP on native lattice observations.

    Standardization uses training pixels only. Prediction coordinates must be
    the full acquired rectangle, but training may omit held-out sites. Saved
    variances describe the total latent field; predictive variance adds noise.
    This experimental backend does not imply empirical uncertainty calibration.
    """
    import torch
    import gpytorch
    from needlepoint.gpytorch_baselines import _finish_outputs, _validate_inputs
    from needlepoint.gpytorch_vski import _choose_device, _standardize_columns_with_stats
    from needlepoint.posterior_blocks import atomic_savez_maps

    settings = settings or GaussianLatticeSettings()
    settings.validate()
    y_raw, coords_np, pred_np, names = _validate_inputs(expression, coords, prediction_coords, feature_names)
    lattice_geometry(pred_np)
    standardized, means, stds = _standardize_columns_with_stats(y_raw)
    device = _choose_device(torch, settings.device)
    dtype = getattr(torch, settings.dtype)
    x = torch.as_tensor(coords_np, device=device, dtype=dtype)
    xp = torch.as_tensor(pred_np, device=device, dtype=dtype)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Gaussian lattice output directory must be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    metadata, chunks, history, variance_audits = [], [], [], []
    lookup = {tuple(row): i for i, row in enumerate(pred_np)}
    try:
        training_rows = np.array([lookup[tuple(row)] for row in coords_np], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("Every training coordinate must be an acquired prediction site") from exc
    if len(np.unique(training_rows)) != len(training_rows):
        raise ValueError("Training pixels must be unique")
    torch.manual_seed(settings.random_state)
    for chunk_id, start in enumerate(range(0, y_raw.shape[1], settings.peak_chunk_size)):
        stop = min(start + settings.peak_chunk_size, y_raw.shape[1])
        y = torch.as_tensor(standardized[:, start:stop].T, device=device, dtype=dtype)
        model, likelihood = _build_model(x, y, xp, settings)
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        # Fixed trace probes stabilize the stochastic MLL while learning. Reset
        # their context per feature chunk so probes are never reused across shapes.
        with _numerical_context(settings), gpytorch.settings.deterministic_probes(True):
            for epoch in range(1, settings.n_epochs + 1):
                model.train()
                likelihood.train()
                optimizer.zero_grad(set_to_none=True)
                loss = -mll(model(x), y).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite Gaussian lattice likelihood")
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite Gaussian lattice gradient")
                optimizer.step()
                history.append({"chunk_id": chunk_id, "epoch": epoch,
                                "mean_negative_objective": float(loss.detach().cpu()) / (stop - start)})
                if epoch == 1 or epoch % 10 == 0 or epoch == settings.n_epochs:
                    print(f"Gaussian lattice chunk {chunk_id} epoch {epoch}/{settings.n_epochs} "
                          f"negative_mll={history[-1]['mean_negative_objective']:.7g}", flush=True)
        model.eval()
        likelihood.eval()
        mean_parts = []
        with torch.no_grad(), _numerical_context(settings), gpytorch.settings.skip_posterior_variances(True):
            for offset in range(0, len(xp), settings.eval_batch_size_spots):
                posterior = model(xp[offset:offset + settings.eval_batch_size_spots])
                mean_parts.append(posterior.mean.T.cpu().numpy())
        mean = np.concatenate(mean_parts)
        variance, variance_audit = _posterior_variance(model, likelihood, xp, training_rows, settings)
        variance_audits.append({"chunk_id": chunk_id, **variance_audit})
        noise = likelihood.noise.detach().cpu().numpy().reshape(-1)
        arrays = encode_gp_maps(settings, dict(posterior_mean=mean, posterior_variance=variance,
                                              posterior_predictive_variance=variance + noise))
        filename = f"vski_gp_maps_chunk_{chunk_id:04d}.npz"
        atomic_savez_maps(output_dir / filename, **arrays, gene=names[start:stop],
                          feature_position=np.arange(start, stop))
        kernel = model.covar_module
        def parameter(tensor):
            return tensor.detach().cpu().numpy().reshape(-1)
        metadata.append(pd.DataFrame(dict(
            gene=names[start:stop], feature_position=np.arange(start, stop),
            standardization_mean=means[start:stop], standardization_std=stds[start:stop],
            learned_global_length_scale=parameter(kernel.global_kernel.base_kernel.lengthscale),
            learned_local_length_scale=parameter(kernel.local_kernel.base_kernel.lengthscale),
            learned_global_outputscale=parameter(kernel.global_kernel.outputscale),
            learned_local_outputscale=parameter(kernel.local_kernel.outputscale),
            learned_noise_variance=noise, map_chunk_id=chunk_id, map_chunk_file=filename,
            map_column=np.arange(stop - start), backend="gaussian_native_lattice_fft")))
        chunks.append(dict(chunk_id=chunk_id, file=filename, n_spots=len(xp),
                           n_training_spots=len(x), n_features=stop - start,
                           feature_start=start, feature_stop=stop))
        torch.save(model.state_dict(), output_dir / f"gaussian_lattice_state_{chunk_id:04d}.pt")
        del model, likelihood, optimizer, mll
        if device.type == "cuda":
            torch.cuda.empty_cache()
    full = len(x) == len(xp) and len(np.unique(coords_np, axis=0)) == len(xp)
    manifest = _finish_outputs(
        output_dir, method="coupled additive Gaussian GP on the native lattice",
        backend="gaussian_native_lattice_fft", settings=settings, coords=coords_np,
        prediction_coords=pred_np, prediction_spot_indices=prediction_spot_indices,
        metadata_records=metadata, chunk_records=chunks, history_records=history,
        elapsed_seconds=time.time() - started,
        extra=dict(observation_model="gaussian", posterior_map_scale="standardized_gaussian",
                   scale="input intensity standardized using training pixels only",
                   model="y = global_RBF + local_radial_Matern32 + Gaussian_noise",
                   fit_purpose="production" if full else "evaluation",
                   downstream_ready=bool(full),
                   full_training_coverage=bool(full), training_coverage_fraction=len(x) / len(xp),
                   n_likelihood_eligible_spots=len(xp),
                   likelihood_coverage_fraction=len(x) / len(xp), full_likelihood_coverage=bool(full),
                   n_prediction_only_spots=len(xp) - len(x),
                   inference_note="Gaussian conditioning; FFT covariance products are exact on native sites. "
                                  "CG/SLQ and spatial variance probing introduce numerical errors; "
                                  "probing is refined and checked against independent direct CG solves.",
                   variance_numerical_audits=variance_audits,
                   optimizer_updates_per_chunk=int(settings.n_epochs),
                   component_covariance="retained implicitly by conditioning the sum kernel",
                   latent_variance="Var(global + local | y), including posterior cross-covariance",
                   uncertainty_calibration="not implied by Gaussian conditioning"))
    spots = pd.read_csv(output_dir / "vski_gp_map_spots.csv.gz")
    used = np.zeros(len(xp), dtype=bool)
    used[training_rows] = True
    spots["used_for_training"] = used
    spots["prediction_only"] = ~used
    spots["likelihood_eligible"] = True
    spots.to_csv(output_dir / "vski_gp_map_spots.csv.gz", index=False)
    return manifest
