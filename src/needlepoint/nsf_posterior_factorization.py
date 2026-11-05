"""Gaussian NSF-style factorization of feature-GP posterior mean maps."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .shared_spatial_factorization import (
    GPPosteriorFactorStore,
    _linked_normal_moments,
    _linked_reconstruction_moments,
    _sample_feature_positions,
    normalize_factor_identifiability,
    scaled_gaussian_log_likelihood,
    spatially_balanced_sample,
)


@dataclass
class NSFPosteriorSettings:
    n_factors: int = 20
    n_inducing: int = 256
    n_epochs: int = 10
    steps_per_epoch: int = 24
    location_batch_size: int = 8192
    feature_batch_size: int = 256
    nmf_sample_spots: int = 20_000
    nmf_max_iter: int = 100
    nmf_shrinkage: float = 0.2
    learning_rate: float = 0.01
    variational_learning_rate: float = 0.01
    hyperparameter_learning_rate: float = 0.005
    initial_length_scale: float = 0.1
    sigma_min: float = 1.0e-4
    variance_min: float = 1.0e-6
    variance_max: float = 1.0e3
    validation_fraction: float = 0.01
    map_chunk_size_spots: int = 32_768
    residual_feature_chunk_size: int = 32
    dtype: str = "float32"
    device: str | None = None
    random_state: int = 0

    def validate(self) -> None:
        positive = {
            "n_factors": self.n_factors,
            "n_inducing": self.n_inducing,
            "n_epochs": self.n_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "location_batch_size": self.location_batch_size,
            "feature_batch_size": self.feature_batch_size,
            "nmf_sample_spots": self.nmf_sample_spots,
        }
        if any(int(value) <= 0 for value in positive.values()):
            raise ValueError(f"positive settings required: {positive}")
        if not 0.0 <= self.nmf_shrinkage < 1.0:
            raise ValueError("nmf_shrinkage must be in [0, 1)")
        if not 0.0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")


def rescale_coordinates(coords: np.ndarray, box_side: float = 4.0):
    """Match the paper's centered, aspect-preserving coordinate scaling."""
    coords = np.asarray(coords, dtype=np.float64)
    minimum = coords.min(axis=0)
    shifted = coords - minimum
    spans = np.maximum(shifted.max(axis=0), 1.0e-12)
    geometric_span = float(np.exp(np.mean(np.log(spans))))
    multiplier = float(box_side) / geometric_span
    scaled = shifted * multiplier
    center = scaled.mean(axis=0)
    return (scaled - center).astype(np.float32), {
        "minimum": minimum.tolist(),
        "center_after_scaling": center.tolist(),
        "multiplier_raw_to_scaled": multiplier,
        "box_side": float(box_side),
    }


def _paper_regularized_nmf(matrix, n_factors, shrinkage, max_iter, random_state):
    from sklearn.decomposition import NMF

    model = NMF(
        n_components=int(n_factors),
        init="nndsvd",
        solver="cd",
        beta_loss="frobenius",
        max_iter=int(max_iter),
        random_state=int(random_state),
    )
    nonnegative_factors = model.fit_transform(matrix)
    loadings = model.components_.T
    if 0.0 < shrinkage < 1.0:
        loading_sums = loadings.sum(axis=0)
        loadings = (
            (1.0 - shrinkage) * loadings
            + shrinkage * loading_sums[None, :] / loadings.shape[0]
        )
        nonnegative_factors *= loading_sums[None, :]
        factor_sums = nonnegative_factors.sum(axis=1, keepdims=True)
        nonnegative_factors = (
            (1.0 - shrinkage) * nonnegative_factors
            + shrinkage * factor_sums / nonnegative_factors.shape[1]
        )
    else:
        loading_sums = loadings.sum(axis=0)
        nonnegative_factors *= loading_sums[None, :]
    loading_sums = np.maximum(loadings.sum(axis=0), 1.0e-12)
    log_factors = np.log(1.0e-2 + nonnegative_factors)
    sigma2 = np.log(2.0 * n_factors) - np.log(n_factors + 1.0)
    prior_intercept = -np.log(n_factors) - sigma2 / 2.0
    transfer = log_factors.mean(axis=0) - prior_intercept
    log_factors -= transfer[None, :]
    loadings *= np.exp(transfer - np.log(loading_sums))[None, :]
    return log_factors.astype(np.float32), loadings.astype(np.float32), model


def _build_nsf_gp(torch, gpytorch, n_factors, inducing_points, initial_length_scale):
    batch_shape = torch.Size([int(n_factors)])
    variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
        inducing_points.shape[-2], batch_shape=batch_shape
    )

    class NSFVariationalGP(gpytorch.models.ApproximateGP):
        def __init__(self):
            strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=False,
                jitter_val=1.0e-4,
            )
            super().__init__(strategy)
            self.mean_module = gpytorch.means.LinearMean(
                input_size=2, batch_shape=batch_shape, bias=True
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=batch_shape),
                batch_shape=batch_shape,
            )
            self.covar_module.base_kernel.lengthscale = float(initial_length_scale)

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    return NSFVariationalGP()


def _softplus_inverse(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 1.0e-8)
    return np.where(values > 20.0, values, np.log(np.expm1(values)))


def fit_nsf_posterior_factorization(
    store: GPPosteriorFactorStore,
    output_dir: Path,
    settings: NSFPosteriorSettings | None = None,
):
    """Fit Gaussian NSF to posterior means and export standard factor outputs."""
    import gpytorch
    import torch
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.linear_model import LinearRegression
    from sklearn.neighbors import KNeighborsRegressor

    settings = settings or NSFPosteriorSettings()
    settings.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = store.prepare()
    started = time.time()
    device = torch.device(
        settings.device
        if settings.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float64 if settings.dtype == "float64" else torch.float32
    torch.manual_seed(settings.random_state)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(settings.random_state)
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(settings.random_state)

    raw_coords = store.spots[["x", "y"]].to_numpy(dtype=np.float32)
    coords, coordinate_transform = rescale_coordinates(raw_coords)
    n_validation = int(round(settings.validation_fraction * store.n_spots))
    validation_rows = (
        spatially_balanced_sample(raw_coords, n_validation)
        if n_validation
        else np.empty(0, dtype=np.int64)
    )
    training_rows = np.setdiff1d(
        np.arange(store.n_spots, dtype=np.int64), validation_rows, assume_unique=True
    )
    local_sample = spatially_balanced_sample(
        raw_coords[training_rows], min(settings.nmf_sample_spots, training_rows.size)
    )
    sample_rows = training_rows[local_sample]
    sample_matrix, _ = store.read_all_features(sample_rows)
    log_factors, loading_init, nmf = _paper_regularized_nmf(
        sample_matrix,
        settings.n_factors,
        settings.nmf_shrinkage,
        settings.nmf_max_iter,
        settings.random_state,
    )
    sample_coords = coords[sample_rows]
    kmeans = MiniBatchKMeans(
        n_clusters=min(settings.n_inducing, sample_rows.size),
        batch_size=min(4096, sample_rows.size),
        random_state=settings.random_state,
        n_init=3,
    ).fit(sample_coords)
    inducing_np = kmeans.cluster_centers_.astype(np.float32)
    inducing = torch.as_tensor(inducing_np, dtype=dtype, device=device)
    gp = _build_nsf_gp(
        torch,
        gpytorch,
        settings.n_factors,
        inducing,
        settings.initial_length_scale,
    ).to(device=device, dtype=dtype)

    linear = LinearRegression().fit(sample_coords, log_factors)
    neighbor_count = max(2, int(math.ceil(sample_rows.size / inducing_np.shape[0])))
    inducing_log_factors = KNeighborsRegressor(n_neighbors=neighbor_count).fit(
        sample_coords, log_factors
    ).predict(inducing_np)
    variational = gp.variational_strategy._variational_distribution
    with torch.no_grad():
        gp.mean_module.bias.copy_(
            torch.as_tensor(linear.intercept_, dtype=dtype, device=device).unsqueeze(-1)
        )
        gp.mean_module.weights.copy_(
            torch.as_tensor(linear.coef_, dtype=dtype, device=device).unsqueeze(-1)
        )
        variational.variational_mean.copy_(
            torch.as_tensor(inducing_log_factors.T, dtype=dtype, device=device)
        )
        identity = torch.eye(inducing_np.shape[0], dtype=dtype, device=device)
        variational.chol_variational_covar.copy_(
            0.1 * identity.expand(settings.n_factors, -1, -1)
        )

    initial_reconstruction = np.exp(log_factors) @ loading_init.T
    sigma_init = np.maximum(
        np.sqrt(np.mean(np.square(sample_matrix - initial_reconstruction), axis=0)),
        settings.sigma_min,
    )

    class FactorParameters(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw_loadings = torch.nn.Parameter(
                torch.as_tensor(_softplus_inverse(loading_init), dtype=dtype, device=device)
            )
            sigma_offset = np.maximum(sigma_init - settings.sigma_min, 1.0e-8)
            self.raw_sigma = torch.nn.Parameter(
                torch.as_tensor(_softplus_inverse(sigma_offset), dtype=dtype, device=device)
            )

        def values(self):
            return (
                torch.nn.functional.softplus(self.raw_loadings),
                torch.nn.functional.softplus(self.raw_sigma) + settings.sigma_min,
            )

    parameters = FactorParameters()
    # The first call converts the initialized unwhitened q(u) to GPyTorch's
    # whitened representation before optimizer construction.
    _ = gp(torch.as_tensor(coords[:1], dtype=dtype, device=device))
    variational_parameters = list(gp.variational_parameters())
    variational_ids = {id(value) for value in variational_parameters}
    hyperparameters = [value for value in gp.parameters() if id(value) not in variational_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": variational_parameters, "lr": settings.variational_learning_rate},
            {"params": hyperparameters, "lr": settings.hyperparameter_learning_rate},
            {"params": parameters.parameters(), "lr": settings.learning_rate},
        ]
    )
    coords_t = torch.as_tensor(coords, dtype=dtype, device=device)
    history = []
    for epoch in range(1, settings.n_epochs + 1):
        epoch_started = time.time()
        gp.train()
        parameters.train()
        totals = {"elbo": 0.0, "likelihood": 0.0, "kl": 0.0}
        for _step in range(settings.steps_per_epoch):
            rows = rng.choice(
                training_rows,
                size=min(settings.location_batch_size, training_rows.size),
                replace=training_rows.size < settings.location_batch_size,
            ).astype(np.int64)
            features = _sample_feature_positions(
                rng, store.n_features, settings.feature_batch_size
            )
            target_np, inherited_np = store.read_block(rows, features)
            target = torch.as_tensor(target_np, dtype=dtype, device=device)
            inherited = torch.as_tensor(inherited_np, dtype=dtype, device=device)
            posterior = gp(coords_t[rows])
            factor_mean, factor_variance = _linked_normal_moments(
                torch, posterior.mean, posterior.variance, "exp"
            )
            loadings, sigma = parameters.values()
            reconstruction, reconstruction_variance = _linked_reconstruction_moments(
                torch,
                factor_mean,
                factor_variance,
                loadings[features],
                torch.zeros(features.size, dtype=dtype, device=device),
            )
            effective_variance = (
                inherited + sigma[features].square()[None, :]
            ).clamp(settings.variance_min, settings.variance_max)
            likelihood = scaled_gaussian_log_likelihood(
                target,
                reconstruction,
                effective_variance,
                n_spots_total=training_rows.size,
                n_features_total=store.n_features,
                reconstruction_variance=reconstruction_variance,
            )
            kl = gp.variational_strategy.kl_divergence().sum()
            elbo = likelihood - kl
            loss = -elbo / (store.n_spots * store.n_features)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(gp.parameters()) + list(parameters.parameters()), 10.0
            )
            optimizer.step()
            totals["elbo"] += float(elbo.detach().cpu())
            totals["likelihood"] += float(likelihood.detach().cpu())
            totals["kl"] += float(kl.detach().cpu())

        validation = _validation_metrics(
            torch,
            gp,
            parameters,
            store,
            coords_t,
            validation_rows,
            settings,
            device,
            dtype,
        )
        record = {
            "epoch": epoch,
            "elbo": totals["elbo"] / settings.steps_per_epoch,
            "likelihood_term": totals["likelihood"] / settings.steps_per_epoch,
            "kl_term": totals["kl"] / settings.steps_per_epoch,
            "validation_rmse": validation["rmse"],
            "validation_weighted_nll": validation["weighted_nll"],
            "epoch_seconds": float(time.time() - epoch_started),
            "length_scale_min_scaled": float(
                gp.covar_module.base_kernel.lengthscale.min().detach().cpu()
            ),
            "length_scale_median_scaled": float(
                gp.covar_module.base_kernel.lengthscale.median().detach().cpu()
            ),
            "length_scale_max_scaled": float(
                gp.covar_module.base_kernel.lengthscale.max().detach().cpu()
            ),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "shared_factor_training_history.csv", index=False)
        _save_checkpoint(torch, output_dir, epoch, gp, parameters, optimizer, history, settings)
        print(
            f"NSF-map epoch {epoch:03d}/{settings.n_epochs} "
            f"val_rmse={record['validation_rmse']:.5f} "
            f"seconds={record['epoch_seconds']:.1f}",
            flush=True,
        )

    maps, variances, map_index = _export_maps(
        torch, gp, coords_t, output_dir, settings
    )
    loadings_t, sigma_t = parameters.values()
    loadings = loadings_t.detach().cpu().numpy().astype(np.float64)
    sigma = sigma_t.detach().cpu().numpy().astype(np.float64)
    normalized_maps, normalized_loadings, mixtures = normalize_factor_identifiability(
        maps, loadings
    )
    _finish_outputs(
        store,
        output_dir,
        settings,
        maps,
        variances,
        map_index,
        normalized_maps,
        loadings,
        normalized_loadings,
        mixtures,
        sigma,
        gp,
        coordinate_transform,
        history,
        cache_manifest,
        started,
        device,
        torch,
        nmf,
    )
    return output_dir


def _validation_metrics(
    torch, gp, parameters, store, coords_t, rows, settings, device, dtype
):
    if rows.size == 0:
        return {"rmse": float("nan"), "weighted_nll": float("nan")}
    rows = rows[: min(rows.size, settings.location_batch_size)]
    features = np.arange(store.n_features, dtype=np.int64)
    target_np, inherited_np = store.read_block(rows, features)
    target = torch.as_tensor(target_np, dtype=dtype, device=device)
    inherited = torch.as_tensor(inherited_np, dtype=dtype, device=device)
    gp.eval()
    parameters.eval()
    with torch.no_grad():
        posterior = gp(coords_t[rows])
        factor_mean, factor_variance = _linked_normal_moments(
            torch, posterior.mean, posterior.variance, "exp"
        )
        loadings, sigma = parameters.values()
        reconstruction, reconstruction_variance = _linked_reconstruction_moments(
            torch,
            factor_mean,
            factor_variance,
            loadings,
            torch.zeros(store.n_features, dtype=dtype, device=device),
        )
        effective_variance = (
            inherited + sigma.square()[None, :]
        ).clamp(settings.variance_min, settings.variance_max)
        rmse = torch.sqrt(torch.mean((target - reconstruction).square()))
        nll = 0.5 * torch.mean(
            target.new_tensor(math.log(2.0 * math.pi))
            + effective_variance.log()
            + ((target - reconstruction).square() + reconstruction_variance)
            / effective_variance
        )
    return {"rmse": float(rmse.cpu()), "weighted_nll": float(nll.cpu())}


def _save_checkpoint(torch, output_dir, epoch, gp, parameters, optimizer, history, settings):
    path = output_dir / "checkpoint_latest.pt"
    temporary = Path(str(path) + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "epoch": epoch,
            "gp_state": gp.state_dict(),
            "factor_parameter_state": parameters.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "settings": asdict(settings),
        },
        temporary,
    )
    os.replace(temporary, path)


def _export_maps(torch, gp, coords_t, output_dir, settings):
    map_dir = output_dir / "shared_factor_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    means = np.empty((coords_t.shape[0], settings.n_factors), dtype=np.float32)
    variances = np.empty_like(means)
    rows = []
    gp.eval()
    with torch.no_grad():
        for chunk, start in enumerate(range(0, coords_t.shape[0], settings.map_chunk_size_spots)):
            stop = min(start + settings.map_chunk_size_spots, coords_t.shape[0])
            posterior = gp(coords_t[start:stop])
            mean, variance = _linked_normal_moments(
                torch, posterior.mean, posterior.variance, "exp"
            )
            means[start:stop] = mean.T.cpu().numpy()
            variances[start:stop] = variance.T.cpu().numpy()
            file_name = f"shared_factor_maps_chunk_{chunk:04d}.npz"
            np.savez_compressed(
                map_dir / file_name,
                posterior_mean=means[start:stop],
                posterior_variance=variances[start:stop],
            )
            rows.append(
                {
                    "chunk_id": chunk,
                    "file": file_name,
                    "spot_start": start,
                    "spot_stop": stop,
                    "n_spots": stop - start,
                    "n_factors": settings.n_factors,
                }
            )
    index = pd.DataFrame(rows)
    index.to_csv(map_dir / "shared_factor_map_chunks.csv", index=False)
    return means, variances, index


def _finish_outputs(
    store,
    output_dir,
    settings,
    maps,
    variances,
    map_index,
    normalized_maps,
    loadings,
    normalized_loadings,
    mixtures,
    sigma,
    gp,
    coordinate_transform,
    history,
    cache_manifest,
    started,
    device,
    torch,
    nmf,
):
    map_dir = output_dir / "shared_factor_maps"
    for row in map_index.itertuples(index=False):
        path = map_dir / row.file
        with np.load(path) as archive:
            mean = np.asarray(archive["posterior_mean"], dtype=np.float32)
            variance = np.asarray(archive["posterior_variance"], dtype=np.float32)
        np.savez_compressed(
            path,
            posterior_mean=mean,
            posterior_variance=variance,
            normalized_mean=normalized_maps[row.spot_start : row.spot_stop].astype(np.float32),
        )
    store.spots.to_csv(output_dir / "shared_factor_spots.csv.gz", index=False)

    residual_sum = np.zeros(store.n_features, dtype=np.float64)
    residual_square = np.zeros(store.n_features, dtype=np.float64)
    residual_absolute = np.zeros(store.n_features, dtype=np.float64)
    for feature_start in range(0, store.n_features, settings.residual_feature_chunk_size):
        feature_stop = min(
            feature_start + settings.residual_feature_chunk_size, store.n_features
        )
        features = np.arange(feature_start, feature_stop, dtype=np.int64)
        for spot_start in range(0, store.n_spots, settings.map_chunk_size_spots):
            spot_stop = min(spot_start + settings.map_chunk_size_spots, store.n_spots)
            rows = np.arange(spot_start, spot_stop, dtype=np.int64)
            target, _ = store.read_block(rows, features)
            residual = target - maps[rows] @ loadings[features].T
            residual_sum[features] += residual.sum(axis=0, dtype=np.float64)
            residual_square[features] += np.square(residual).sum(axis=0, dtype=np.float64)
            residual_absolute[features] += np.abs(residual).sum(axis=0, dtype=np.float64)
    table = store.metadata.copy()
    table["feature_amplitude"] = store.amplitudes
    table["baseline"] = 0.0
    table["feature_specific_sigma"] = sigma
    table["residual_mean"] = residual_sum / store.n_spots
    table["residual_rmse"] = np.sqrt(residual_square / store.n_spots)
    table["residual_mean_absolute"] = residual_absolute / store.n_spots
    for factor in range(settings.n_factors):
        table[f"A_tilde_{factor:02d}"] = normalized_loadings[:, factor]
        table[f"Pi_{factor:02d}"] = mixtures[:, factor]
    table.to_parquet(output_dir / "shared_factor_loadings.parquet", index=False)
    table.to_csv(output_dir / "shared_factor_loadings.csv.gz", index=False)
    table[
        [
            "gene",
            "feature_position",
            "residual_mean",
            "residual_rmse",
            "residual_mean_absolute",
        ]
    ].to_csv(output_dir / "feature_specific_spatial_residual_summary.csv", index=False)

    raw_multiplier = coordinate_transform["multiplier_raw_to_scaled"]
    length_scales = (
        gp.covar_module.base_kernel.lengthscale.detach().cpu().reshape(-1).numpy()
        / raw_multiplier
    )
    output_scales = gp.covar_module.outputscale.detach().cpu().reshape(-1).numpy()
    global_fraction = np.full(store.n_features, np.nan)
    if {"joint_global_outputscale", "joint_local_outputscale"}.issubset(table.columns):
        global_scale = table["joint_global_outputscale"].to_numpy(dtype=float)
        local_scale = table["joint_local_outputscale"].to_numpy(dtype=float)
        global_fraction = global_scale / np.maximum(global_scale + local_scale, 1.0e-12)
    inherited = np.nansum(loadings * global_fraction[:, None], axis=0) / np.maximum(
        np.nansum(loadings * np.isfinite(global_fraction)[:, None], axis=0), 1.0e-12
    )
    factor_sums = maps.sum(axis=0, dtype=np.float64)
    normalized = maps / np.maximum(factor_sums[None, :], 1.0e-12)
    coords = store.spots[["x", "y"]].to_numpy(dtype=np.float64)
    summary = []
    for factor in range(settings.n_factors):
        center = np.average(coords, axis=0, weights=maps[:, factor])
        summary.append(
            {
                "factor": factor,
                "length_scale": float(length_scales[factor]),
                "output_scale": float(output_scales[factor]),
                "inherited_global_fraction": float(inherited[factor]),
                "factor_map_sum": float(factor_sums[factor]),
                "concentration_score": float(
                    store.n_spots * np.square(normalized[:, factor]).sum()
                ),
                "spatial_extent_weighted_variance": float(
                    np.average(np.square(coords - center).sum(axis=1), weights=maps[:, factor])
                ),
            }
        )
    pd.DataFrame(summary).to_csv(output_dir / "shared_factor_summary.csv", index=False)
    metadata = {
        "format_version": 1,
        "model": "Gaussian NSF-style factorization of feature-GP posterior means",
        "input": {
            "reference_vski_dir": str(store.reference_vski_dir),
            "n_spots": store.n_spots,
            "n_features": store.n_features,
            "feature_order_sha256": hashlib.sha256(
                "\n".join(store.feature_ids).encode()
            ).hexdigest(),
        },
        "settings": asdict(settings),
        "coordinate_transform": coordinate_transform,
        "likelihood": "Gaussian(mean_map | NSF reconstruction, inherited_variance + feature_sigma^2)",
        "factor_link": "exp clipped at latent value 20 for numerical stability",
        "kernel": "isotropic Matern-3/2 with factor-specific length and output scales",
        "variational_family": "full Cholesky covariance per factor with standard inducing-point VariationalStrategy",
        "initialization": {
            "paper_style_regularized_nmf": True,
            "nmf_iterations": int(nmf.n_iter_),
        },
        "cache": cache_manifest,
        "runtime": {
            "elapsed_seconds": float(time.time() - started),
            "device": str(device),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
        },
    }
    (output_dir / "shared_factor_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(output_dir / "shared_factor_training_history.csv", index=False)
