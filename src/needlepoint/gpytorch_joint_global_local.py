from __future__ import annotations

from dataclasses import replace
from needlepoint.storage import encode_gp_maps, storage_manifest

import gc
import hashlib
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse

from needlepoint.gpytorch_global_local import (
    GlobalLocalGPSettings,
    _BatchedCachedVNNGP,
    load_or_build_vnngp_geometry,
)
from needlepoint.gpytorch_vski import (
    _BatchedGridSVGP,
    _choose_device,
    _coordinate_grid_bounds,
    _evaluate_vski_posterior,
    _require_gpytorch,
    _standardize_columns_with_stats,
)
from needlepoint.svg import _as_dense_float
from needlepoint.artifacts import (
    atomic_savez_compressed,
    atomic_write_json,
    sha256_file,
)


LocalKLMode = Literal["stochastic", "full"]
LocalPretrainTarget = Literal["global_residual", "observed"]
LocalPretrainCacheMode = Literal["off", "read", "write", "readwrite"]
ObservationModel = Literal["gaussian", "poisson-log"]


@dataclass
class JointGlobalLocalGPSettings(GlobalLocalGPSettings):
    """Settings for additive global/local GP joint variational training."""

    joint_epochs: int = 5
    global_min_length_scale: float = 60.0
    local_kl_batch_size: int = 1024
    local_kl_mode: LocalKLMode = "stochastic"
    joint_global_variational_learning_rate: float = 0.002
    joint_global_hyperparameter_learning_rate: float = 0.005
    joint_local_variational_learning_rate: float = 0.002
    joint_local_hyperparameter_learning_rate: float = 0.002
    joint_likelihood_learning_rate: float = 0.002
    joint_noise_lower_bound: float = 1.0e-5
    joint_gradient_clip: float | None = None
    save_component_state_dicts: bool = False
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    local_pretrain_target: LocalPretrainTarget = "global_residual"
    local_pretrain_cache_dir: str | None = None
    local_pretrain_cache_mode: LocalPretrainCacheMode = "off"
    observation_model: ObservationModel = "gaussian"
    global_grid_eigensolver: str = "torch"
    global_reuse_identical_axes: bool = False
    global_variational_covariance: str = "meanfield"
    global_covariance_rank: int = 0
    global_covariance_initial_fraction: float = 0.01
    global_covariance_init_seed: int | None = None
    global_prior_jitter: float = 1.0e-3
    global_pretrain_updates: int | None = None
    local_pretrain_updates: int | None = None
    joint_updates: int | None = None
    global_grid_phase: tuple[float, float] = (0.0, 0.0)
    fixed_kernel_parameters: dict | None = None
    fixed_local_length_scale: bool = False


def expected_additive_poisson_log_likelihood(
    target,
    global_mean,
    global_variance,
    local_mean,
    local_variance,
    exposure,
):
    """Analytic ``E_q[log Poisson(y | exposure * exp(g + l))]``.

    ``exposure`` is normalized to a declared reference before entering this
    function.  The latent field therefore represents log expected count at the
    reference exposure, while preserving the original count sampling model.
    """
    exposure = exposure.clamp_min(torch_finfo_tiny(exposure))
    while exposure.ndim < target.ndim:
        exposure = exposure.unsqueeze(0)
    latent_mean = global_mean + local_mean
    latent_variance = global_variance + local_variance
    log_expected_rate = latent_mean + 0.5 * latent_variance
    expected_rate = exposure * log_expected_rate.clamp(max=30.0).exp()
    return (
        target * (exposure.log() + latent_mean)
        - expected_rate
        - (target + 1.0).lgamma()
    )


def torch_finfo_tiny(reference):
    """Return the positive finite floor for a floating-point tensor."""
    import torch

    return torch.finfo(reference.dtype).tiny


def lognormal_rate_moments(mean: np.ndarray, variance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Gaussian log-rate moments into rate mean and variance."""
    mean64 = np.asarray(mean, dtype=np.float64)
    variance64 = np.asarray(variance, dtype=np.float64)
    if not np.isfinite(mean64).all() or not np.isfinite(variance64).all():
        raise FloatingPointError("Log-rate moments must be finite")
    if np.any(variance64 < 0.0):
        raise FloatingPointError("Log-rate variance must be nonnegative")
    log_rate_mean = np.clip(mean64 + 0.5 * variance64, -60.0, 30.0)
    rate_mean = np.exp(log_rate_mean)
    log_second_moment = np.clip(2.0 * mean64 + variance64, -60.0, 60.0)
    rate_variance = np.expm1(np.clip(variance64, 0.0, 30.0)) * np.exp(
        log_second_moment
    )
    return rate_mean.astype(np.float32), rate_variance.astype(np.float32)


def expected_additive_gaussian_log_likelihood(
    target,
    global_mean,
    global_variance,
    local_mean,
    local_variance,
    noise_variance,
):
    """Analytic E_q[log p(y | g + l)] for factorized Gaussian q(g)q(l)."""
    noise = noise_variance
    while noise.ndim < target.ndim:
        noise = noise.unsqueeze(-1)
    squared_error = (target - global_mean - local_mean).square()
    return -0.5 * (
        np.log(2.0 * np.pi)
        + noise.log()
        + (squared_error + global_variance + local_variance) / noise
    )


def additive_joint_elbo_per_feature(
    target,
    global_mean,
    global_variance,
    local_mean,
    local_variance,
    noise_variance,
    global_kl,
    local_kl,
    n_training_spots: int,
):
    """Average-observation additive ELBO with each KL divided by N once."""
    if int(n_training_spots) <= 0:
        raise ValueError("n_training_spots must be positive")
    expected_log_likelihood = expected_additive_gaussian_log_likelihood(
        target,
        global_mean,
        global_variance,
        local_mean,
        local_variance,
        noise_variance,
    ).mean(dim=-1)
    return (
        expected_log_likelihood
        - global_kl / float(n_training_spots)
        - local_kl / float(n_training_spots)
    )


def fit_joint_global_local_gp_spatial_maps(
    expression: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    feature_names: np.ndarray,
    output_dir: str | Path,
    *,
    settings: JointGlobalLocalGPSettings | None = None,
    prediction_coords: np.ndarray | None = None,
    prediction_spot_indices: np.ndarray | None = None,
    training_prediction_rows: np.ndarray | None = None,
    global_initial_length_scales: np.ndarray | None = None,
    exposure: np.ndarray | None = None,
    include_gaussian_latent_aliases: bool | None = None,
) -> dict:
    """Fit additive global V-SKI and local VNNGP components jointly.

    Sequential global and local fits initialize the model. The final stage uses
    either the historical Gaussian observation model or an exposure-offset
    Poisson log-link and a factorized variational family ``q(global) q(local)``.
    Consequently, component covariance is zero under q and latent variances
    add; this is not exact full-posterior inference.
    """
    torch, gpytorch = _require_gpytorch()
    settings = settings or JointGlobalLocalGPSettings()
    _validate_joint_settings(settings)
    if include_gaussian_latent_aliases is not None and settings.observation_model == "gaussian":
        settings = replace(settings, retain_latent_maps=include_gaussian_latent_aliases)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coords_np = np.asarray(coords, dtype=np.float32)
    prediction_coords_np = (
        coords_np
        if prediction_coords is None
        else np.asarray(prediction_coords, dtype=np.float32)
    )
    feature_names = np.asarray(feature_names).astype(str)
    if expression.ndim != 2:
        raise ValueError("expression must have shape n_spots x n_features")
    n_training_spots, n_features = expression.shape
    if coords_np.shape != (n_training_spots, 2):
        raise ValueError("coords must have shape n_training_spots x 2")
    if prediction_coords_np.ndim != 2 or prediction_coords_np.shape[1] != 2:
        raise ValueError("prediction_coords must have shape n_prediction_spots x 2")
    if feature_names.shape != (n_features,):
        raise ValueError("feature_names must have one value per feature")
    if not np.isfinite(coords_np).all() or not np.isfinite(prediction_coords_np).all():
        raise ValueError("coordinates must be finite")
    if len(np.unique(coords_np, axis=0)) != n_training_spots:
        raise ValueError("duplicate training coordinates are not supported")
    exposure_reference = None
    normalized_exposure = None
    if settings.observation_model == "poisson-log":
        if exposure is None:
            raise ValueError("exposure is required for poisson-log observations")
        exposure_values = np.asarray(exposure, dtype=np.float64)
        if exposure_values.shape != (n_training_spots,):
            raise ValueError("exposure must have one value per training spot")
        if not np.isfinite(exposure_values).all() or np.any(exposure_values <= 0.0):
            raise ValueError("poisson-log exposure must be finite and strictly positive")
        exposure_reference = float(np.median(exposure_values))
        if not np.isfinite(exposure_reference) or exposure_reference <= 0.0:
            raise ValueError("exposure reference must be finite and positive")
        normalized_exposure = (exposure_values / exposure_reference).astype(
            np.float32
        )
    elif exposure is not None:
        raise ValueError("exposure is only valid with observation_model='poisson-log'")
    if global_initial_length_scales is None:
        initial_global_scales = np.full(
            n_features, float(settings.global_initial_length_scale), dtype=np.float32
        )
        global_initialization_mode = "constant"
    else:
        initial_global_scales = np.asarray(
            global_initial_length_scales, dtype=np.float32
        )
        if initial_global_scales.shape != (n_features,):
            raise ValueError(
                "global_initial_length_scales must have one value per feature"
            )
        global_initialization_mode = "per_feature"
    if not np.isfinite(initial_global_scales).all():
        raise ValueError("global initial length scales must be finite")
    if np.any(initial_global_scales <= float(settings.global_min_length_scale)):
        raise ValueError(
            "every global initial length scale must be strictly greater than "
            "global_min_length_scale"
        )
    near_boundary = initial_global_scales - float(settings.global_min_length_scale) < 1.0e-3
    if np.any(near_boundary):
        warnings.warn(
            "Global length scales initialized within 0.001 of their softplus "
            "lower bound have nearly zero parameter gradients. Use "
            "interior_lengthscale_initialization for screened scales. Values "
            "are preserved here for historical reproducibility.",
            RuntimeWarning,
            stacklevel=2,
        )
    if n_training_spots <= int(settings.local_neighbors):
        raise ValueError("local_neighbors must be smaller than n_training_spots")

    n_prediction_spots = prediction_coords_np.shape[0]
    if training_prediction_rows is None:
        if n_prediction_spots != n_training_spots:
            raise ValueError(
                "training_prediction_rows is required when prediction coordinates differ"
            )
        training_prediction_rows = np.arange(n_training_spots, dtype=np.int64)
    training_prediction_rows = np.asarray(training_prediction_rows, dtype=np.int64)
    if training_prediction_rows.shape != (n_training_spots,):
        raise ValueError("training_prediction_rows must map every training row")
    if (
        np.unique(training_prediction_rows).size != n_training_spots
        or training_prediction_rows.min(initial=0) < 0
        or training_prediction_rows.max(initial=-1) >= n_prediction_spots
    ):
        raise ValueError("training_prediction_rows must be unique valid prediction rows")
    if not np.array_equal(coords_np, prediction_coords_np[training_prediction_rows]):
        raise ValueError("training_prediction_rows is incompatible with training coordinates")

    torch_dtype = torch.float64 if settings.dtype == "float64" else torch.float32
    device = _choose_device(torch, settings.device)
    torch.manual_seed(int(settings.random_state))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(settings.random_state))
    output_dtype = np.float16 if settings.output_dtype == "float16" else np.float32

    shared_started = time.time()
    order, predecessor_indices, geometry_cache_audit = load_or_build_vnngp_geometry(
        coords_np,
        settings.local_neighbors,
        candidate_multiplier=settings.predecessor_candidate_multiplier,
        query_batch_size=settings.neighbor_query_batch_size,
        cache_dir=settings.geometry_cache_dir,
        cache_mode=settings.geometry_cache_mode,
    )
    ordered_coords = coords_np[order]
    ordered_prediction_rows = training_prediction_rows[order]
    ordered_exposure_cpu = (
        None
        if normalized_exposure is None
        else torch.as_tensor(
            normalized_exposure[order], dtype=torch_dtype, device="cpu"
        )
    )
    grid_bounds = _coordinate_grid_bounds(
        np.vstack([ordered_coords, prediction_coords_np])
    )
    coords_t = torch.as_tensor(ordered_coords, dtype=torch_dtype, device=device)
    prediction_coords_t = torch.as_tensor(
        prediction_coords_np, dtype=torch_dtype, device=device
    )
    predecessor_indices_t = torch.as_tensor(
        predecessor_indices, dtype=torch.long, device=device
    )
    shared_setup_seconds = time.time() - shared_started
    local_cache_dir = (
        Path(settings.local_pretrain_cache_dir)
        if settings.local_pretrain_cache_dir is not None
        else None
    )
    if local_cache_dir is not None and settings.local_pretrain_cache_mode in {
        "write",
        "readwrite",
    }:
        local_cache_dir.mkdir(parents=True, exist_ok=True)
    local_cache_context = {
        "version": 1,
        "coordinate_sha256": _array_sha256(ordered_coords),
        "predecessor_sha256": _array_sha256(predecessor_indices),
        "n_training_spots": int(n_training_spots),
        "torch_version": str(torch.__version__),
        "gpytorch_version": str(gpytorch.__version__),
    }
    local_cache_hits = 0
    local_cache_writes = 0

    chunk_records: list[dict] = []
    metadata_records: list[dict] = []
    pretrain_history: list[dict] = []
    joint_history: list[dict] = []
    hyperparameter_history: list[dict] = []
    stage_records: list[dict] = []
    covariance_initializations: list[dict] = []
    inducing_grid_axes: list[list[float]] | None = None
    started = time.time()

    for chunk_id, start in enumerate(
        range(0, n_features, int(settings.peak_chunk_size))
    ):
        stop = min(start + int(settings.peak_chunk_size), n_features)
        positions = np.arange(start, stop, dtype=np.int64)
        chunk_feature_names = feature_names[positions]
        chunk_raw = _dense_feature_chunk(expression, positions)
        if not np.isfinite(chunk_raw).all():
            raise ValueError("nonfinite measurements require an explicit missing-observation mask")
        if settings.observation_model == "gaussian":
            chunk_model_values, chunk_means, chunk_stds = (
                _standardize_columns_with_stats(chunk_raw)
            )
        else:
            if not np.isfinite(chunk_raw).all() or np.any(chunk_raw < 0.0):
                raise ValueError("poisson-log observations must be finite and nonnegative")
            chunk_model_values = chunk_raw
            chunk_means = np.zeros(positions.size, dtype=np.float32)
            chunk_stds = np.ones(positions.size, dtype=np.float32)
        ordered_y = chunk_model_values[order]
        y_cpu = torch.as_tensor(
            ordered_y.T, dtype=torch_dtype, device="cpu"
        )
        n_chunk_features = positions.size

        global_model = _BatchedGridSVGP.build(
            gpytorch,
            torch,
            n_features=n_chunk_features,
            grid_size=int(settings.global_grid_size),
            grid_bounds=grid_bounds,
            variational_distribution=("cholesky" if settings.global_variational_covariance ==
                "cholesky_reference" else settings.global_variational_covariance),
            kl_backend={"meanfield": "structured_kron", "lowrank_diag": "structured_lowrank",
                        "cholesky_reference": "generic"}[settings.global_variational_covariance],
            prior_jitter=settings.global_prior_jitter,
            covariance_rank=settings.global_covariance_rank,
            covariance_initial_fraction=settings.global_covariance_initial_fraction,
            covariance_init_seed=(settings.random_state if settings.global_covariance_init_seed
                is None else settings.global_covariance_init_seed),
            min_length_scale=float(settings.global_min_length_scale),
            grid_eigensolver=settings.global_grid_eigensolver,
            reuse_identical_axes=settings.global_reuse_identical_axes,
        ).to(device=device, dtype=torch_dtype)
        if any(settings.global_grid_phase):
            strategy = global_model.variational_strategy
            with torch.no_grad():
                shift = torch.as_tensor(settings.global_grid_phase,device=device,
                    dtype=torch_dtype)*(strategy.grid[1]-strategy.grid[0])
                strategy.grid.add_(shift)
                strategy.inducing_points.add_(shift)
        global_likelihood = None
        if settings.observation_model == "gaussian":
            global_likelihood = gpytorch.likelihoods.GaussianLikelihood(
                batch_shape=torch.Size([n_chunk_features])
            ).to(device=device, dtype=torch_dtype)
        chunk_initial_global_scales = torch.as_tensor(
            initial_global_scales[positions],
            dtype=torch_dtype,
            device=device,
        ).view(-1, 1, 1)
        global_model.covar_module.base_kernel.lengthscale = (
            chunk_initial_global_scales
        )
        global_model.covar_module.outputscale = 0.5
        if global_likelihood is not None:
            global_likelihood.noise = 0.5
        else:
            assert normalized_exposure is not None
            base_rates = (
                chunk_raw.sum(axis=0, dtype=np.float64) + 0.5
            ) / (normalized_exposure.sum(dtype=np.float64) + 0.5)
            global_model.mean_module.constant = torch.as_tensor(
                np.log(np.clip(base_rates, 1.0e-8, None)),
                dtype=torch_dtype,
                device=device,
            )
        _apply_fixed_parameters(global_model, global_likelihood, settings, "global")
        if inducing_grid_axes is None:
            grid = global_model.variational_strategy.grid.detach().cpu().numpy()
            inducing_grid_axes = [
                grid[:, dimension].astype(float).tolist()
                for dimension in range(grid.shape[1])
            ]

        stage_started, memory_started = _start_stage(torch, device)
        _pretrain_global_component(
            torch=torch,
            gpytorch=gpytorch,
            model=global_model,
            likelihood=global_likelihood,
            coords_t=coords_t,
            y_cpu=y_cpu,
            settings=settings,
            chunk_id=chunk_id,
            history=pretrain_history,
            hyperparameter_history=hyperparameter_history,
            feature_names=chunk_feature_names,
            feature_positions=positions,
            exposure_cpu=ordered_exposure_cpu,
        )
        stage_records.append(
            _finish_stage(
                torch,
                device,
                chunk_id,
                "global_pretrain",
                stage_started,
                memory_started,
            )
        )
        global_pretrain = _component_parameters(global_model, global_likelihood)
        global_q = global_model.variational_strategy._variational_distribution
        if hasattr(global_q, "initialization_statistics"):
            covariance_initializations.append(dict(chunk_id=chunk_id,**global_q.initialization_statistics))

        global_training_variance = None
        if (
            settings.local_pretrain_target == "global_residual"
            or settings.observation_model == "poisson-log"
        ):
            global_training_mean, global_training_variance = _evaluate_vski_posterior(
                torch=torch,
                gpytorch=gpytorch,
                model=global_model,
                coords_t=coords_t,
                n_spots=n_training_spots,
                eval_batch_size=int(settings.eval_batch_size_spots),
                output_dtype=np.float32,
            )
            local_pretrain_target = (
                ordered_y - global_training_mean
                if settings.observation_model == "gaussian"
                else ordered_y
            )
        else:
            global_training_mean = None
            local_pretrain_target = ordered_y
        local_pretrain_target_cpu = torch.as_tensor(
            local_pretrain_target.T, dtype=torch_dtype, device="cpu"
        )
        global_training_mean_cpu = (
            None
            if settings.observation_model == "gaussian"
            else torch.as_tensor(
                global_training_mean.T, dtype=torch_dtype, device="cpu"
            )
        )
        global_training_variance_cpu = (
            None
            if settings.observation_model == "gaussian"
            else torch.as_tensor(
                global_training_variance.T, dtype=torch_dtype, device="cpu"
            )
        )

        local_model = _BatchedCachedVNNGP.build(
            gpytorch,
            torch,
            coords_t,
            predecessor_indices_t,
            n_features=n_chunk_features,
            n_neighbors=settings.local_neighbors,
            training_batch_size=settings.batch_size_spots,
            min_length_scale=settings.local_min_length_scale,
            max_length_scale=settings.local_max_length_scale,
        ).to(device=device, dtype=torch_dtype)
        local_likelihood = None
        if settings.observation_model == "gaussian":
            local_likelihood = gpytorch.likelihoods.GaussianLikelihood(
                batch_shape=torch.Size([n_chunk_features])
            ).to(device=device, dtype=torch_dtype)
        local_model.covar_module.base_kernel.lengthscale = float(
            settings.local_initial_length_scale
        )
        local_model.covar_module.outputscale = 0.25
        if local_likelihood is not None:
            local_likelihood.noise = 0.25
        _apply_fixed_parameters(local_model, local_likelihood, settings, "local")
        if settings.fixed_local_length_scale:
            # Physical-policy fits use one local spatial scale rather than a
            # separate feature-wise search/optimization of its length scale.
            # Other local hyperparameters and all variational parameters remain
            # trainable during both the local and joint stages.
            local_model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
        _initialize_local_variational_distribution(local_model)

        cache_signature = None
        cache_paths = None
        if settings.local_pretrain_cache_mode != "off":
            cache_signature = _local_pretrain_cache_signature(
                context=local_cache_context,
                settings=settings,
                feature_names=chunk_feature_names,
                feature_positions=positions,
                target=local_pretrain_target,
            )
            cache_paths = _local_pretrain_cache_paths(
                local_cache_dir,
                positions,
                cache_signature,
            )
        stage_started, memory_started = _start_stage(torch, device)
        cache_hit = False
        if settings.local_pretrain_cache_mode in {"read", "readwrite"}:
            cache_hit = _restore_local_pretrain_cache(
                torch=torch,
                cache_paths=cache_paths,
                expected_signature=cache_signature,
                model=local_model,
                likelihood=local_likelihood,
            )
            if not cache_hit and settings.local_pretrain_cache_mode == "read":
                raise FileNotFoundError(
                    "No compatible local-pretraining cache entry for "
                    f"feature positions {start}:{stop} in {local_cache_dir}"
                )
        if cache_hit:
            local_cache_hits += 1
            stage_name = "local_pretrain_cache_load"
        else:
            _pretrain_local_component(
                torch=torch,
                model=local_model,
                likelihood=local_likelihood,
                target_cpu=local_pretrain_target_cpu,
                settings=settings,
                chunk_id=chunk_id,
                history=pretrain_history,
                hyperparameter_history=hyperparameter_history,
                feature_names=chunk_feature_names,
                feature_positions=positions,
                exposure_cpu=ordered_exposure_cpu,
                fixed_global_mean_cpu=global_training_mean_cpu,
                fixed_global_variance_cpu=global_training_variance_cpu,
            )
            stage_name = "local_pretrain"
            if settings.local_pretrain_cache_mode in {"write", "readwrite"}:
                assert cache_signature is not None
                _write_local_pretrain_cache(
                    torch=torch,
                    cache_paths=cache_paths,
                    signature=cache_signature,
                    model=local_model,
                    likelihood=local_likelihood,
                )
                local_cache_writes += 1
        stage_records.append(
            _finish_stage(
                torch,
                device,
                chunk_id,
                stage_name,
                stage_started,
                memory_started,
            )
        )
        local_pretrain = _component_parameters(local_model, local_likelihood)

        joint_likelihood = None
        if settings.observation_model == "gaussian":
            joint_likelihood = gpytorch.likelihoods.GaussianLikelihood(
                batch_shape=torch.Size([n_chunk_features]),
                noise_constraint=gpytorch.constraints.GreaterThan(
                    float(settings.joint_noise_lower_bound)
                ),
            ).to(device=device, dtype=torch_dtype)
            joint_likelihood.noise = local_likelihood.noise.detach().clone()
            if settings.fixed_kernel_parameters is not None:
                joint_likelihood.noise = settings.fixed_kernel_parameters["noise"]
                joint_likelihood.requires_grad_(False)

        stage_started, memory_started = _start_stage(torch, device)
        _jointly_train_components(
            torch=torch,
            global_model=global_model,
            local_model=local_model,
            joint_likelihood=joint_likelihood,
            coords_t=coords_t,
            y_cpu=y_cpu,
            settings=settings,
            chunk_id=chunk_id,
            history=joint_history,
            hyperparameter_history=hyperparameter_history,
            feature_names=chunk_feature_names,
            feature_positions=positions,
            exposure_cpu=ordered_exposure_cpu,
        )
        stage_records.append(
            _finish_stage(
                torch,
                device,
                chunk_id,
                "joint_finetune",
                stage_started,
                memory_started,
            )
        )

        prediction_started, prediction_memory = _start_stage(torch, device)
        global_mean, global_variance = _evaluate_vski_posterior(
            torch=torch,
            gpytorch=gpytorch,
            model=global_model,
            coords_t=prediction_coords_t,
            n_spots=n_prediction_spots,
            eval_batch_size=int(settings.eval_batch_size_spots),
            output_dtype=np.float32,
        )
        local_mean, local_variance = _evaluate_local_posterior(
            torch=torch,
            model=local_model,
            coords_t=coords_t,
            prediction_coords_t=prediction_coords_t,
            ordered_prediction_rows=ordered_prediction_rows,
            n_prediction_spots=n_prediction_spots,
            eval_batch_size=int(settings.eval_batch_size_spots),
        )
        stage_records.append(_finish_stage(torch, device, chunk_id, "prediction",
            prediction_started, prediction_memory))
        latent_mean = global_mean + local_mean
        latent_variance = global_variance + local_variance
        if settings.observation_model == "gaussian":
            posterior_mean = latent_mean
            posterior_variance = latent_variance
            joint_noise = (
                joint_likelihood.noise.detach()
                .cpu()
                .numpy()
                .astype(np.float32)
                .reshape(-1)
            )
            posterior_predictive_variance = (
                posterior_variance + joint_noise[None, :]
            )
        else:
            posterior_mean, posterior_variance = lognormal_rate_moments(
                latent_mean, latent_variance
            )
            posterior_predictive_variance = posterior_variance + posterior_mean
        _validate_output_arrays(
            posterior_mean,
            posterior_variance,
            posterior_predictive_variance,
            global_mean,
            global_variance,
            local_mean,
            local_variance,
        )

        chunk_name = f"vski_gp_maps_chunk_{chunk_id:04d}.npz"
        io_started, io_memory = _start_stage(torch, device)
        chunk_path = output_dir / chunk_name
        map_arrays = encode_gp_maps(settings, dict(
            posterior_mean=posterior_mean,
            posterior_variance=posterior_variance,
            posterior_predictive_variance=posterior_predictive_variance,
            global_posterior_mean=global_mean,
            local_posterior_mean=local_mean,
            global_posterior_variance=global_variance,
            local_posterior_variance=local_variance,
            latent_log_rate_mean=latent_mean,
            latent_log_rate_variance=latent_variance,
        ))
        map_arrays.update(gene=feature_names[positions], feature_position=positions)
        from needlepoint.posterior_blocks import atomic_savez_maps
        atomic_savez_maps(chunk_path, **map_arrays)
        del map_arrays

        joint_parameters = _joint_component_parameters(
            global_model, local_model, joint_likelihood
        )
        for local_column, feature_position in enumerate(positions):
            initial_global_scale = float(initial_global_scales[feature_position])
            learned_global_scale = float(
                joint_parameters["global_length_scale"][local_column]
            )
            global_scale_log2_change = float(
                np.log2(learned_global_scale / initial_global_scale)
            )
            metadata_records.append(
                {
                    "gene": str(feature_names[feature_position]),
                    "feature_position": int(feature_position),
                    "standardization_mean": float(chunk_means[local_column]),
                    "standardization_std": float(chunk_stds[local_column]),
                    "observation_model": settings.observation_model,
                    "posterior_map_scale": (
                        "standardized_gaussian"
                        if settings.observation_model == "gaussian"
                        else "expected_count_rate_at_reference_exposure"
                    ),
                    "exposure_reference": exposure_reference,
                    "global_pretrain_length_scale": global_pretrain["length_scale"][local_column],
                    "global_initial_length_scale": initial_global_scale,
                    "global_pretrain_outputscale": global_pretrain["outputscale"][local_column],
                    "global_pretrain_noise_variance": global_pretrain["noise"][local_column],
                    "local_pretrain_length_scale": local_pretrain["length_scale"][local_column],
                    "local_pretrain_outputscale": local_pretrain["outputscale"][local_column],
                    "local_pretrain_noise_variance": local_pretrain["noise"][local_column],
                    "local_pretrain_target": settings.local_pretrain_target,
                    "local_pretrain_cache_hit": bool(cache_hit),
                    "joint_global_length_scale": learned_global_scale,
                    "global_length_scale_log2_change": global_scale_log2_change,
                    "global_length_scale_changed_over_one_octave": bool(
                        abs(global_scale_log2_change) > 1.0
                    ),
                    "joint_global_outputscale": joint_parameters["global_outputscale"][local_column],
                    "joint_local_length_scale": joint_parameters["local_length_scale"][local_column],
                    "joint_local_outputscale": joint_parameters["local_outputscale"][local_column],
                    "joint_noise_variance": joint_parameters["joint_noise"][local_column],
                    "learned_length_scale": learned_global_scale,
                    "learned_outputscale": joint_parameters["global_outputscale"][local_column],
                    "learned_noise_variance": joint_parameters["joint_noise"][local_column],
                    "map_chunk_id": int(chunk_id),
                    "map_chunk_file": chunk_name,
                    "map_column": int(local_column),
                }
            )

        state_file = None
        if settings.save_component_state_dicts:
            state_file = f"joint_global_local_state_chunk_{chunk_id:04d}.pt"
            state_payload = {
                "checkpoint_schema": 2,
                "covariance_family": settings.global_variational_covariance,
                "covariance_rank": settings.global_covariance_rank,
                "covariance_basis": "inducing_values_unwhitened",
                "global_model": global_model.state_dict(),
                "local_model": local_model.state_dict(),
                "global_pretrain_optimizer": getattr(global_model, "_pretrain_optimizer_state", None),
                "local_pretrain_optimizer": getattr(local_model, "_pretrain_optimizer_state", None),
                "joint_optimizer": getattr(global_model, "_joint_optimizer_state", None),
                "joint_rng": getattr(global_model, "_joint_rng_state", None),
                "settings": asdict(settings),
                "training_coordinate_sha256": _array_sha256(ordered_coords),
                "training_value_sha256": _array_sha256(ordered_y),
                "neighbor_dag_sha256": _array_sha256(predecessor_indices),
            }
            if joint_likelihood is not None:
                state_payload["joint_likelihood"] = joint_likelihood.state_dict()
            torch.save(
                state_payload,
                output_dir / state_file,
            )
        stage_records.append(_finish_stage(torch, device, chunk_id, "map_and_checkpoint_io",
            io_started, io_memory))
        chunk_records.append(
            {
                "chunk_id": int(chunk_id),
                "file": chunk_name,
                "state_dict_file": state_file,
                "n_spots": int(n_prediction_spots),
                "n_training_spots": int(n_training_spots),
                "n_features": int(n_chunk_features),
                "feature_start": int(start),
                "feature_stop": int(stop),
                "local_pretrain_target": settings.local_pretrain_target,
                "local_pretrain_cache_hit": bool(cache_hit),
                "observation_model": settings.observation_model,
                "file_size_bytes": int(chunk_path.stat().st_size),
                "file_sha256": sha256_file(chunk_path),
            }
        )
        print(
            f"Joint global-local wrote {chunk_name} for features {start}:{stop}",
            flush=True,
        )

        del (
            chunk_raw,
            chunk_model_values,
            ordered_y,
            y_cpu,
            global_training_mean,
            global_training_variance,
            global_training_mean_cpu,
            global_training_variance_cpu,
            local_pretrain_target,
            local_pretrain_target_cpu,
            global_model,
            global_likelihood,
            local_model,
            local_likelihood,
            joint_likelihood,
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(metadata_records).to_csv(
        output_dir / "vski_gp_model_metadata.csv", index=False
    )
    pd.DataFrame(chunk_records).to_csv(
        output_dir / "vski_gp_map_chunks.csv", index=False
    )
    pd.DataFrame(pretrain_history).to_csv(
        output_dir / "joint_global_local_pretraining_history.csv", index=False
    )
    pd.DataFrame(joint_history).to_csv(
        output_dir / "joint_global_local_training_history.csv", index=False
    )
    pd.DataFrame(hyperparameter_history).to_csv(
        output_dir / "joint_global_local_hyperparameter_history.csv", index=False
    )
    pd.DataFrame(stage_records).to_csv(
        output_dir / "joint_global_local_stage_timing.csv", index=False
    )

    if prediction_spot_indices is None:
        prediction_spot_indices = np.arange(n_prediction_spots, dtype=np.int64)
    prediction_spot_indices = np.asarray(prediction_spot_indices)
    if prediction_spot_indices.shape != (n_prediction_spots,):
        raise ValueError("prediction_spot_indices must match prediction coordinates")
    used_for_training = np.zeros(n_prediction_spots, dtype=bool)
    used_for_training[training_prediction_rows] = True
    pd.DataFrame(
        {
            "map_row": np.arange(n_prediction_spots, dtype=np.int64),
            "spot_index": prediction_spot_indices,
            "x": prediction_coords_np[:, 0],
            "y": prediction_coords_np[:, 1],
            "used_for_training": used_for_training,
            "prediction_only": ~used_for_training,
        }
    ).to_csv(output_dir / "vski_gp_map_spots.csv.gz", index=False)

    manifest = {
        "method": "joint additive global V-SKI plus local VNNGP",
        "model": (
            "y = global_rbf + local_matern32 + Gaussian_noise"
            if settings.observation_model == "gaussian"
            else "y ~ Poisson(exposure/reference * exp(global_rbf + local_matern32))"
        ),
        "observation_model": settings.observation_model,
        "exposure": (
            None
            if settings.observation_model == "gaussian"
            else {
                "definition": "full-library row count",
                "reference_statistic": "median",
                "reference_value": exposure_reference,
                "normalized_minimum": float(normalized_exposure.min()),
                "normalized_median": float(np.median(normalized_exposure)),
                "normalized_maximum": float(normalized_exposure.max()),
            }
        ),
        "posterior_map_scale": (
            "standardized_gaussian"
            if settings.observation_model == "gaussian"
            else "expected_count_rate_at_reference_exposure"
        ),
        "variational_family": "q(global) q(local)",
        "component_covariance_under_q": "zero by factorization",
        "latent_variance": "global_variance + local_variance",
        "predictive_variance": (
            "latent_variance + joint_noise_variance"
            if settings.observation_model == "gaussian"
            else "lognormal_rate_variance + expected_rate (Poisson sampling)"
        ),
        "fitting": "sequential initialization followed by joint fine-tuning",
        "global_length_scale_initialization": {
            "mode": global_initialization_mode,
            "minimum": float(initial_global_scales.min()),
            "median": float(np.median(initial_global_scales)),
            "maximum": float(initial_global_scales.max()),
            "n_unique": int(np.unique(initial_global_scales).size),
            "n_near_softplus_boundary": int(np.count_nonzero(near_boundary)),
        },
        "optimization_audit": _optimization_audit(
            pretrain_history, joint_history, n_training_spots, settings.batch_size_spots
        ),
        "local_pretrain_target": settings.local_pretrain_target,
        "local_pretrain_cache": {
            "mode": settings.local_pretrain_cache_mode,
            "directory": settings.local_pretrain_cache_dir,
            "hits": int(local_cache_hits),
            "writes": int(local_cache_writes),
            "note": (
                "Reusable only for observed-target initialization; residual-target "
                "pretraining depends on the fitted global start."
            ),
        },
        "early_stopping_metric": (
            "mean training negative ELBO; optimization safeguard only"
        ),
        "early_stopping_restoration": (
            "CPU-cached best state is restored when patience is positive"
        ),
        "inference_note": "factorized variational approximation; not exact posterior inference",
        "n_training_spots": int(n_training_spots),
        "n_prediction_spots": int(n_prediction_spots),
        "n_prediction_only_spots": int(n_prediction_spots - n_training_spots),
        "training_coverage_fraction": float(n_training_spots / n_prediction_spots),
        "full_training_coverage": bool(n_training_spots == n_prediction_spots),
        "n_features": int(n_features),
        "grid_bounds": [(float(lo), float(hi)) for lo, hi in grid_bounds],
        "inducing_grid_axes": inducing_grid_axes,
        "n_inducing_grid_points": int(settings.global_grid_size) ** 2,
        "neighbor_dag_shape": list(predecessor_indices.shape),
        "vnngp_geometry_cache": geometry_cache_audit,
        "vnngp_conditional_kl_backend": "indexed-neighbor-gather-v1",
        "global_grid_eigensolver": settings.global_grid_eigensolver,
        "global_covariance": {
            "family": settings.global_variational_covariance,
            "rank": settings.global_covariance_rank,
            "basis": "inducing_values_unwhitened",
            "implementation_version": "inducing-values-diag-plus-root-v1",
            "prior_jitter": settings.global_prior_jitter,
            "axis_factor_stabilization": 1e-10,
            "spectral_work_dtype": "float64",
            "initial_fraction": settings.global_covariance_initial_fraction,
            "init_seed": (settings.random_state if settings.global_covariance_init_seed is None
                          else settings.global_covariance_init_seed),
            "diagonal_parameterization": ("squared_scale_plus_1e-10_floor" if
                settings.global_variational_covariance == "lowrank_diag" else "legacy"),
        },
        "global_covariance_initializations": covariance_initializations,
        "fit_purpose": ("evaluation" if n_training_spots < n_prediction_spots else
                        "full_support"),
        "neighbor_dag_sha256": _array_sha256(predecessor_indices),
        "training_coordinate_sha256": _array_sha256(ordered_coords),
        "update_budget_precedence": "explicit updates override epochs per stage; early stopping disabled in fixed-update mode",
        "global_reuse_identical_axes": settings.global_reuse_identical_axes,
        "gaussian_latent_aliases_retained": bool(settings.retain_latent_maps and settings.observation_model == "gaussian"),
        "storage": storage_manifest(settings),
        "shared_setup_seconds": float(shared_setup_seconds),
        "settings": asdict(settings),
        "device": str(device),
        "dtype": settings.dtype,
        "output_dtype": settings.output_dtype,
        "chunks": chunk_records,
        "metadata_file": "vski_gp_model_metadata.csv",
        "training_history_file": "joint_global_local_training_history.csv",
        "pretraining_history_file": "joint_global_local_pretraining_history.csv",
        "hyperparameter_history_file": "joint_global_local_hyperparameter_history.csv",
        "stage_timing_file": "joint_global_local_stage_timing.csv",
        "spot_file": "vski_gp_map_spots.csv.gz",
        "chunk_index_file": "vski_gp_map_chunks.csv",
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_write_json(output_dir / "vski_gp_maps_manifest.json", manifest)
    return manifest


def _optimization_audit(pretrain_history, joint_history, n_training, batch_size):
    """Report update counts and unresolved descent; an epoch cap is not convergence."""
    steps = int(np.ceil(n_training / batch_size))
    stages = {}
    for row in pretrain_history:
        key = f"chunk_{row['chunk_id']}_{row.get('stage', 'pretrain')}"
        stages[key] = stages.get(key, 0) + row.get("optimizer_updates", steps)
    for row in joint_history:
        key = f"chunk_{row['chunk_id']}_joint"
        stages[key] = stages.get(key, 0) + row.get("optimizer_updates", steps)
    descent = []
    for chunk in sorted({row['chunk_id'] for row in joint_history}):
        rows = [row for row in joint_history if row['chunk_id'] == chunk]
        if len(rows) < 2:
            continue
        previous, final = [float(row['negative_joint_elbo']) for row in rows[-2:]]
        improvement = (previous - final) / max(abs(previous), 1.0e-12)
        if improvement > 0.01 and not rows[-1].get('stopped_early', False):
            descent.append({"chunk_id": int(chunk), "final_relative_improvement": improvement})
    if descent:
        warnings.warn(
            "Joint training reached its epoch cap while the final training "
            "objective improved by more than 1%; convergence is unverified. "
            "Inspect optimization_audit and use held-out validation.",
            RuntimeWarning,
            stacklevel=2,
        )
    return {
        "updates_per_epoch": steps,
        "optimizer_updates_by_stage_and_chunk": stages,
        "likelihood_examples_processed": sum(r.get("likelihood_examples_processed", 0)
            for r in pretrain_history + joint_history),
        "local_conditional_terms_processed": sum(r.get("local_conditional_terms_processed", 0)
            for r in pretrain_history + joint_history),
        "chunks_still_descending_at_cap": descent,
        "convergence_verified": False,
        "note": "Training objectives are stochastic; this audit is not a convergence certificate.",
    }


def _validate_joint_settings(settings: JointGlobalLocalGPSettings) -> None:
    if not isinstance(settings.fixed_local_length_scale, bool):
        raise ValueError("fixed_local_length_scale must be boolean")
    family = settings.global_variational_covariance
    if family not in {"meanfield", "lowrank_diag", "cholesky_reference"}:
        raise ValueError("unsupported global_variational_covariance")
    rank = settings.global_covariance_rank
    if isinstance(rank, bool) or int(rank) != rank or rank < 0:
        raise ValueError("global_covariance_rank must be a nonnegative integer")
    if family == "lowrank_diag":
        if not 0 < rank <= settings.global_grid_size**2:
            raise ValueError("lowrank_diag requires positive rank <= grid size squared")
    elif rank:
        raise ValueError("global_covariance_rank is unused outside lowrank_diag")
    if family != "meanfield" and settings.observation_model != "gaussian":
        raise ValueError("new covariance modes are validated for Gaussian observations only")
    if family == "cholesky_reference" and settings.global_grid_size**2 > 1024:
        raise ValueError("cholesky_reference is limited to 1024 inducing values")
    if family == "lowrank_diag" and settings.global_grid_eigensolver != "torch":
        raise ValueError("lowrank_diag uses the audited Torch spectral backend")
    if not 0 < settings.global_covariance_initial_fraction < 1:
        raise ValueError("global_covariance_initial_fraction must be in (0,1)")
    if not np.isfinite(settings.global_prior_jitter) or settings.global_prior_jitter <= 0:
        raise ValueError("global_prior_jitter must be finite and positive")
    if settings.global_prior_jitter != 1e-3:
        raise ValueError("the installed grid strategy has fixed prior jitter 0.001")
    if len(settings.global_grid_phase) != 2 or not np.isfinite(settings.global_grid_phase).all():
        raise ValueError("global_grid_phase must contain two finite fractions")
    for key in ("global_pretrain_updates", "local_pretrain_updates", "joint_updates"):
        value = getattr(settings,key)
        if value is not None and (isinstance(value,bool) or int(value) != value or value < 0):
            raise ValueError(f"{key} must be a nonnegative integer or None")
        if value is not None and settings.early_stopping_patience:
            raise ValueError("fixed-update mode requires early_stopping_patience=0")
    if settings.fixed_kernel_parameters is not None:
        required = {"global_length", "global_scale", "local_length", "local_scale", "noise"}
        if set(settings.fixed_kernel_parameters) != required:
            raise ValueError(f"fixed_kernel_parameters must contain exactly {sorted(required)}")
        if any(not np.isfinite(v) or v <= 0 for v in settings.fixed_kernel_parameters.values()):
            raise ValueError("fixed kernel parameters must be finite and positive")
    if settings.observation_model not in {"gaussian", "poisson-log"}:
        raise ValueError("observation_model must be gaussian or poisson-log")
    if (
        not np.isfinite(settings.global_min_length_scale)
        or settings.global_min_length_scale <= 0
    ):
        raise ValueError("global_min_length_scale must be finite and positive")
    if (
        not np.isfinite(settings.local_min_length_scale)
        or settings.local_min_length_scale <= 0
    ):
        raise ValueError("local_min_length_scale must be finite and positive")
    if not settings.global_initial_length_scale > settings.global_min_length_scale:
        raise ValueError(
            "global_initial_length_scale must lie strictly above its lower bound"
        )
    if not (
        settings.local_min_length_scale
        < settings.local_initial_length_scale
        < settings.local_max_length_scale
    ):
        raise ValueError(
            "local_initial_length_scale must lie strictly inside its bounds"
        )
    if settings.global_grid_eigensolver not in {"torch", "jacobi-experimental"}:
        raise ValueError("Unsupported global grid eigensolver")
    if settings.local_kl_mode not in {"stochastic", "full"}:
        raise ValueError("local_kl_mode must be 'stochastic' or 'full'")
    if settings.local_pretrain_target not in {"global_residual", "observed"}:
        raise ValueError(
            "local_pretrain_target must be 'global_residual' or 'observed'"
        )
    if settings.local_pretrain_cache_mode not in {
        "off",
        "read",
        "write",
        "readwrite",
    }:
        raise ValueError(
            "local_pretrain_cache_mode must be off, read, write, or readwrite"
        )
    if settings.local_pretrain_cache_mode != "off":
        if settings.observation_model != "gaussian":
            raise ValueError(
                "Local pretraining caches are not yet supported for poisson-log"
            )
        if settings.local_pretrain_target != "observed":
            raise ValueError(
                "Reusable local pretraining requires local_pretrain_target='observed'; "
                "global-residual targets differ across global starts"
            )
        if not settings.local_pretrain_cache_dir:
            raise ValueError(
                "local_pretrain_cache_dir is required when cache mode is enabled"
            )
    if settings.geometry_cache_mode not in {"off", "read", "readwrite"}:
        raise ValueError("geometry_cache_mode must be off, read, or readwrite")
    if settings.geometry_cache_mode != "off" and not settings.geometry_cache_dir:
        raise ValueError(
            "geometry_cache_dir is required when geometry_cache_mode is enabled"
        )
    for name in (
        "global_epochs",
        "local_epochs",
        "joint_epochs",
        "batch_size_spots",
        "peak_chunk_size",
        "eval_batch_size_spots",
        "local_kl_batch_size",
    ):
        if int(getattr(settings, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if settings.joint_noise_lower_bound <= 0:
        raise ValueError("joint_noise_lower_bound must be positive")
    if settings.joint_gradient_clip is not None and settings.joint_gradient_clip <= 0:
        raise ValueError("joint_gradient_clip must be positive when provided")
    if int(settings.early_stopping_patience) < 0:
        raise ValueError("early_stopping_patience must be nonnegative")
    if float(settings.early_stopping_min_delta) < 0:
        raise ValueError("early_stopping_min_delta must be nonnegative")


def _dense_feature_chunk(expression, positions: np.ndarray) -> np.ndarray:
    block = expression[:, positions]
    return _as_dense_float(block).astype(np.float32, copy=False)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _local_pretrain_cache_signature(
    *,
    context: dict,
    settings: JointGlobalLocalGPSettings,
    feature_names: np.ndarray,
    feature_positions: np.ndarray,
    target: np.ndarray,
) -> dict:
    """Describe every input that can change a reusable local initialization."""
    return {
        **context,
        "local_pretrain_target": settings.local_pretrain_target,
        "feature_names": [str(value) for value in feature_names],
        "feature_positions": [int(value) for value in feature_positions],
        "target_sha256": _array_sha256(target),
        "settings": {
            "local_pretrain_updates": settings.local_pretrain_updates,
            "fixed_kernel_parameters": settings.fixed_kernel_parameters,
            "local_epochs": int(settings.local_epochs),
            "batch_size_spots": int(settings.batch_size_spots),
            "local_learning_rate": float(settings.local_learning_rate),
            "local_hyperparameter_learning_rate": float(
                settings.local_hyperparameter_learning_rate
            ),
            "local_initial_length_scale": float(
                settings.local_initial_length_scale
            ),
            "fixed_local_length_scale": bool(settings.fixed_local_length_scale),
            "local_min_length_scale": float(settings.local_min_length_scale),
            "local_max_length_scale": float(settings.local_max_length_scale),
            "local_neighbors": int(settings.local_neighbors),
            "local_kl_batch_size": int(settings.local_kl_batch_size),
            "local_kl_mode": settings.local_kl_mode,
            "predecessor_candidate_multiplier": int(
                settings.predecessor_candidate_multiplier
            ),
            "dtype": settings.dtype,
            "random_state": int(settings.random_state),
            "early_stopping_patience": int(settings.early_stopping_patience),
            "early_stopping_min_delta": float(settings.early_stopping_min_delta),
        },
    }


def _signature_digest(signature: dict) -> str:
    encoded = json.dumps(
        signature, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _local_pretrain_cache_paths(
    cache_dir: Path | None,
    feature_positions: np.ndarray,
    signature: dict,
) -> tuple[Path, Path] | None:
    if cache_dir is None:
        return None
    digest = _signature_digest(signature)[:16]
    start = int(feature_positions[0])
    stop = int(feature_positions[-1]) + 1
    stem = f"local_pretrain_{start:06d}_{stop:06d}_{digest}"
    return cache_dir / f"{stem}.pt", cache_dir / f"{stem}.json"


def _restore_local_pretrain_cache(
    *,
    torch,
    cache_paths: tuple[Path, Path] | None,
    expected_signature: dict,
    model,
    likelihood,
) -> bool:
    if cache_paths is None:
        return False
    state_path, metadata_path = cache_paths
    if not state_path.is_file() or not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("signature") != expected_signature:
        raise ValueError(f"Local-pretraining cache signature mismatch: {metadata_path}")
    states = torch.load(state_path, map_location="cpu", weights_only=True)
    _restore_module_states(states, model=model, likelihood=likelihood)
    return True


def _write_local_pretrain_cache(
    *,
    torch,
    cache_paths: tuple[Path, Path] | None,
    signature: dict,
    model,
    likelihood,
) -> None:
    if cache_paths is None:
        raise ValueError("cache_paths is required when writing local pretraining")
    state_path, metadata_path = cache_paths
    state_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".tmp-{os.getpid()}"
    temporary_state = state_path.with_name(state_path.name + suffix)
    temporary_metadata = metadata_path.with_name(metadata_path.name + suffix)
    torch.save(
        _capture_module_states(model=model, likelihood=likelihood),
        temporary_state,
    )
    temporary_metadata.write_text(
        json.dumps(
            {
                "signature": signature,
                "signature_sha256": _signature_digest(signature),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary_state.replace(state_path)
    temporary_metadata.replace(metadata_path)


def _model_parameter_partition(model):
    variational = list(model.variational_parameters())
    variational_ids = {id(parameter) for parameter in variational}
    hyperparameters = [
        parameter for parameter in model.parameters() if id(parameter) not in variational_ids
    ]
    return variational, hyperparameters


def _build_optimizer(torch, groups: list[tuple[str, list, float]]):
    parameter_ids: dict[int, str] = {}
    optimizer_groups = []
    for name, parameters, learning_rate in groups:
        parameters = [parameter for parameter in parameters if parameter.requires_grad]
        for parameter in parameters:
            previous = parameter_ids.setdefault(id(parameter), name)
            if previous != name:
                raise RuntimeError(
                    f"Parameter appears in optimizer groups {previous!r} and {name!r}"
                )
        optimizer_groups.append(
            {"params": parameters, "lr": float(learning_rate), "name": name}
        )
    return torch.optim.Adam(optimizer_groups), [
        parameter for _, parameters, _ in groups for parameter in parameters
    ]


def _apply_fixed_parameters(model, likelihood, settings, component):
    parameters = settings.fixed_kernel_parameters
    if parameters is None:
        return
    model.covar_module.base_kernel.lengthscale = parameters[f"{component}_length"]
    model.covar_module.outputscale = parameters[f"{component}_scale"]
    model.mean_module.constant = 0.0
    _, hyperparameters = _model_parameter_partition(model)
    for parameter in hyperparameters:
        parameter.requires_grad_(False)
    if likelihood is not None:
        likelihood.noise = parameters["noise"]
        likelihood.requires_grad_(False)


def _stage_epoch_limit(settings, stage, n_training):
    """Explicit update counts take deterministic precedence over epoch fields."""
    update_field = {"global":"global_pretrain_updates", "local":"local_pretrain_updates",
                    "joint":"joint_updates"}[stage]
    budget = getattr(settings, update_field)
    if budget is None:
        return int(getattr(settings, f"{stage}_epochs")), None
    steps = int(np.ceil(n_training/settings.batch_size_spots))
    return int(np.ceil(budget/steps)), int(budget)


def restore_joint_optimizer_checkpoint(torch, checkpoint, *, global_model, local_model,
                                       joint_likelihood, optimizer, expected_identity):
    """Restore a controlled research continuation, rejecting incompatible state.

    ``expected_identity`` must contain the caller's verified training/source/run
    identities. The caller supplies its next observation batch; this helper does
    not claim bitwise continuation of an interrupted epoch permutation.
    Legacy model-only checkpoints still load through ordinary load_state_dict.
    """
    if checkpoint.get("checkpoint_schema") != 2:
        raise ValueError("optimizer continuation requires checkpoint_schema=2")
    for key,value in expected_identity.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"incompatible checkpoint identity: {key}")
    global_model.load_state_dict(checkpoint["global_model"])
    local_model.load_state_dict(checkpoint["local_model"])
    if joint_likelihood is not None:
        joint_likelihood.load_state_dict(checkpoint["joint_likelihood"])
    optimizer.load_state_dict(checkpoint["joint_optimizer"])
    return checkpoint["joint_rng"]


def _pretrain_global_component(
    *,
    torch,
    gpytorch,
    model,
    likelihood,
    coords_t,
    y_cpu,
    settings,
    chunk_id: int,
    history: list[dict],
    hyperparameter_history: list[dict],
    feature_names: np.ndarray,
    feature_positions: np.ndarray,
    exposure_cpu=None,
) -> None:
    variational, hyperparameters = _model_parameter_partition(model)
    likelihood_parameters = [] if likelihood is None else list(likelihood.parameters())
    optimizer, parameters = _build_optimizer(
        torch,
        [
            ("global_variational", variational, settings.global_learning_rate),
            (
                "global_hyperparameters",
                hyperparameters + likelihood_parameters,
                settings.global_hyperparameter_learning_rate,
            ),
        ],
    )
    model.train()
    if likelihood is not None:
        likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=coords_t.shape[0]
        )
    else:
        mll = None
        if exposure_cpu is None:
            raise ValueError("poisson-log global pretraining requires exposure")
    rng = np.random.default_rng(int(settings.random_state) + 1009 * chunk_id)
    best_loss = np.inf
    stale_epochs = 0
    best_epoch = 0
    best_state = None
    epoch_limit, update_budget = _stage_epoch_limit(settings, "global", coords_t.shape[0])
    optimizer_updates = 0
    for epoch in range(epoch_limit):
        permutation = rng.permutation(coords_t.shape[0])
        examples_processed = 0
        conditional_terms_processed = 0
        losses = []
        for start in range(0, coords_t.shape[0], int(settings.batch_size_spots)):
            indices_np = permutation[start : start + int(settings.batch_size_spots)]
            indices = torch.as_tensor(indices_np, dtype=torch.long, device=coords_t.device)
            if update_budget is not None and optimizer_updates >= update_budget:
                break
            examples_processed += len(indices_np)
            target = y_cpu[:, indices_np].to(device=coords_t.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            distribution = model(coords_t.index_select(0, indices))
            if settings.observation_model == "gaussian":
                loss = (-mll(distribution, target)).sum()
            else:
                batch_exposure = exposure_cpu[indices_np].to(
                    device=coords_t.device, non_blocking=True
                )
                expected_log_likelihood = expected_additive_poisson_log_likelihood(
                    target,
                    distribution.mean,
                    distribution.variance,
                    torch.zeros_like(distribution.mean),
                    torch.zeros_like(distribution.variance),
                    batch_exposure,
                ).mean(dim=-1)
                global_kl = model.variational_strategy.kl_divergence()
                loss = -(
                    expected_log_likelihood
                    - global_kl / float(coords_t.shape[0])
                ).sum()
            _backward_step(
                torch,
                loss,
                optimizer,
                parameters,
                gradient_clip=None,
                stage="global pretraining",
            )
            optimizer_updates += 1
            losses.append(float(loss.detach().cpu()))
        _assert_finite_model(torch, model, likelihood, "global pretraining")
        mean_loss = float(np.mean(losses))
        improved = mean_loss < best_loss - float(settings.early_stopping_min_delta)
        if int(settings.early_stopping_patience) > 0 and improved:
            best_state = _capture_module_states(
                **(
                    {"model": model, "likelihood": likelihood}
                    if likelihood is not None
                    else {"model": model}
                )
            )
            best_epoch = int(epoch + 1)
        best_loss, stale_epochs, should_stop = _update_early_stopping(
            mean_loss,
            best_loss,
            stale_epochs,
            patience=settings.early_stopping_patience,
            min_delta=settings.early_stopping_min_delta,
        )
        history.append(
            {
                "stage": "global_pretrain",
                "chunk_id": int(chunk_id),
                "epoch": int(epoch + 1),
                "optimizer_updates": len(losses),
                "cumulative_optimizer_updates": optimizer_updates,
                "likelihood_examples_processed": examples_processed,
                "local_conditional_terms_processed": conditional_terms_processed,
                "mean_negative_elbo": mean_loss,
                "early_stopping_best": float(best_loss),
                "early_stopping_best_epoch": int(best_epoch),
                "early_stopping_stale_epochs": int(stale_epochs),
                "stopped_early": bool(should_stop),
                "restored_best_state": False,
            }
        )
        _record_pretrain_hyperparameters(
            hyperparameter_history,
            stage="global_pretrain",
            chunk_id=chunk_id,
            epoch=epoch + 1,
            feature_names=feature_names,
            feature_positions=feature_positions,
            parameters=_component_parameters(model, likelihood),
        )
        if should_stop:
            print(
                f"Global pretraining chunk {chunk_id} stopped after {epoch + 1} epochs",
                flush=True,
            )
            break
    if best_state is not None:
        _restore_module_states(
            best_state,
            **(
                {"model": model, "likelihood": likelihood}
                if likelihood is not None
                else {"model": model}
            ),
        )
        history[-1]["restored_best_state"] = True
        history[-1]["restored_best_epoch"] = int(best_epoch)


    model._pretrain_optimizer_state = optimizer.state_dict()


def _initialize_local_variational_distribution(model) -> None:
    strategy = model.variational_strategy
    if strategy.variational_params_initialized.item():
        return
    strategy._variational_distribution.variational_mean.data.zero_()
    strategy._variational_distribution._variational_stddev.data.fill_(1.0e-2)
    strategy.variational_params_initialized.fill_(1)


def _pretrain_local_component(
    *,
    torch,
    model,
    likelihood,
    target_cpu,
    settings,
    chunk_id: int,
    history: list[dict],
    hyperparameter_history: list[dict],
    feature_names: np.ndarray,
    feature_positions: np.ndarray,
    exposure_cpu=None,
    fixed_global_mean_cpu=None,
    fixed_global_variance_cpu=None,
) -> None:
    variational, hyperparameters = _model_parameter_partition(model)
    likelihood_parameters = [] if likelihood is None else list(likelihood.parameters())
    optimizer, parameters = _build_optimizer(
        torch,
        [
            ("local_variational", variational, settings.local_learning_rate),
            (
                "local_hyperparameters",
                hyperparameters + likelihood_parameters,
                settings.local_hyperparameter_learning_rate,
            ),
        ],
    )
    model.train()
    if likelihood is not None:
        likelihood.train()
    elif (
        exposure_cpu is None
        or fixed_global_mean_cpu is None
        or fixed_global_variance_cpu is None
    ):
        raise ValueError(
            "poisson-log local pretraining requires exposure and fixed global moments"
        )
    n_training = target_cpu.shape[1]
    rng = np.random.default_rng(int(settings.random_state) + 2017 * chunk_id)
    generator = _torch_generator(torch, model.variational_strategy.inducing_points.device, 3001 + chunk_id)
    best_loss = np.inf
    stale_epochs = 0
    best_epoch = 0
    best_state = None
    n_steps = int(np.ceil(n_training / int(settings.batch_size_spots)))
    epoch_limit, update_budget = _stage_epoch_limit(settings, "local", n_training)
    optimizer_updates = 0
    for epoch in range(epoch_limit):
        permutation = rng.permutation(n_training)
        examples_processed = 0
        conditional_terms_processed = 0
        kl_batches = (
            _local_kl_epoch_batches(
                torch,
                model.variational_strategy,
                n_steps,
                generator,
            )
            if settings.local_kl_mode == "stochastic" and update_budget is None
            else (None,) * n_steps
        )
        losses = []
        for step, start in enumerate(
            range(0, n_training, int(settings.batch_size_spots))
        ):
            indices_np = permutation[start : start + int(settings.batch_size_spots)]
            indices = torch.as_tensor(
                indices_np,
                dtype=torch.long,
                device=model.variational_strategy.inducing_points.device,
            )
            if update_budget is not None and optimizer_updates >= update_budget:
                break
            examples_processed += len(indices_np)
            target = target_cpu[:, indices_np].to(
                device=indices.device, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            local_mean, local_variance = model.variational_strategy.training_marginals(
                indices
            )
            if settings.observation_model == "gaussian":
                expected_log_likelihood = expected_additive_gaussian_log_likelihood(
                    target,
                    torch.zeros_like(local_mean),
                    torch.zeros_like(local_variance),
                    local_mean,
                    local_variance,
                    likelihood.noise,
                ).mean(dim=-1)
            else:
                global_mean = fixed_global_mean_cpu[:, indices_np].to(
                    device=indices.device, non_blocking=True
                )
                global_variance = fixed_global_variance_cpu[:, indices_np].to(
                    device=indices.device, non_blocking=True
                )
                batch_exposure = exposure_cpu[indices_np].to(
                    device=indices.device, non_blocking=True
                )
                expected_log_likelihood = expected_additive_poisson_log_likelihood(
                    target,
                    global_mean,
                    global_variance,
                    local_mean,
                    local_variance,
                    batch_exposure,
                ).mean(dim=-1)
            n_cond = int(model.variational_strategy.M - model.variational_strategy.k)
            conditional_terms_processed += (n_cond if settings.local_kl_mode == "full"
                or settings.local_kl_batch_size >= n_cond else
                (len(kl_batches[step]) if kl_batches[step] is not None else min(settings.local_kl_batch_size,n_cond)))
            local_kl = model.variational_strategy.local_kl(
                mode=settings.local_kl_mode,
                batch_size=settings.local_kl_batch_size,
                generator=generator,
                conditional_indices=kl_batches[step],
            )
            loss = -(
                expected_log_likelihood - local_kl / float(n_training)
            ).sum()
            _backward_step(
                torch,
                loss,
                optimizer,
                parameters,
                gradient_clip=None,
                stage="local pretraining",
            )
            optimizer_updates += 1
            losses.append(float(loss.detach().cpu()))
        _assert_finite_model(torch, model, likelihood, "local pretraining")
        mean_loss = float(np.mean(losses))
        improved = mean_loss < best_loss - float(settings.early_stopping_min_delta)
        if int(settings.early_stopping_patience) > 0 and improved:
            best_state = _capture_module_states(
                **(
                    {"model": model, "likelihood": likelihood}
                    if likelihood is not None
                    else {"model": model}
                )
            )
            best_epoch = int(epoch + 1)
        best_loss, stale_epochs, should_stop = _update_early_stopping(
            mean_loss,
            best_loss,
            stale_epochs,
            patience=settings.early_stopping_patience,
            min_delta=settings.early_stopping_min_delta,
        )
        history.append(
            {
                "stage": "local_pretrain",
                "chunk_id": int(chunk_id),
                "epoch": int(epoch + 1),
                "optimizer_updates": len(losses),
                "cumulative_optimizer_updates": optimizer_updates,
                "likelihood_examples_processed": examples_processed,
                "local_conditional_terms_processed": conditional_terms_processed,
                "mean_negative_elbo": mean_loss,
                "early_stopping_best": float(best_loss),
                "early_stopping_best_epoch": int(best_epoch),
                "early_stopping_stale_epochs": int(stale_epochs),
                "stopped_early": bool(should_stop),
                "restored_best_state": False,
            }
        )
        _record_pretrain_hyperparameters(
            hyperparameter_history,
            stage="local_pretrain",
            chunk_id=chunk_id,
            epoch=epoch + 1,
            feature_names=feature_names,
            feature_positions=feature_positions,
            parameters=_component_parameters(model, likelihood),
        )
        if should_stop:
            print(
                f"Local pretraining chunk {chunk_id} stopped after {epoch + 1} epochs",
                flush=True,
            )
            break
    if best_state is not None:
        _restore_module_states(
            best_state,
            **(
                {"model": model, "likelihood": likelihood}
                if likelihood is not None
                else {"model": model}
            ),
        )
        history[-1]["restored_best_state"] = True
        history[-1]["restored_best_epoch"] = int(best_epoch)


    model._pretrain_optimizer_state = optimizer.state_dict()


def _jointly_train_components(
    *,
    torch,
    global_model,
    local_model,
    joint_likelihood,
    coords_t,
    y_cpu,
    settings,
    chunk_id: int,
    history: list[dict],
    hyperparameter_history: list[dict],
    feature_names: np.ndarray,
    feature_positions: np.ndarray,
    exposure_cpu=None,
) -> None:
    global_variational, global_hyperparameters = _model_parameter_partition(global_model)
    local_variational, local_hyperparameters = _model_parameter_partition(local_model)
    optimizer_groups = [
            (
                "global_variational",
                global_variational,
                settings.joint_global_variational_learning_rate,
            ),
            (
                "global_hyperparameters",
                global_hyperparameters,
                settings.joint_global_hyperparameter_learning_rate,
            ),
            (
                "local_variational",
                local_variational,
                settings.joint_local_variational_learning_rate,
            ),
            (
                "local_hyperparameters",
                local_hyperparameters,
                settings.joint_local_hyperparameter_learning_rate,
            ),
        ]
    if joint_likelihood is not None:
        optimizer_groups.append(
            (
                "joint_likelihood",
                list(joint_likelihood.parameters()),
                settings.joint_likelihood_learning_rate,
            )
        )
    elif exposure_cpu is None:
        raise ValueError("poisson-log joint training requires exposure")
    optimizer, parameters = _build_optimizer(torch, optimizer_groups)
    global_model.train()
    local_model.train()
    if joint_likelihood is not None:
        joint_likelihood.train()
    n_training = coords_t.shape[0]
    rng = np.random.default_rng(int(settings.random_state) + 4001 * chunk_id)
    generator = _torch_generator(torch, coords_t.device, 5003 + chunk_id)
    best_loss = np.inf
    stale_epochs = 0
    best_epoch = 0
    best_state = None
    n_steps = int(np.ceil(n_training / int(settings.batch_size_spots)))

    epoch_limit, update_budget = _stage_epoch_limit(settings, "joint", n_training)
    optimizer_updates = 0
    for epoch in range(epoch_limit):
        permutation = rng.permutation(n_training)
        examples_processed = 0
        conditional_terms_processed = 0
        kl_batches = (
            _local_kl_epoch_batches(
                torch,
                local_model.variational_strategy,
                n_steps,
                generator,
            )
            if settings.local_kl_mode == "stochastic" and update_budget is None
            else (None,) * n_steps
        )
        aggregates = {
            "expected_log_likelihood": [],
            "global_kl": [],
            "local_kl": [],
            "negative_joint_elbo": [],
            "gradient_norm": [],
        }
        epoch_started = time.time()
        for step, start in enumerate(
            range(0, n_training, int(settings.batch_size_spots))
        ):
            indices_np = permutation[start : start + int(settings.batch_size_spots)]
            indices = torch.as_tensor(
                indices_np, dtype=torch.long, device=coords_t.device
            )
            if update_budget is not None and optimizer_updates >= update_budget:
                break
            examples_processed += len(indices_np)
            target = y_cpu[:, indices_np].to(device=coords_t.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            global_distribution = global_model(coords_t.index_select(0, indices))
            global_mean = global_distribution.mean
            global_variance = global_distribution.variance
            local_mean, local_variance = local_model.variational_strategy.training_marginals(
                indices
            )
            _assert_positive_finite_variance(
                torch, global_variance, "global training posterior variance"
            )
            _assert_positive_finite_variance(
                torch, local_variance, "local training posterior variance"
            )
            global_kl = global_model.variational_strategy.kl_divergence()
            n_cond = int(local_model.variational_strategy.M - local_model.variational_strategy.k)
            conditional_terms_processed += (n_cond if settings.local_kl_mode == "full"
                or settings.local_kl_batch_size >= n_cond else
                (len(kl_batches[step]) if kl_batches[step] is not None else min(settings.local_kl_batch_size,n_cond)))
            local_kl = local_model.variational_strategy.local_kl(
                mode=settings.local_kl_mode,
                batch_size=settings.local_kl_batch_size,
                generator=generator,
                conditional_indices=kl_batches[step],
            )
            if settings.observation_model == "gaussian":
                expected_log_likelihood = expected_additive_gaussian_log_likelihood(
                    target,
                    global_mean,
                    global_variance,
                    local_mean,
                    local_variance,
                    joint_likelihood.noise,
                )
            else:
                batch_exposure = exposure_cpu[indices_np].to(
                    device=coords_t.device, non_blocking=True
                )
                expected_log_likelihood = expected_additive_poisson_log_likelihood(
                    target,
                    global_mean,
                    global_variance,
                    local_mean,
                    local_variance,
                    batch_exposure,
                )
            elbo = (
                expected_log_likelihood.mean(dim=-1)
                - global_kl / float(n_training)
                - local_kl / float(n_training)
            )
            loss = -elbo.sum()
            gradient_norm = _backward_step(
                torch,
                loss,
                optimizer,
                parameters,
                gradient_clip=settings.joint_gradient_clip,
                stage="joint fine-tuning",
            )
            optimizer_updates += 1
            aggregates["expected_log_likelihood"].append(
                float(expected_log_likelihood.mean().detach().cpu())
            )
            aggregates["global_kl"].append(float(global_kl.mean().detach().cpu()))
            aggregates["local_kl"].append(float(local_kl.mean().detach().cpu()))
            aggregates["negative_joint_elbo"].append(float(loss.detach().cpu()))
            aggregates["gradient_norm"].append(float(gradient_norm))

        _assert_finite_model(torch, global_model, None, "joint global model")
        _assert_finite_model(torch, local_model, joint_likelihood, "joint local model")
        parameters_now = _joint_component_parameters(
            global_model, local_model, joint_likelihood
        )
        mean_loss = float(np.mean(aggregates["negative_joint_elbo"]))
        improved = mean_loss < best_loss - float(settings.early_stopping_min_delta)
        if int(settings.early_stopping_patience) > 0 and improved:
            state_modules = {
                "global_model": global_model,
                "local_model": local_model,
            }
            if joint_likelihood is not None:
                state_modules["joint_likelihood"] = joint_likelihood
            best_state = _capture_module_states(**state_modules)
            best_epoch = int(epoch + 1)
        best_loss, stale_epochs, should_stop = _update_early_stopping(
            mean_loss,
            best_loss,
            stale_epochs,
            patience=settings.early_stopping_patience,
            min_delta=settings.early_stopping_min_delta,
        )
        history.append(
            {
                "chunk_id": int(chunk_id),
                "epoch": int(epoch + 1),
                "optimizer_updates": len(aggregates["negative_joint_elbo"]),
                "cumulative_optimizer_updates": optimizer_updates,
                "likelihood_examples_processed": examples_processed,
                "local_conditional_terms_processed": conditional_terms_processed,
                "expected_log_likelihood": float(
                    np.mean(aggregates["expected_log_likelihood"])
                ),
                "global_kl": float(np.mean(aggregates["global_kl"])),
                "local_kl": float(np.mean(aggregates["local_kl"])),
                "total_kl": float(
                    np.mean(aggregates["global_kl"])
                    + np.mean(aggregates["local_kl"])
                ),
                "negative_joint_elbo": mean_loss,
                "global_length_scale_median": float(
                    np.median(parameters_now["global_length_scale"])
                ),
                "local_length_scale_median": float(
                    np.median(parameters_now["local_length_scale"])
                ),
                "global_outputscale_median": float(
                    np.median(parameters_now["global_outputscale"])
                ),
                "local_outputscale_median": float(
                    np.median(parameters_now["local_outputscale"])
                ),
                "joint_noise_median": float(
                    np.median(parameters_now["joint_noise"])
                ),
                "gradient_norm": float(np.mean(aggregates["gradient_norm"])),
                "elapsed_seconds": float(time.time() - epoch_started),
                "early_stopping_best": float(best_loss),
                "early_stopping_best_epoch": int(best_epoch),
                "early_stopping_stale_epochs": int(stale_epochs),
                "stopped_early": bool(should_stop),
                "restored_best_state": False,
            }
        )
        _record_joint_hyperparameters(
            hyperparameter_history,
            chunk_id=chunk_id,
            epoch=epoch + 1,
            feature_names=feature_names,
            feature_positions=feature_positions,
            parameters=parameters_now,
        )
        print(
            f"Joint global-local chunk {chunk_id} epoch {epoch + 1}/"
            f"{epoch_limit} negative_elbo="
            f"{np.mean(aggregates['negative_joint_elbo']):.6g}",
            flush=True,
        )
        if should_stop:
            print(
                f"Joint fine-tuning chunk {chunk_id} stopped after {epoch + 1} epochs",
                flush=True,
            )
            break
    if best_state is not None:
        state_modules = {
            "global_model": global_model,
            "local_model": local_model,
        }
        if joint_likelihood is not None:
            state_modules["joint_likelihood"] = joint_likelihood
        _restore_module_states(
            best_state,
            **state_modules,
        )
        history[-1]["restored_best_state"] = True
        history[-1]["restored_best_epoch"] = int(best_epoch)


    global_model._joint_optimizer_state = optimizer.state_dict()
    global_model._joint_rng_state = {"numpy": rng.bit_generator.state,
        "torch": generator.get_state(), "updates": optimizer_updates}


def _evaluate_local_posterior(
    *,
    torch,
    model,
    coords_t,
    prediction_coords_t,
    ordered_prediction_rows,
    n_prediction_spots: int,
    eval_batch_size: int,
):
    strategy = model.variational_strategy
    all_indices = torch.arange(coords_t.shape[0], device=coords_t.device)
    training_mean, training_variance = strategy.training_marginals(all_indices)
    n_features = training_mean.shape[0]
    means = np.full((n_prediction_spots, n_features), np.nan, dtype=np.float32)
    variances = np.full((n_prediction_spots, n_features), np.nan, dtype=np.float32)
    means[ordered_prediction_rows] = training_mean.transpose(0, 1).detach().cpu().numpy()
    variances[ordered_prediction_rows] = (
        training_variance.transpose(0, 1).detach().cpu().numpy()
    )

    heldout_rows = np.flatnonzero(~np.isfinite(means[:, 0]))
    if heldout_rows.size:
        strategy.nn_util.set_nn_idx(coords_t.float())
        model.eval()
        with torch.no_grad():
            for start in range(0, heldout_rows.size, int(eval_batch_size)):
                rows = heldout_rows[start : start + int(eval_batch_size)]
                posterior = model(prediction_coords_t[rows])
                means[rows] = posterior.mean.transpose(0, 1).detach().cpu().numpy()
                variances[rows] = (
                    posterior.variance.transpose(0, 1).detach().cpu().numpy()
                )
    return means, variances


def _backward_step(
    torch,
    loss,
    optimizer,
    parameters,
    *,
    gradient_clip,
    stage: str,
) -> float:
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss during {stage}")
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError(f"Missing or non-finite gradients during {stage}")
    if gradient_clip is not None:
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, float(gradient_clip))
    else:
        gradient_norm = torch.sqrt(
            sum(gradient.detach().square().sum() for gradient in gradients)
        )
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"Non-finite gradient norm during {stage}")
    optimizer.step()
    return float(gradient_norm.detach().cpu())


def _assert_finite_model(torch, model, likelihood, stage: str) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Non-finite {name} during {stage}")
    length_scale = model.covar_module.base_kernel.lengthscale
    outputscale = model.covar_module.outputscale
    if not torch.isfinite(length_scale).all() or not torch.isfinite(outputscale).all():
        raise FloatingPointError(f"Non-finite kernel parameters during {stage}")
    if likelihood is not None and hasattr(likelihood, "noise"):
        noise = likelihood.noise
        if not torch.isfinite(noise).all() or not torch.all(noise > 0):
            raise FloatingPointError(f"Invalid likelihood noise during {stage}")


def _assert_positive_finite_variance(torch, variance, name: str) -> None:
    if not torch.isfinite(variance).all() or not torch.all(variance > 0):
        raise FloatingPointError(f"{name} must be finite and positive")


def _local_kl_epoch_batches(torch, strategy, n_steps: int, generator) -> tuple:
    """Partition one conditional-KL permutation across an observation epoch."""
    n_steps = int(n_steps)
    n_conditionals = int(strategy.M - strategy.k)
    if n_steps <= 0 or n_conditionals <= 0:
        raise ValueError("KL scheduling requires positive steps and conditionals")
    device = strategy.inducing_points.device
    if n_steps > n_conditionals:
        return tuple(
            torch.randint(
                n_conditionals,
                (1,),
                device=device,
                generator=generator,
            )
            for _ in range(n_steps)
        )
    permutation = torch.randperm(
        n_conditionals,
        device=device,
        generator=generator,
    )
    return tuple(torch.tensor_split(permutation, n_steps))


def _capture_module_states(**modules) -> dict[str, dict]:
    """Clone module state dictionaries to CPU for memory-safe restoration."""
    return {
        name: {
            key: value.detach().cpu().clone()
            for key, value in module.state_dict().items()
        }
        for name, module in modules.items()
    }


def _restore_module_states(states: dict[str, dict], **modules) -> None:
    for name, module in modules.items():
        module.load_state_dict(states[name])


def _update_early_stopping(
    loss: float,
    best_loss: float,
    stale_epochs: int,
    *,
    patience: int,
    min_delta: float,
) -> tuple[float, int, bool]:
    """Update a stage-local patience counter from the mean epoch loss."""
    if loss < best_loss - float(min_delta):
        return float(loss), 0, False
    stale_epochs = int(stale_epochs) + 1
    should_stop = int(patience) > 0 and stale_epochs >= int(patience)
    return float(best_loss), stale_epochs, should_stop


def _record_pretrain_hyperparameters(
    history: list[dict],
    *,
    stage: str,
    chunk_id: int,
    epoch: int,
    feature_names: np.ndarray,
    feature_positions: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> None:
    for column, (gene, feature_position) in enumerate(
        zip(feature_names, feature_positions, strict=True)
    ):
        history.append(
            {
                "stage": str(stage),
                "chunk_id": int(chunk_id),
                "epoch": int(epoch),
                "gene": str(gene),
                "feature_position": int(feature_position),
                "global_length_scale": (
                    float(parameters["length_scale"][column])
                    if stage == "global_pretrain"
                    else np.nan
                ),
                "global_outputscale": (
                    float(parameters["outputscale"][column])
                    if stage == "global_pretrain"
                    else np.nan
                ),
                "local_length_scale": (
                    float(parameters["length_scale"][column])
                    if stage == "local_pretrain"
                    else np.nan
                ),
                "local_outputscale": (
                    float(parameters["outputscale"][column])
                    if stage == "local_pretrain"
                    else np.nan
                ),
                "noise_variance": float(parameters["noise"][column]),
            }
        )


def _record_joint_hyperparameters(
    history: list[dict],
    *,
    chunk_id: int,
    epoch: int,
    feature_names: np.ndarray,
    feature_positions: np.ndarray,
    parameters: dict[str, np.ndarray],
) -> None:
    for column, (gene, feature_position) in enumerate(
        zip(feature_names, feature_positions, strict=True)
    ):
        history.append(
            {
                "stage": "joint_finetune",
                "chunk_id": int(chunk_id),
                "epoch": int(epoch),
                "gene": str(gene),
                "feature_position": int(feature_position),
                "global_length_scale": float(
                    parameters["global_length_scale"][column]
                ),
                "global_outputscale": float(
                    parameters["global_outputscale"][column]
                ),
                "local_length_scale": float(
                    parameters["local_length_scale"][column]
                ),
                "local_outputscale": float(
                    parameters["local_outputscale"][column]
                ),
                "noise_variance": float(parameters["joint_noise"][column]),
            }
        )


def _component_parameters(model, likelihood) -> dict[str, np.ndarray]:
    length_scale = model.covar_module.base_kernel.lengthscale.detach()
    result = {
        "length_scale": length_scale
        .cpu()
        .reshape(-1)
        .numpy()
        .astype(float),
        "outputscale": model.covar_module.outputscale.detach()
        .cpu()
        .reshape(-1)
        .numpy()
        .astype(float),
    }
    result["noise"] = (
        likelihood.noise.detach().cpu().reshape(-1).numpy().astype(float)
        if likelihood is not None and hasattr(likelihood, "noise")
        else np.full(result["length_scale"].shape, np.nan, dtype=float)
    )
    return result


def _joint_component_parameters(global_model, local_model, joint_likelihood):
    result = {
        "global_length_scale": global_model.covar_module.base_kernel.lengthscale.detach()
        .cpu()
        .reshape(-1)
        .numpy()
        .astype(float),
        "global_outputscale": global_model.covar_module.outputscale.detach()
        .cpu()
        .reshape(-1)
        .numpy()
        .astype(float),
        "local_length_scale": local_model.covar_module.base_kernel.lengthscale.detach()
        .cpu()
        .reshape(-1)
        .numpy()
        .astype(float),
        "local_outputscale": local_model.covar_module.outputscale.detach()
        .cpu()
        .reshape(-1)
        .numpy()
        .astype(float),
    }
    result["joint_noise"] = (
        joint_likelihood.noise.detach().cpu().reshape(-1).numpy().astype(float)
        if joint_likelihood is not None and hasattr(joint_likelihood, "noise")
        else np.full(result["global_length_scale"].shape, np.nan, dtype=float)
    )
    return result


def _validate_output_arrays(
    posterior_mean,
    posterior_variance,
    posterior_predictive_variance,
    global_mean,
    global_variance,
    local_mean,
    local_variance,
) -> None:
    for name, values in (
        ("posterior_mean", posterior_mean),
        ("posterior_variance", posterior_variance),
        ("posterior_predictive_variance", posterior_predictive_variance),
        ("global_posterior_mean", global_mean),
        ("global_posterior_variance", global_variance),
        ("local_posterior_mean", local_mean),
        ("local_posterior_variance", local_variance),
    ):
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{name} contains non-finite values")
    if not np.all(global_variance > 0) or not np.all(local_variance > 0):
        raise FloatingPointError("Component posterior variances must be positive")
    if not np.all(posterior_variance > 0):
        raise FloatingPointError("Combined posterior variance must be positive")
    if not np.all(posterior_predictive_variance > posterior_variance):
        raise FloatingPointError("Predictive variance must include positive joint noise")


def _torch_generator(torch, device, seed: int):
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def _start_stage(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        memory_started = int(torch.cuda.memory_allocated(device))
    else:
        memory_started = 0
    return time.time(), memory_started


def _finish_stage(torch, device, chunk_id, stage, started, memory_started):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    return {
        "chunk_id": int(chunk_id),
        "stage": str(stage),
        "elapsed_seconds": float(time.time() - started),
        "starting_gpu_memory_bytes": int(memory_started),
        "peak_gpu_memory_bytes": int(peak_memory),
    }
