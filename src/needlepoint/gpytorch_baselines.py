"""Reference GP regressors for TEMI scalability and fidelity benchmarks.

These models are intentionally conventional rather than TEMI-specific:

* ``exact_dense`` is an exact Gaussian RBF GP and is restricted to small data.
* ``standard_svgp`` is a minibatched inducing-point variational GP with a dense
  Cholesky variational distribution and unstructured inducing locations.

Both write the same posterior-map schema as the V-SKI implementations so the
existing blocked-holdout fidelity evaluator can compare methods directly.
"""

from __future__ import annotations

from needlepoint.storage import GPStorageSettings, encode_gp_maps, storage_manifest

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans

from needlepoint.gpytorch_vski import (
    _choose_device,
    _require_gpytorch,
    _standardize_columns_with_stats,
)
from needlepoint.svg import _as_dense_float


@dataclass
class ReferenceGPSettings(GPStorageSettings):
    """Configuration shared by dense exact GP and conventional SVGP fits."""

    n_epochs: int = 30
    batch_size_spots: int = 4096
    eval_batch_size_spots: int = 8192
    peak_chunk_size: int = 4
    learning_rate: float = 0.01
    initial_length_scale: float = 160.0
    n_inducing: int = 256
    inducing_selection: str = "minibatch-kmeans"
    learn_inducing_locations: bool = False
    exact_max_spots: int = 3000
    dtype: str = "float32"
    device: str | None = None
    random_state: int = 0

    def validate(self) -> None:
        if self.n_epochs <= 0:
            raise ValueError("n_epochs must be positive")
        if min(self.batch_size_spots, self.eval_batch_size_spots, self.peak_chunk_size) <= 0:
            raise ValueError("batch and feature chunk sizes must be positive")
        if self.learning_rate <= 0 or self.initial_length_scale <= 0:
            raise ValueError("learning rate and initial length scale must be positive")
        if self.n_inducing <= 1:
            raise ValueError("n_inducing must exceed one")
        if self.exact_max_spots <= 1:
            raise ValueError("exact_max_spots must exceed one")
        if self.inducing_selection not in {"minibatch-kmeans", "random"}:
            raise ValueError("inducing_selection must be minibatch-kmeans or random")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.output_dtype not in {"float32", "float16"}:
            raise ValueError("output_dtype must be float32 or float16")


def fit_exact_gp_spatial_maps(
    expression: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    feature_names: np.ndarray,
    output_dir: str | Path,
    *,
    settings: ReferenceGPSettings | None = None,
    prediction_coords: np.ndarray | None = None,
    prediction_spot_indices: np.ndarray | None = None,
) -> dict:
    """Fit independent exact Gaussian RBF GPs in feature batches."""

    settings = settings or ReferenceGPSettings()
    settings.validate()
    y_raw, coords_np, prediction_coords_np, feature_names = _validate_inputs(
        expression, coords, prediction_coords, feature_names
    )
    if coords_np.shape[0] > settings.exact_max_spots:
        raise ValueError(
            f"exact_dense is limited to {settings.exact_max_spots} training spots; "
            f"received {coords_np.shape[0]}"
        )
    y_np, means, stds = _standardize_columns_with_stats(y_raw)
    torch, gpytorch = _require_gpytorch()
    device = _choose_device(torch, settings.device)
    dtype = getattr(torch, settings.dtype)
    started = time.time()
    torch.manual_seed(settings.random_state)
    np.random.seed(settings.random_state)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train = torch.as_tensor(coords_np, device=device, dtype=dtype)
    x_pred = torch.as_tensor(prediction_coords_np, device=device, dtype=dtype)
    metadata_records: list[pd.DataFrame] = []
    chunk_records: list[dict] = []
    history_records: list[dict] = []

    for chunk_id, start in enumerate(range(0, y_np.shape[1], settings.peak_chunk_size)):
        stop = min(start + settings.peak_chunk_size, y_np.shape[1])
        y_chunk = torch.as_tensor(y_np[:, start:stop].T, device=device, dtype=dtype)
        n_features = stop - start
        batch_shape = torch.Size([n_features])
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape).to(device)

        class BatchedExactGP(gpytorch.models.ExactGP):
            def __init__(self):
                super().__init__(x_train, y_chunk, likelihood)
                self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
                    batch_shape=batch_shape,
                )

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        model = BatchedExactGP().to(device=device, dtype=dtype)
        model.covar_module.base_kernel.lengthscale = float(settings.initial_length_scale)
        likelihood.noise = 0.5
        # ExactGP registers its likelihood, so model.parameters() already
        # contains the noise parameter.
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for epoch in range(1, settings.n_epochs + 1):
            model.train()
            likelihood.train()
            optimizer.zero_grad(set_to_none=True)
            loss = -mll(model(x_train), y_chunk).sum()
            loss.backward()
            optimizer.step()
            history_records.append(
                {
                    "chunk_id": chunk_id,
                    "epoch": epoch,
                    "mean_negative_objective": float(loss.detach().cpu()) / n_features,
                }
            )
        arrays = _predict_exact(
            torch, gpytorch, model, likelihood, x_pred, settings.eval_batch_size_spots
        )
        metadata, chunk = _write_chunk(
            output_dir,
            "exact_dense",
            chunk_id,
            start,
            stop,
            feature_names,
            means,
            stds,
            arrays,
            model,
            likelihood,
            coords_np.shape[0],
            settings,
        )
        metadata_records.append(metadata)
        chunk_records.append(chunk)
        del model, likelihood, optimizer, mll, y_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return _finish_outputs(
        output_dir,
        method="dense exact Gaussian RBF GP",
        backend="exact_dense",
        settings=settings,
        coords=coords_np,
        prediction_coords=prediction_coords_np,
        prediction_spot_indices=prediction_spot_indices,
        metadata_records=metadata_records,
        chunk_records=chunk_records,
        history_records=history_records,
        elapsed_seconds=time.time() - started,
        extra={"computational_scope": "small-N accuracy anchor; cubic dense covariance"},
    )


def fit_standard_svgp_spatial_maps(
    expression: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    feature_names: np.ndarray,
    output_dir: str | Path,
    *,
    settings: ReferenceGPSettings | None = None,
    prediction_coords: np.ndarray | None = None,
    prediction_spot_indices: np.ndarray | None = None,
) -> dict:
    """Fit a conventional unstructured inducing-point Gaussian SVGP."""

    settings = settings or ReferenceGPSettings()
    settings.validate()
    y_raw, coords_np, prediction_coords_np, feature_names = _validate_inputs(
        expression, coords, prediction_coords, feature_names
    )
    y_np, means, stds = _standardize_columns_with_stats(y_raw)
    torch, gpytorch = _require_gpytorch()
    device = _choose_device(torch, settings.device)
    dtype = getattr(torch, settings.dtype)
    started = time.time()
    torch.manual_seed(settings.random_state)
    rng = np.random.default_rng(settings.random_state)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inducing_np = _select_inducing_points(coords_np, settings, rng)
    np.save(output_dir / "standard_svgp_inducing_points.npy", inducing_np)
    x_train = torch.as_tensor(coords_np, device=device, dtype=dtype)
    x_pred = torch.as_tensor(prediction_coords_np, device=device, dtype=dtype)
    inducing = torch.as_tensor(inducing_np, device=device, dtype=dtype)
    metadata_records: list[pd.DataFrame] = []
    chunk_records: list[dict] = []
    history_records: list[dict] = []

    for chunk_id, start in enumerate(range(0, y_np.shape[1], settings.peak_chunk_size)):
        stop = min(start + settings.peak_chunk_size, y_np.shape[1])
        n_features = stop - start
        batch_shape = torch.Size([n_features])
        q_dist = gpytorch.variational.CholeskyVariationalDistribution(
            inducing.shape[0], batch_shape=batch_shape
        )

        class BatchedSVGP(gpytorch.models.ApproximateGP):
            def __init__(self):
                strategy = gpytorch.variational.VariationalStrategy(
                    self,
                    inducing,
                    q_dist,
                    learn_inducing_locations=settings.learn_inducing_locations,
                )
                super().__init__(strategy)
                self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
                    batch_shape=batch_shape,
                )

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x), self.covar_module(x)
                )

        model = BatchedSVGP().to(device=device, dtype=dtype)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape).to(
            device=device, dtype=dtype
        )
        model.covar_module.base_kernel.lengthscale = float(settings.initial_length_scale)
        likelihood.noise = 0.5
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()),
            lr=settings.learning_rate,
        )
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model, num_data=coords_np.shape[0]
        )
        order = np.arange(coords_np.shape[0], dtype=np.int64)
        for epoch in range(1, settings.n_epochs + 1):
            rng.shuffle(order)
            epoch_loss = 0.0
            seen = 0
            model.train()
            likelihood.train()
            for offset in range(0, order.size, settings.batch_size_spots):
                rows = order[offset : offset + settings.batch_size_spots]
                xb = x_train[rows]
                yb = torch.as_tensor(y_np[rows, start:stop].T, device=device, dtype=dtype)
                optimizer.zero_grad(set_to_none=True)
                loss = -mll(model(xb), yb).sum()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu()) * rows.size
                seen += rows.size
            history_records.append(
                {
                    "chunk_id": chunk_id,
                    "epoch": epoch,
                    "mean_negative_objective": epoch_loss / max(seen * n_features, 1),
                }
            )
        arrays = _predict_variational(
            torch, gpytorch, model, likelihood, x_pred, settings.eval_batch_size_spots
        )
        metadata, chunk = _write_chunk(
            output_dir,
            "standard_svgp",
            chunk_id,
            start,
            stop,
            feature_names,
            means,
            stds,
            arrays,
            model,
            likelihood,
            coords_np.shape[0],
            settings,
        )
        metadata_records.append(metadata)
        chunk_records.append(chunk)
        del model, likelihood, optimizer, mll
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return _finish_outputs(
        output_dir,
        method="standard unstructured inducing-point Gaussian SVGP",
        backend="standard_svgp",
        settings=settings,
        coords=coords_np,
        prediction_coords=prediction_coords_np,
        prediction_spot_indices=prediction_spot_indices,
        metadata_records=metadata_records,
        chunk_records=chunk_records,
        history_records=history_records,
        elapsed_seconds=time.time() - started,
        extra={
            "n_inducing": int(inducing_np.shape[0]),
            "inducing_selection": settings.inducing_selection,
            "learn_inducing_locations": bool(settings.learn_inducing_locations),
            "inducing_points_file": "standard_svgp_inducing_points.npy",
        },
    )


def _validate_inputs(expression, coords, prediction_coords, feature_names):
    y = _as_dense_float(expression).astype(np.float32, copy=False)
    coords = np.asarray(coords, dtype=np.float32)
    prediction = coords if prediction_coords is None else np.asarray(prediction_coords, dtype=np.float32)
    names = np.asarray(feature_names).astype(str)
    if y.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("expression must be N x P and coords must be N x 2")
    if y.shape[0] != coords.shape[0] or y.shape[1] != names.size:
        raise ValueError("expression, coordinates, and feature names do not align")
    if prediction.ndim != 2 or prediction.shape[1] != coords.shape[1]:
        raise ValueError("prediction_coords do not match training coordinate dimensions")
    if not np.isfinite(y).all() or not np.isfinite(coords).all() or not np.isfinite(prediction).all():
        raise ValueError("GP inputs must be finite")
    return y, coords, prediction, names


def _select_inducing_points(coords, settings, rng):
    n_inducing = min(int(settings.n_inducing), coords.shape[0])
    if settings.inducing_selection == "random":
        return coords[np.sort(rng.choice(coords.shape[0], n_inducing, replace=False))].copy()
    model = MiniBatchKMeans(
        n_clusters=n_inducing,
        random_state=settings.random_state,
        batch_size=min(max(4 * n_inducing, 1024), coords.shape[0]),
        n_init=3,
        max_iter=100,
    )
    return model.fit(coords).cluster_centers_.astype(np.float32)


def _predict_exact(torch, gpytorch, model, likelihood, coords, batch_size):
    model.eval()
    likelihood.eval()
    means, latent_vars, predictive_vars = [], [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for start in range(0, coords.shape[0], batch_size):
            latent = model(coords[start : start + batch_size])
            predictive = likelihood(latent)
            means.append(latent.mean.transpose(0, 1).cpu().numpy())
            latent_vars.append(latent.variance.transpose(0, 1).cpu().numpy())
            predictive_vars.append(predictive.variance.transpose(0, 1).cpu().numpy())
    return tuple(np.concatenate(parts, axis=0) for parts in (means, latent_vars, predictive_vars))


def _predict_variational(torch, gpytorch, model, likelihood, coords, batch_size):
    return _predict_exact(torch, gpytorch, model, likelihood, coords, batch_size)


def _write_chunk(
    output_dir,
    backend,
    chunk_id,
    start,
    stop,
    feature_names,
    means,
    stds,
    arrays,
    model,
    likelihood,
    n_training_spots,
    settings,
):
    posterior_mean, posterior_variance, predictive_variance = arrays
    filename = f"vski_gp_maps_chunk_{chunk_id:04d}.npz"
    positions = np.arange(start, stop, dtype=np.int64)
    from needlepoint.artifacts import atomic_savez_compressed
    exported = encode_gp_maps(settings, dict(posterior_mean=posterior_mean,
        posterior_variance=posterior_variance, posterior_predictive_variance=predictive_variance))
    from needlepoint.posterior_blocks import atomic_savez_maps
    atomic_savez_maps(output_dir / filename, **exported,
        gene=feature_names[start:stop], feature_position=positions)

    metadata = pd.DataFrame(
        {
            "gene": feature_names[start:stop],
            "feature_position": positions,
            "standardization_mean": means[start:stop],
            "standardization_std": stds[start:stop],
            "learned_length_scale": model.covar_module.base_kernel.lengthscale.detach()
            .cpu()
            .reshape(-1)
            .numpy(),
            "learned_outputscale": model.covar_module.outputscale.detach()
            .cpu()
            .reshape(-1)
            .numpy(),
            "learned_noise_variance": likelihood.noise.detach().cpu().reshape(-1).numpy(),
            "map_chunk_id": chunk_id,
            "map_chunk_file": filename,
            "map_column": np.arange(stop - start, dtype=np.int64),
            "backend": backend,
        }
    )
    chunk = {
        "chunk_id": chunk_id,
        "file": filename,
        "n_spots": int(posterior_mean.shape[0]),
        "n_training_spots": int(n_training_spots),
        "n_features": int(stop - start),
        "feature_start": int(start),
        "feature_stop": int(stop),
    }
    return metadata, chunk


def _finish_outputs(
    output_dir,
    *,
    method,
    backend,
    settings,
    coords,
    prediction_coords,
    prediction_spot_indices,
    metadata_records,
    chunk_records,
    history_records,
    elapsed_seconds,
    extra,
):
    pd.concat(metadata_records, ignore_index=True).to_csv(
        output_dir / "vski_gp_model_metadata.csv", index=False
    )
    pd.DataFrame(chunk_records).to_csv(output_dir / "vski_gp_map_chunks.csv", index=False)
    pd.DataFrame(history_records).to_csv(output_dir / "vski_training_history.csv", index=False)
    if prediction_spot_indices is None:
        prediction_spot_indices = np.arange(prediction_coords.shape[0], dtype=np.int64)
    prediction_spot_indices = np.asarray(prediction_spot_indices)
    if prediction_spot_indices.shape != (prediction_coords.shape[0],):
        raise ValueError("prediction_spot_indices must align with prediction coordinates")
    pd.DataFrame(
        {
            "map_row": np.arange(prediction_coords.shape[0]),
            "spot_index": prediction_spot_indices,
            "x": prediction_coords[:, 0],
            "y": prediction_coords[:, 1],
        }
    ).to_csv(output_dir / "vski_gp_map_spots.csv.gz", index=False)
    manifest = {
        "method": method,
        "backend": backend,
        "n_training_spots": int(coords.shape[0]),
        "n_prediction_spots": int(prediction_coords.shape[0]),
        "n_features": int(sum(row["n_features"] for row in chunk_records)),
        "settings": asdict(settings),
        "storage": storage_manifest(settings),
        "output_dtype": settings.output_dtype,
        "scale": "standardized normalized/transformed intensity",
        "map_definition": "latent posterior mean/variance and Gaussian predictive variance",
        "metadata_file": "vski_gp_model_metadata.csv",
        "chunk_index_file": "vski_gp_map_chunks.csv",
        "training_history_file": "vski_training_history.csv",
        "spot_file": "vski_gp_map_spots.csv.gz",
        "chunks": chunk_records,
        "elapsed_seconds": float(elapsed_seconds),
    }
    manifest.update(extra)
    (output_dir / "vski_gp_maps_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
