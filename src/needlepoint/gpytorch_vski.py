from __future__ import annotations

from needlepoint.storage import GPStorageSettings, encode_gp_maps, storage_manifest

import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse

from needlepoint.svg import _as_dense_float
from needlepoint.artifacts import atomic_savez_compressed, atomic_write_json, sha256_file


VariationalDistributionName = Literal["meanfield", "cholesky", "lowrank_diag"]
KLBackendName = Literal["generic", "structured_kron", "structured_lowrank"]


@dataclass
class VSKIFitSettings(GPStorageSettings):
    grid_size: int = 64
    n_epochs: int = 5
    batch_size_spots: int = 8192
    peak_chunk_size: int = 8
    eval_batch_size_spots: int = 16384
    learning_rate: float = 0.01
    variational_learning_rate: float | None = None
    hyperparameter_learning_rate: float | None = None
    variational_distribution: VariationalDistributionName = "meanfield"
    dtype: str = "float32"
    device: str | None = None
    random_state: int = 0
    initial_length_scale: float = 160.0
    initial_spatial_fraction: float = 0.5
    kl_backend: KLBackendName = "generic"
    prior_jitter: float = 1.0e-3
    covariance_rank: int = 0
    covariance_initial_fraction: float = 0.01
    covariance_init_seed: int = 0


class _BatchedGridSVGP:
    """Factory wrapper to keep torch/gpytorch imports lazy."""

    @staticmethod
    def build(
        gpytorch,
        torch,
        n_features: int,
        grid_size: int,
        grid_bounds,
        variational_distribution: str,
        kl_backend: str = "generic",
        prior_jitter: float = 1.0e-3,
        min_length_scale: float | None = None,
        grid_eigensolver: str = "torch",
        reuse_identical_axes: bool = False,
        covariance_rank: int = 0,
        covariance_initial_fraction: float = 0.01,
        covariance_init_seed: int = 0,
    ):
        batch_shape = torch.Size([int(n_features)])
        n_inducing = int(grid_size) ** len(grid_bounds)
        if variational_distribution == "lowrank_diag":
            from needlepoint.gp_covariance import LowRankDiagonalVariationalDistribution
            variational_dist = LowRankDiagonalVariationalDistribution(
                n_inducing, rank=covariance_rank, batch_shape=batch_shape,
                initial_fraction=covariance_initial_fraction, init_seed=covariance_init_seed)
            if kl_backend != "structured_lowrank":
                raise ValueError("lowrank_diag requires structured_lowrank KL")
        elif variational_distribution == "cholesky":
            if n_inducing > 1024:
                raise ValueError("dense reference covariance is limited to 1024 inducing values")
            variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
                n_inducing,
                batch_shape=batch_shape,
            )
        elif variational_distribution == "meanfield":
            variational_dist = gpytorch.variational.MeanFieldVariationalDistribution(
                n_inducing,
                batch_shape=batch_shape,
            )
        else:
            raise ValueError(f"Unsupported variational distribution: {variational_distribution}")

        if kl_backend not in {"generic", "structured_kron", "structured_lowrank"}:
            raise ValueError(f"Unsupported KL backend: {kl_backend}")
        if kl_backend == "structured_kron" and variational_distribution != "meanfield":
            raise ValueError("structured_kron KL requires a meanfield variational distribution")
        if kl_backend == "structured_kron" and len(grid_bounds) != 2:
            raise ValueError("structured_kron KL currently supports two-dimensional grids only")
        if kl_backend == "structured_lowrank" and (
            variational_distribution != "lowrank_diag" or len(grid_bounds) != 2
        ):
            raise ValueError("structured_lowrank requires a 2D lowrank_diag distribution")
        if grid_eigensolver not in {"torch", "jacobi-experimental"}:
            raise ValueError(f"Unsupported grid eigensolver: {grid_eigensolver}")
        if grid_eigensolver != "torch" and kl_backend != "structured_kron":
            raise ValueError("Experimental grid eigensolver requires structured_kron KL")

        class StructuredGridInterpolationVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy
        ):
            """Grid strategy with an exact Kronecker mean-field KL.

            GPyTorch's generic Gaussian KL can materialize a dense ``M x M``
            right-hand side even when q(u) is diagonal. For a 128 x 128 grid,
            that intermediate is prohibitively large. The RBF prior on a
            Cartesian 2-D grid is separable, so its eigensystem is built from
            two ``grid_size x grid_size`` factors instead.
            """

            def kl_divergence(self):
                if kl_backend == "structured_lowrank":
                    from needlepoint.gp_grid_linear_algebra import structured_gaussian_grid_kl
                    q = self._variational_distribution
                    return structured_gaussian_grid_kl(
                        variational_mean=q.variational_mean, diagonal=q.diagonal_variance,
                        root=q.covariance_root, grid=self.grid,
                        prior_mean=self.model.mean_module.constant,
                        lengthscale=self.model.covar_module.base_kernel.lengthscale,
                        outputscale=self.model.covar_module.outputscale, jitter=prior_jitter)
                return _structured_meanfield_grid_kl(
                    torch=torch,
                    variational_mean=self._variational_distribution.variational_mean,
                    variational_variance=self._variational_distribution.variational_stddev.square(),
                    grid=self.grid,
                    prior_mean=self.model.mean_module.constant,
                    lengthscale=self.model.covar_module.base_kernel.lengthscale,
                    outputscale=self.model.covar_module.outputscale,
                    jitter=float(prior_jitter),
                    eigensolver=grid_eigensolver,
                    reuse_identical_axes=reuse_identical_axes,
                )

            def forward(self, x, inducing_points, inducing_values,
                        variational_inducing_covar=None, **kwargs):
                if kl_backend != "structured_lowrank":
                    return super().forward(x, inducing_points, inducing_values,
                        variational_inducing_covar=variational_inducing_covar, **kwargs)
                from needlepoint.gp_covariance import ExactDiagonalInterpolationOperator
                from linear_operator.operators import DiagLinearOperator, LowRankRootLinearOperator, SumLinearOperator
                from linear_operator.utils.interpolation import left_interp
                q = self._variational_distribution
                indices, weights = self._compute_grid(x)
                mean = left_interp(indices, weights, inducing_values.unsqueeze(-1)).squeeze(-1)
                covariance = ExactDiagonalInterpolationOperator(
                    DiagLinearOperator(q.diagonal_variance), indices, weights, indices, weights)
                if q.rank:
                    # __add__(LowRankRootLinearOperator) requests a generic
                    # root update, which can factorize the query covariance.
                    # An explicit sum preserves exact lazy diagonal contraction.
                    covariance = SumLinearOperator(covariance, LowRankRootLinearOperator(
                        left_interp(indices, weights, q.covariance_root)))
                return gpytorch.distributions.MultivariateNormal(mean, covariance)

        class BatchedGridSVGP(gpytorch.models.ApproximateGP):
            def __init__(self):
                strategy_class = (
                    StructuredGridInterpolationVariationalStrategy
                    if kl_backend in {"structured_kron", "structured_lowrank"}
                    else gpytorch.variational.GridInterpolationVariationalStrategy
                )
                strategy = strategy_class(
                    self,
                    grid_size=int(grid_size),
                    grid_bounds=grid_bounds,
                    variational_distribution=variational_dist,
                )
                super().__init__(strategy)
                self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)
                lengthscale_constraint = (
                    gpytorch.constraints.GreaterThan(float(min_length_scale))
                    if min_length_scale is not None
                    else None
                )
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        batch_shape=batch_shape,
                        lengthscale_constraint=lengthscale_constraint,
                    ),
                    batch_shape=batch_shape,
                )

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)

        return BatchedGridSVGP()


def _structured_meanfield_grid_kl(
    *,
    torch,
    variational_mean,
    variational_variance,
    grid,
    prior_mean,
    lengthscale,
    outputscale,
    jitter: float,
    eigensolver: str = "torch",
    reuse_identical_axes: bool = False,
):
    """KL[q(u)||p(u)] without constructing the full grid covariance.

    For a two-dimensional Cartesian grid and an RBF kernel,

    ``Kuu = outputscale * (Ky kron Kx) + jitter * I``.

    The eigendecompositions of ``Kx`` and ``Ky`` make the log determinant,
    quadratic form, and diagonal-covariance trace term exact while requiring
    only ``O(batch * grid_size**3)`` work and ``O(batch * grid_size**2)``
    memory. This is the key operation that permits 128 x 128 inducing grids.
    """
    if grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError("structured mean-field KL requires a two-dimensional grid")
    n_features, n_inducing = variational_mean.shape
    grid_size = int(grid.shape[0])
    if n_inducing != grid_size**2:
        raise ValueError("variational distribution does not match the Cartesian grid")

    lengthscale = lengthscale.reshape(n_features, -1)
    isotropic = lengthscale.shape[1] == 1
    if lengthscale.shape[1] == 1:
        lengthscale = lengthscale.expand(n_features, 2)
    elif lengthscale.shape[1] != 2:
        raise ValueError("structured mean-field KL expects isotropic or two-dimensional length scales")
    outputscale = outputscale.reshape(n_features)
    prior_mean = prior_mean.reshape(n_features)

    # RBF factors at long length scales are nearly rank deficient and have many
    # repeated eigenvalues. CUDA's float32 symmetric eigensolver can fail on
    # these matrices even though the full prior is positive definite after its
    # jitter is added. Float64 is materially more robust. The small factors
    # avoid a dense inducing covariance, but CUDA solver dispatch still matters
    # at large feature batches; the batched-Jacobi route is explicit opt-in.
    output_dtype = variational_mean.dtype
    work_dtype = torch.float64 if output_dtype == torch.float32 else output_dtype
    grid = grid.to(dtype=work_dtype)
    lengthscale = lengthscale.to(dtype=work_dtype)
    outputscale = outputscale.to(dtype=work_dtype)
    prior_mean = prior_mean.to(dtype=work_dtype)
    variational_mean_work = variational_mean.to(dtype=work_dtype)
    variational_variance_work = variational_variance.to(dtype=work_dtype)

    squared_distances = []
    for dimension in range(2):
        values = grid[:, dimension]
        squared_distances.append((values[:, None] - values[None, :]).square())
    kx = torch.exp(
        -0.5 * squared_distances[0][None, :, :] / lengthscale[:, 0, None, None].square()
    )
    ky = torch.exp(
        -0.5 * squared_distances[1][None, :, :] / lengthscale[:, 1, None, None].square()
    )
    factor_jitter = 1.0e-10
    identity = torch.eye(grid_size, dtype=work_dtype, device=grid.device)
    kx = 0.5 * (kx + kx.transpose(-1, -2)) + factor_jitter * identity
    ky = 0.5 * (ky + ky.transpose(-1, -2)) + factor_jitter * identity
    if eigensolver == "torch":
        factor_eigh = torch.linalg.eigh
    elif eigensolver == "jacobi-experimental":
        from needlepoint.experimental_batched_eigh import jacobi_eigh
        factor_eigh = jacobi_eigh
    else:
        raise ValueError(f"Unsupported grid eigensolver: {eigensolver}")
    eigenvalues_x, eigenvectors_x = factor_eigh(kx)
    # Same-step reuse only: independently parameterized ARD scales must retain
    # separate derivatives even when their current numerical values coincide.
    # Never retain an eigensystem across optimizer updates.
    if reuse_identical_axes and isotropic and torch.equal(squared_distances[0], squared_distances[1]):
        eigenvalues_y, eigenvectors_y = eigenvalues_x, eigenvectors_x
    else:
        eigenvalues_y, eigenvectors_y = factor_eigh(ky)
    # Tiny negative eigenvalues can occur from finite precision; the full prior
    # remains positive definite because its actual jitter is added below.
    eigenvalues_x = eigenvalues_x.clamp_min(0.0)
    eigenvalues_y = eigenvalues_y.clamp_min(0.0)
    full_eigenvalues = (
        outputscale[:, None, None]
        * eigenvalues_y[:, :, None]
        * eigenvalues_x[:, None, :]
        + float(jitter)
    ).clamp_min(torch.finfo(work_dtype).tiny)

    centered_mean = (
        variational_mean_work - prior_mean[:, None]
    ).reshape(n_features, grid_size, grid_size)
    projected_mean = torch.matmul(
        torch.matmul(eigenvectors_y.transpose(-1, -2), centered_mean),
        eigenvectors_x,
    )
    mean_quadratic = (projected_mean.square() / full_eigenvalues).sum(dim=(-2, -1))

    inverse_eigenvalues = full_eigenvalues.reciprocal()
    prior_inverse_diagonal = torch.matmul(
        torch.matmul(eigenvectors_y.square(), inverse_eigenvalues),
        eigenvectors_x.square().transpose(-1, -2),
    )
    q_variance_grid = variational_variance_work.reshape(n_features, grid_size, grid_size)
    trace_term = (q_variance_grid * prior_inverse_diagonal).sum(dim=(-2, -1))
    logdet_prior = full_eigenvalues.log().sum(dim=(-2, -1))
    logdet_q = variational_variance_work.clamp_min(
        torch.finfo(work_dtype).tiny
    ).log().sum(dim=-1)
    result = 0.5 * (
        logdet_prior - logdet_q - float(n_inducing) + trace_term + mean_quadratic
    )
    return result.to(dtype=output_dtype)


def fit_vski_gp_spatial_maps(
    expression: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    feature_names: np.ndarray,
    output_dir: str | Path,
    score_table: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
    settings: VSKIFitSettings | None = None,
    standardization_mean: np.ndarray | None = None,
    standardization_std: np.ndarray | None = None,
    prediction_coords: np.ndarray | None = None,
    prediction_spot_indices: np.ndarray | None = None,
) -> dict:
    """Fit batched grid-inducing variational GPs and save posterior maps.

    This is a true variational fitting path: kernel hyperparameters, Gaussian
    noise, and variational parameters are optimized with a minibatch ELBO. The
    inducing locations are fixed on a grid through GPyTorch's
    GridInterpolationVariationalStrategy, which is the V-SKI/KISS-GP structure.

    Outputs are in standardized normalized/transformed intensity scale.
    """
    torch, gpytorch = _require_gpytorch()
    settings = settings or VSKIFitSettings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_raw = _as_dense_float(expression).astype(np.float32, copy=False)
    if (standardization_mean is None) != (standardization_std is None):
        raise ValueError("standardization_mean and standardization_std must be supplied together")
    if standardization_mean is None:
        y_np, feature_means, feature_stds = _standardize_columns_with_stats(y_raw)
    else:
        feature_means = np.asarray(standardization_mean, dtype=np.float32)
        feature_stds = np.asarray(standardization_std, dtype=np.float32)
        if feature_means.shape != (y_raw.shape[1],) or feature_stds.shape != (y_raw.shape[1],):
            raise ValueError("Provided standardization statistics must have one value per feature")
        feature_stds = feature_stds.copy()
        feature_stds[feature_stds < 1.0e-8] = 1.0
        y_np = ((y_raw - feature_means[None, :]) / feature_stds[None, :]).astype(np.float32, copy=False)
    coords_np = np.asarray(coords, dtype=np.float32)
    prediction_coords_np = (
        coords_np if prediction_coords is None else np.asarray(prediction_coords, dtype=np.float32)
    )
    feature_names = np.asarray(feature_names).astype(str)
    if y_np.ndim != 2:
        raise ValueError("expression must have shape n_spots x n_features")
    if coords_np.shape[0] != y_np.shape[0]:
        raise ValueError("coords must have one row per spot")
    if coords_np.ndim != 2 or prediction_coords_np.ndim != 2 or coords_np.shape[1] != prediction_coords_np.shape[1]:
        raise ValueError("coords and prediction_coords must have matching n_spots x n_dimensions shapes")
    if feature_names.size != y_np.shape[1]:
        raise ValueError("feature_names must have one name per expression column")
    if settings.variational_distribution == "cholesky" and settings.grid_size > 32:
        raise ValueError("cholesky variational distribution is only allowed for --grid-size <= 32")
    if settings.kl_backend == "structured_kron" and settings.variational_distribution != "meanfield":
        raise ValueError("structured_kron KL requires --variational-distribution meanfield")
    if settings.prior_jitter <= 0:
        raise ValueError("prior_jitter must be positive")

    torch_dtype = torch.float64 if settings.dtype == "float64" else torch.float32
    torch_device = _choose_device(torch, settings.device)
    torch.manual_seed(int(settings.random_state))
    if torch_device.type == "cuda":
        torch.cuda.manual_seed_all(int(settings.random_state))
    coords_t = torch.as_tensor(coords_np, dtype=torch_dtype, device=torch_device)
    prediction_coords_t = torch.as_tensor(prediction_coords_np, dtype=torch_dtype, device=torch_device)
    y_cpu = torch.as_tensor(y_np.T, dtype=torch_dtype, device="cpu")
    n_spots, n_features = y_np.shape
    grid_bounds = _coordinate_grid_bounds(np.vstack([coords_np, prediction_coords_np]))
    np_output_dtype = np.float16 if settings.output_dtype == "float16" else np.float32

    base_table = _build_feature_table(
        feature_names=feature_names,
        feature_means=feature_means,
        feature_stds=feature_stds,
        score_table=score_table,
        metadata=metadata,
        default_length_scale=settings.initial_length_scale,
        default_spatial_fraction=settings.initial_spatial_fraction,
    )

    chunk_records: list[dict] = []
    metadata_records: list[pd.DataFrame] = []
    history_records: list[dict] = []
    hyperparameter_history_records: list[dict] = []
    inducing_grid_axes: list[list[float]] | None = None
    hyperparameter_history_path = output_dir / "vski_hyperparameter_history.csv"
    if hyperparameter_history_path.exists():
        hyperparameter_history_path.unlink()
    started = time.time()

    for chunk_id, start in enumerate(range(0, n_features, int(settings.peak_chunk_size))):
        stop = min(start + int(settings.peak_chunk_size), n_features)
        feature_positions = np.arange(start, stop, dtype=np.int64)
        chunk_table = base_table.iloc[start:stop].copy().reset_index(drop=True)
        n_chunk_features = len(chunk_table)

        model = _BatchedGridSVGP.build(
            gpytorch,
            torch,
            n_features=n_chunk_features,
            grid_size=int(settings.grid_size),
            grid_bounds=grid_bounds,
            variational_distribution=settings.variational_distribution,
            kl_backend=settings.kl_backend,
            prior_jitter=float(settings.prior_jitter),
            covariance_rank=settings.covariance_rank,
            covariance_initial_fraction=settings.covariance_initial_fraction,
            covariance_init_seed=settings.covariance_init_seed,
        ).to(device=torch_device, dtype=torch_dtype)
        if inducing_grid_axes is None:
            grid = model.variational_strategy.grid.detach().cpu().numpy()
            inducing_grid_axes = [
                grid[:, dimension].astype(float).tolist()
                for dimension in range(grid.shape[1])
            ]
        likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([n_chunk_features])).to(
            device=torch_device,
            dtype=torch_dtype,
        )
        _initialize_vski_hyperparameters(torch, model, likelihood, chunk_table, torch_device, torch_dtype)

        model.train()
        likelihood.train()
        optimizer = _build_vski_optimizer(torch, model, likelihood, settings)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_spots)
        rng = np.random.default_rng(int(settings.random_state) + chunk_id)

        for epoch in range(int(settings.n_epochs)):
            permutation = rng.permutation(n_spots)
            epoch_loss = 0.0
            n_batches = 0
            for batch_start in range(0, n_spots, int(settings.batch_size_spots)):
                batch_indices_np = permutation[batch_start : batch_start + int(settings.batch_size_spots)]
                batch_indices = torch.as_tensor(batch_indices_np, dtype=torch.long, device=torch_device)
                x_batch = coords_t.index_select(0, batch_indices)
                y_batch = y_cpu[feature_positions][:, batch_indices_np].to(device=torch_device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                output = model(x_batch)
                loss_per_feature = -mll(output, y_batch)
                loss = loss_per_feature.sum()
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.detach().cpu().item())
                n_batches += 1

            history_records.append(
                {
                    "chunk_id": int(chunk_id),
                    "epoch": int(epoch + 1),
                    "n_features": int(n_chunk_features),
                    "mean_negative_elbo": epoch_loss / max(n_batches, 1),
                }
            )
            epoch_hyperparameter_records = _collect_hyperparameter_history(
                model=model,
                likelihood=likelihood,
                chunk_table=chunk_table,
                chunk_id=chunk_id,
                epoch=epoch + 1,
                mean_negative_elbo=epoch_loss / max(n_batches, 1),
            )
            hyperparameter_history_records.extend(epoch_hyperparameter_records)
            pd.DataFrame(epoch_hyperparameter_records).to_csv(
                hyperparameter_history_path,
                mode="a",
                header=not hyperparameter_history_path.exists(),
                index=False,
            )
            print(
                f"V-SKI chunk {chunk_id} features {start}:{stop} "
                f"epoch {epoch + 1}/{int(settings.n_epochs)} "
                f"mean_negative_elbo={epoch_loss / max(n_batches, 1):.6g}",
                flush=True,
            )

        chunk_name = f"vski_gp_maps_chunk_{chunk_id:04d}.npz"
        means, variances = _evaluate_vski_posterior(
            torch=torch,
            gpytorch=gpytorch,
            model=model,
            coords_t=prediction_coords_t,
            n_spots=prediction_coords_np.shape[0],
            eval_batch_size=int(settings.eval_batch_size_spots),
            output_dtype=np.float32,
        )
        chunk_path = output_dir / chunk_name
        arrays = encode_gp_maps(settings, dict(
            posterior_mean=means, posterior_variance=variances,
            posterior_predictive_variance=variances + likelihood.noise.detach().cpu().numpy().astype(np.float32).reshape(1, -1),
        ))
        from needlepoint.posterior_blocks import atomic_savez_maps
        atomic_savez_maps(chunk_path, **arrays,
            gene=chunk_table["gene"].astype(str).to_numpy(),
            feature_position=chunk_table["feature_position"].to_numpy(dtype=np.int64))


        chunk_table["map_chunk_id"] = int(chunk_id)
        chunk_table["map_chunk_file"] = chunk_name
        chunk_table["map_column"] = np.arange(n_chunk_features, dtype=np.int64)
        _append_learned_hyperparameters(torch, model, likelihood, chunk_table)
        chunk_records.append(
            {
                "chunk_id": int(chunk_id),
                "file": chunk_name,
                "n_spots": int(prediction_coords_np.shape[0]),
                "n_training_spots": int(n_spots),
                "n_features": int(n_chunk_features),
                "feature_start": int(start),
                "feature_stop": int(stop),
                "file_size_bytes": int(chunk_path.stat().st_size),
                "file_sha256": sha256_file(chunk_path),
            }
        )
        metadata_records.append(chunk_table)
        print(f"V-SKI wrote {chunk_name} for features {start}:{stop}", flush=True)

    fitted_metadata = pd.concat(metadata_records, ignore_index=True)
    history = pd.DataFrame(history_records)
    hyperparameter_history = pd.DataFrame(hyperparameter_history_records)
    chunks = pd.DataFrame(chunk_records)
    fitted_metadata.to_csv(output_dir / "vski_gp_model_metadata.csv", index=False)
    history.to_csv(output_dir / "vski_training_history.csv", index=False)
    hyperparameter_history.to_csv(hyperparameter_history_path, index=False)
    chunks.to_csv(output_dir / "vski_gp_map_chunks.csv", index=False)
    if prediction_spot_indices is None:
        prediction_spot_indices = np.arange(prediction_coords_np.shape[0], dtype=np.int64)
    prediction_spot_indices = np.asarray(prediction_spot_indices)
    if prediction_spot_indices.shape != (prediction_coords_np.shape[0],):
        raise ValueError("prediction_spot_indices must have one value per prediction coordinate")
    pd.DataFrame(
        {
            "map_row": np.arange(prediction_coords_np.shape[0], dtype=np.int64),
            "spot_index": prediction_spot_indices,
            "x": prediction_coords_np[:, 0],
            "y": prediction_coords_np[:, 1],
        }
    ).to_csv(
        output_dir / "vski_gp_map_spots.csv.gz",
        index=False,
    )

    manifest = {
        "method": "V-SKI Gaussian variational GP",
        "n_spots": int(n_spots),
        "n_training_spots": int(n_spots),
        "n_prediction_spots": int(prediction_coords_np.shape[0]),
        "n_prediction_only_spots": int(prediction_coords_np.shape[0] - n_spots),
        "training_coverage_fraction": float(n_spots / prediction_coords_np.shape[0]),
        "full_training_coverage": bool(n_spots == prediction_coords_np.shape[0]),
        "n_features": int(n_features),
        "grid_size": int(settings.grid_size),
        "n_inducing_grid_points": int(settings.grid_size) ** int(coords_np.shape[1]),
        "grid_bounds": [(float(lo), float(hi)) for lo, hi in grid_bounds],
        "inducing_grid_axes": inducing_grid_axes,
        "n_epochs": int(settings.n_epochs),
        "batch_size_spots": int(settings.batch_size_spots),
        "peak_chunk_size": int(settings.peak_chunk_size),
        "eval_batch_size_spots": int(settings.eval_batch_size_spots),
        "learning_rate": float(settings.learning_rate),
        "variational_learning_rate": float(
            settings.variational_learning_rate or settings.learning_rate
        ),
        "hyperparameter_learning_rate": float(
            settings.hyperparameter_learning_rate or settings.learning_rate
        ),
        "variational_distribution": settings.variational_distribution,
        "kl_backend": settings.kl_backend,
        "prior_jitter": float(settings.prior_jitter),
        "likelihood": "GaussianLikelihood",
        "dtype": settings.dtype,
        "device": str(torch_device),
        "output_dtype": settings.output_dtype,
        "storage": storage_manifest(settings),
        "scale": "standardized normalized/transformed intensity",
        "map_definition": (
            "posterior latent GP mean, latent variance, and Gaussian predictive variance "
            "at each requested prediction coordinate"
        ),
        "metadata_file": "vski_gp_model_metadata.csv",
        "training_history_file": "vski_training_history.csv",
        "hyperparameter_history_file": "vski_hyperparameter_history.csv",
        "spot_file": "vski_gp_map_spots.csv.gz",
        "chunk_index_file": "vski_gp_map_chunks.csv",
        "chunks": chunk_records,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output_dir / "vski_gp_maps_manifest.json", manifest)
    return manifest


def _require_gpytorch():
    try:
        import torch
        import gpytorch
    except ImportError as exc:
        raise ImportError("V-SKI fitting requires torch and gpytorch.") from exc
    return torch, gpytorch


def _choose_device(torch, device: str | None):
    if device is not None:
        chosen = torch.device(device)
        if chosen.type == "cuda":
            torch.empty(1, device=chosen)
        return chosen
    if torch.cuda.is_available():
        chosen = torch.device("cuda")
        torch.empty(1, device=chosen)
        return chosen
    return torch.device("cpu")


def _build_vski_optimizer(torch, model, likelihood, settings: VSKIFitSettings):
    variational_parameters = list(
        model.variational_strategy._variational_distribution.parameters()
    )
    variational_ids = {id(parameter) for parameter in variational_parameters}
    hyperparameters = [
        parameter for parameter in model.parameters() if id(parameter) not in variational_ids
    ] + list(likelihood.parameters())
    return torch.optim.Adam(
        [
            {
                "params": variational_parameters,
                "lr": float(settings.variational_learning_rate or settings.learning_rate),
            },
            {
                "params": hyperparameters,
                "lr": float(settings.hyperparameter_learning_rate or settings.learning_rate),
            },
        ]
    )


def _standardize_columns_with_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    stds[stds < 1.0e-8] = 1.0
    standardized = ((matrix - means[None, :]) / stds[None, :]).astype(np.float32, copy=False)
    return standardized, means.astype(np.float32), stds.astype(np.float32)


def _coordinate_grid_bounds(coords: np.ndarray) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for dim in range(coords.shape[1]):
        lo = float(np.min(coords[:, dim]))
        hi = float(np.max(coords[:, dim]))
        if hi <= lo:
            hi = lo + 1.0
        bounds.append((lo, hi))
    return bounds


def _build_feature_table(
    feature_names: np.ndarray,
    feature_means: np.ndarray,
    feature_stds: np.ndarray,
    score_table: pd.DataFrame | None,
    metadata: pd.DataFrame | None,
    default_length_scale: float,
    default_spatial_fraction: float,
) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "gene": feature_names.astype(str),
            "feature_position": np.arange(feature_names.size, dtype=np.int64),
            "standardization_mean": feature_means,
            "standardization_std": feature_stds,
        }
    )
    if score_table is not None:
        cols = [
            c
            for c in [
                "gene",
                "rank",
                "peak_index",
                "peak_mz",
                "best_length_scale",
                "spatial_fraction",
                "LLR",
                "qvalue",
                "peak_pattern_group",
                "spatial_interest_score",
                "artifact_like_score",
                "relative_prevalence",
                "peak_total_intensity",
            ]
            if c in score_table.columns
        ]
        table = table.merge(score_table[cols].drop_duplicates("gene"), on="gene", how="left")
    if metadata is not None and "gpytorch_feature_id" in metadata.columns:
        meta_cols = [
            c
            for c in [
                "gpytorch_feature_id",
                "peak_index",
                "peak_mz",
                "relative_prevalence",
                "peak_total_intensity",
            ]
            if c in metadata.columns
        ]
        table = table.merge(
            metadata[meta_cols].drop_duplicates("gpytorch_feature_id"),
            left_on="gene",
            right_on="gpytorch_feature_id",
            how="left",
            suffixes=("", "_metadata"),
        )
    if "best_length_scale" not in table.columns:
        table["best_length_scale"] = np.nan
    if "spatial_fraction" not in table.columns:
        table["spatial_fraction"] = np.nan
    table["initial_length_scale"] = table["best_length_scale"].fillna(float(default_length_scale)).astype(float)
    rho = table["spatial_fraction"].fillna(float(default_spatial_fraction)).astype(float)
    table["initial_spatial_fraction"] = rho.clip(1.0e-4, 0.95)
    return table


def _initialize_vski_hyperparameters(torch, model, likelihood, chunk_table: pd.DataFrame, device, dtype) -> None:
    lengthscale = torch.as_tensor(
        chunk_table["initial_length_scale"].to_numpy(dtype=np.float32),
        device=device,
        dtype=dtype,
    ).view(-1, 1, 1)
    rho = torch.as_tensor(
        chunk_table["initial_spatial_fraction"].to_numpy(dtype=np.float32),
        device=device,
        dtype=dtype,
    ).clamp(1.0e-4, 0.95)
    model.covar_module.base_kernel.lengthscale = lengthscale
    model.covar_module.outputscale = rho
    likelihood.noise = (1.0 - rho).clamp_min(1.0e-4)


def _evaluate_vski_posterior(
    torch,
    gpytorch,
    model,
    coords_t,
    n_spots: int,
    eval_batch_size: int,
    output_dtype,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    means = None
    variances = None
    batch_size = min(int(eval_batch_size), int(n_spots))
    if batch_size <= 0:
        raise ValueError("eval_batch_size and n_spots must be positive")
    start = 0
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        while start < n_spots:
            stop = min(start + batch_size, n_spots)
            posterior = None
            mean_np = None
            try:
                posterior = model(coords_t[start:stop])
                mean_np = (
                    posterior.mean.transpose(0, 1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(output_dtype, copy=False)
                )
                var_np = (
                    posterior.variance.transpose(0, 1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(output_dtype, copy=False)
                )
            except torch.OutOfMemoryError:
                if not coords_t.is_cuda or batch_size == 1:
                    raise
                posterior = None
                mean_np = None
                torch.cuda.empty_cache()
                reduced_batch_size = max(1, batch_size // 2)
                warnings.warn(
                    "CUDA OOM during V-SKI posterior evaluation; reducing "
                    f"eval batch size from {batch_size} to {reduced_batch_size}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                batch_size = reduced_batch_size
                continue
            if means is None:
                n_features = mean_np.shape[1]
                means = np.empty((n_spots, n_features), dtype=output_dtype)
                variances = np.empty((n_spots, n_features), dtype=output_dtype)
            means[start:stop] = mean_np
            variances[start:stop] = var_np
            start = stop
    if means is None or variances is None:
        raise RuntimeError("No posterior batches were evaluated")
    return means, variances


def _append_learned_hyperparameters(torch, model, likelihood, chunk_table: pd.DataFrame) -> None:
    chunk_table["learned_length_scale"] = (
        model.covar_module.base_kernel.lengthscale.detach().cpu().reshape(-1).numpy().astype(np.float64)
    )
    chunk_table["learned_outputscale"] = (
        model.covar_module.outputscale.detach().cpu().reshape(-1).numpy().astype(np.float64)
    )
    chunk_table["learned_noise_variance"] = likelihood.noise.detach().cpu().reshape(-1).numpy().astype(np.float64)


def _collect_hyperparameter_history(
    *,
    model,
    likelihood,
    chunk_table: pd.DataFrame,
    chunk_id: int,
    epoch: int,
    mean_negative_elbo: float,
) -> list[dict]:
    """Record per-feature kernel and likelihood parameters after each epoch.

    The chunk-level ELBO alone can improve while length scales remain nearly
    fixed. This history makes that separation explicit for convergence checks.
    """
    length_scales = model.covar_module.base_kernel.lengthscale.detach().cpu().reshape(-1).numpy().astype(np.float64)
    outputscales = model.covar_module.outputscale.detach().cpu().reshape(-1).numpy().astype(np.float64)
    noise_variances = likelihood.noise.detach().cpu().reshape(-1).numpy().astype(np.float64)
    constant_means = model.mean_module.constant.detach().cpu().reshape(-1).numpy().astype(np.float64)

    records: list[dict] = []
    for map_column, row in chunk_table.reset_index(drop=True).iterrows():
        initial_length_scale = float(row["initial_length_scale"])
        learned_length_scale = float(length_scales[map_column])
        record = {
            "chunk_id": int(chunk_id),
            "epoch": int(epoch),
            "map_column": int(map_column),
            "feature_position": int(row["feature_position"]),
            "gene": str(row["gene"]),
            "initial_length_scale": initial_length_scale,
            "current_length_scale": learned_length_scale,
            "length_scale_delta": learned_length_scale - initial_length_scale,
            "length_scale_relative_delta": (learned_length_scale - initial_length_scale) / initial_length_scale
            if initial_length_scale != 0
            else np.nan,
            "current_outputscale": float(outputscales[map_column]),
            "current_noise_variance": float(noise_variances[map_column]),
            "current_constant_mean": float(constant_means[map_column]),
            "mean_negative_elbo": float(mean_negative_elbo),
        }
        for optional_col in ["peak_index", "peak_mz", "rank", "spatial_fraction", "spatial_interest_score"]:
            if optional_col in row.index and pd.notna(row[optional_col]):
                value = row[optional_col]
                if optional_col in {"peak_index", "rank"}:
                    value = int(value)
                elif optional_col in {"peak_mz", "spatial_fraction", "spatial_interest_score"}:
                    value = float(value)
                record[optional_col] = value
        records.append(record)
    return records
