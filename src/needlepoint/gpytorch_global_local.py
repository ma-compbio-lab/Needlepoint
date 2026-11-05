from __future__ import annotations

from needlepoint.storage import GPStorageSettings, encode_gp_maps, storage_manifest

import fcntl
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

from needlepoint.gpytorch_vski import (
    VSKIFitSettings,
    _choose_device,
    _require_gpytorch,
    _standardize_columns_with_stats,
    fit_vski_gp_spatial_maps,
)
from needlepoint.svg import _as_dense_float
from needlepoint.artifacts import atomic_savez_compressed, atomic_write_json, sha256_file


@dataclass
class GlobalLocalGPSettings(GPStorageSettings):
    """Settings for a structured global GP plus local VNNGP residual model."""

    global_grid_size: int = 128
    global_epochs: int = 5
    local_epochs: int = 5
    batch_size_spots: int = 4096
    peak_chunk_size: int = 4
    eval_batch_size_spots: int = 8192
    global_learning_rate: float = 0.01
    local_learning_rate: float = 0.01
    global_hyperparameter_learning_rate: float = 0.05
    local_hyperparameter_learning_rate: float = 0.02
    # Defaults are in the same physical coordinate units as the input spots.
    global_initial_length_scale: float = 160.0
    local_initial_length_scale: float = 20.0
    local_min_length_scale: float = 5.0
    local_max_length_scale: float = 40.0
    local_neighbors: int = 16
    predecessor_candidate_multiplier: int = 8
    neighbor_query_batch_size: int = 50_000
    geometry_cache_dir: str | None = None
    geometry_cache_mode: str = "off"
    dtype: str = "float32"
    device: str | None = None
    random_state: int = 0
    keep_component_outputs: bool = False


def morton_order(coords: np.ndarray, bits: int = 16) -> np.ndarray:
    """Return a deterministic space-filling-curve order for 2-D coordinates."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape n_spots x 2")
    lo = coords.min(axis=0)
    span = np.maximum(coords.max(axis=0) - lo, 1.0e-12)
    quantized = np.rint((coords - lo) / span * ((1 << bits) - 1)).astype(np.uint32)

    def spread(value: np.ndarray) -> np.ndarray:
        value = value.astype(np.uint64)
        value = (value | (value << 8)) & np.uint64(0x00FF00FF)
        value = (value | (value << 4)) & np.uint64(0x0F0F0F0F)
        value = (value | (value << 2)) & np.uint64(0x33333333)
        value = (value | (value << 1)) & np.uint64(0x55555555)
        return value

    code = spread(quantized[:, 0]) | (spread(quantized[:, 1]) << np.uint64(1))
    return np.argsort(code, kind="stable")


def build_predecessor_neighbor_dag(
    ordered_coords: np.ndarray,
    n_neighbors: int,
    *,
    candidate_multiplier: int = 8,
    query_batch_size: int = 50_000,
) -> np.ndarray:
    """Find local predecessors without GPyTorch's O(N) repeated tree fitting.

    The returned row ``j`` contains neighbors for point ``j + n_neighbors``.
    A single cKDTree is queried in bounded-memory batches. Rare early points
    without enough predecessors in the candidate set use an exact fallback.
    """
    coords = np.asarray(ordered_coords, dtype=np.float32)
    n_spots = coords.shape[0]
    k = int(n_neighbors)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("ordered_coords must have shape n_spots x 2")
    if not 0 < k < n_spots:
        raise ValueError("n_neighbors must be positive and smaller than n_spots")
    query_k = min(n_spots, max(k + 1, int(candidate_multiplier) * k + 1))
    tree = cKDTree(coords)
    result = np.empty((n_spots - k, k), dtype=np.int64)

    for start in range(k, n_spots, int(query_batch_size)):
        stop = min(start + int(query_batch_size), n_spots)
        _, candidates = tree.query(coords[start:stop], k=query_k, workers=-1)
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.ndim == 1:
            candidates = candidates[:, None]
        for offset, point_index in enumerate(range(start, stop)):
            predecessors = candidates[offset][candidates[offset] < point_index]
            if predecessors.size < k:
                distances = np.square(coords[:point_index] - coords[point_index]).sum(axis=1)
                predecessors = np.argpartition(distances, kth=k - 1)[:k]
                predecessors = predecessors[np.argsort(distances[predecessors], kind="stable")]
            result[point_index - k] = predecessors[:k]
    return result


def load_or_build_vnngp_geometry(
    coords: np.ndarray,
    n_neighbors: int,
    *,
    candidate_multiplier: int = 8,
    query_batch_size: int = 50_000,
    cache_dir: str | Path | None = None,
    cache_mode: str = "off",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return one reusable Morton order and predecessor DAG.

    The cache key includes the exact canonical float32 coordinates and every
    neighbor-graph parameter that can affect the DAG.  A file lock prevents
    concurrent feature shards from redundantly constructing the same graph;
    arrays and the completion manifest are written atomically, with the
    manifest written last.
    """

    if cache_mode not in {"off", "read", "readwrite"}:
        raise ValueError("geometry_cache_mode must be off, read, or readwrite")
    coords = np.ascontiguousarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape n_spots x 2")
    if not np.isfinite(coords).all():
        raise ValueError("coords must be finite")
    n_spots = int(coords.shape[0])
    k = int(n_neighbors)
    if not 0 < k < n_spots:
        raise ValueError("n_neighbors must be positive and smaller than n_spots")
    if int(candidate_multiplier) <= 0 or int(query_batch_size) <= 0:
        raise ValueError("candidate_multiplier and query_batch_size must be positive")

    signature = {
        "schema_version": 1,
        "coordinate_sha256": _geometry_array_sha256(coords),
        "coordinate_shape": [n_spots, 2],
        "coordinate_dtype": str(coords.dtype),
        "morton_bits": 16,
        "n_neighbors": k,
        "candidate_multiplier": int(candidate_multiplier),
        "query_batch_size": int(query_batch_size),
    }
    signature_sha256 = hashlib.sha256(
        json.dumps(signature, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    def build() -> tuple[np.ndarray, np.ndarray]:
        order = morton_order(coords)
        ordered = coords[order]
        predecessors = build_predecessor_neighbor_dag(
            ordered,
            k,
            candidate_multiplier=int(candidate_multiplier),
            query_batch_size=int(query_batch_size),
        )
        return order.astype(np.int64, copy=False), predecessors.astype(np.int64, copy=False)

    if cache_mode == "off":
        order, predecessors = build()
        return order, predecessors, {
            **signature,
            "signature_sha256": signature_sha256,
            "mode": "off",
            "cache_hit": False,
            "cache_root": None,
        }
    if cache_dir is None:
        raise ValueError("geometry_cache_dir is required when geometry_cache_mode is enabled")

    cache_root = Path(cache_dir) / f"vnngp_geometry_{signature_sha256[:24]}"
    if cache_mode == "read" and not cache_root.is_dir():
        raise FileNotFoundError(f"VNNGP geometry cache is missing: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / "cache.lock"
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        cached = _load_vnngp_geometry_cache(cache_root, signature, signature_sha256)
        if cached is not None:
            order, predecessors, manifest = cached
            return order, predecessors, {
                **manifest,
                "mode": cache_mode,
                "cache_hit": True,
                "cache_root": str(cache_root),
            }
        if cache_mode == "read":
            raise FileNotFoundError(
                f"VNNGP geometry cache has no complete compatible manifest: {cache_root}"
            )

        order, predecessors = build()
        order_path = cache_root / "morton_order.npy"
        predecessor_path = cache_root / "predecessor_indices.npy"
        _atomic_save_npy(order_path, order)
        _atomic_save_npy(predecessor_path, predecessors)
        manifest = {
            **signature,
            "signature_sha256": signature_sha256,
            "order_file": order_path.name,
            "order_shape": list(order.shape),
            "order_dtype": str(order.dtype),
            "order_sha256": _geometry_array_sha256(order),
            "predecessor_file": predecessor_path.name,
            "predecessor_shape": list(predecessors.shape),
            "predecessor_dtype": str(predecessors.dtype),
            "predecessor_sha256": _geometry_array_sha256(predecessors),
        }
        atomic_write_json(cache_root / "geometry_manifest.json", manifest)
        return order, predecessors, {
            **manifest,
            "mode": cache_mode,
            "cache_hit": False,
            "cache_root": str(cache_root),
        }


def _load_vnngp_geometry_cache(
    cache_root: Path,
    signature: dict,
    signature_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    manifest_path = cache_root / "geometry_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("signature_sha256") != signature_sha256:
        raise ValueError(f"VNNGP geometry cache signature mismatch: {manifest_path}")
    for key, value in signature.items():
        if manifest.get(key) != value:
            raise ValueError(f"VNNGP geometry cache differs for {key}: {manifest_path}")
    order_path = cache_root / str(manifest.get("order_file", ""))
    predecessor_path = cache_root / str(manifest.get("predecessor_file", ""))
    if not order_path.is_file() or not predecessor_path.is_file():
        raise ValueError(f"VNNGP geometry cache manifest references missing arrays: {cache_root}")
    order = np.load(order_path, mmap_mode="c", allow_pickle=False)
    predecessors = np.load(predecessor_path, mmap_mode="c", allow_pickle=False)
    expected_order_shape = (int(signature["coordinate_shape"][0]),)
    expected_predecessor_shape = (
        int(signature["coordinate_shape"][0]) - int(signature["n_neighbors"]),
        int(signature["n_neighbors"]),
    )
    if order.shape != expected_order_shape or order.dtype != np.dtype(np.int64):
        raise ValueError(f"Invalid cached Morton-order schema: {order_path}")
    if predecessors.shape != expected_predecessor_shape or predecessors.dtype != np.dtype(np.int64):
        raise ValueError(f"Invalid cached predecessor schema: {predecessor_path}")
    if predecessors.size and (
        int(predecessors.min()) < 0
        or int(predecessors.max()) >= int(signature["coordinate_shape"][0])
    ):
        raise ValueError(f"Cached predecessor indices are out of bounds: {predecessor_path}")
    return order, predecessors, manifest


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _geometry_array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _gather_vnngp_neighbors(values, neighbor_indices):
    """Gather (..., batch, k) values without an (..., all_spots, k) view.

    A gather backward allocates a gradient with its input's shape. Expanding
    the input over k before gathering therefore creates an avoidable O(G*N*k)
    backward temporary, even though expand itself allocates no forward data.
    Flatten only the requested indices and gather along the original N axis.
    Repeated indices still accumulate gradients into the same source entries.
    """
    import torch

    if values.ndim < 1 or neighbor_indices.ndim < 2:
        raise ValueError("Require values (..., N) and indices (..., batch, k)")
    batch_shape = torch.broadcast_shapes(values.shape[:-1], neighbor_indices.shape[:-2])
    output_shape = batch_shape + neighbor_indices.shape[-2:]
    indices = neighbor_indices.expand(output_shape).flatten(start_dim=-2)
    source = values.expand(batch_shape + values.shape[-1:])
    return source.gather(-1, indices).reshape(output_shape)


class _BatchedCachedVNNGP:
    """Lazy GPyTorch factory for a batched VNNGP with a precomputed DAG."""

    @staticmethod
    def build(
        gpytorch,
        torch,
        coords_t,
        predecessor_indices_t,
        n_features: int,
        n_neighbors: int,
        training_batch_size: int,
        min_length_scale: float,
        max_length_scale: float,
    ):
        batch_shape = torch.Size([int(n_features)])
        variational_dist = gpytorch.variational.MeanFieldVariationalDistribution(
            coords_t.shape[0],
            batch_shape=batch_shape,
        )

        class CachedNNVariationalStrategy(gpytorch.variational.NNVariationalStrategy):
            """NN strategy with a cached DAG and explicit joint-fit primitives.

            All use of GPyTorch's private VNNGP KL helpers is confined here so
            version-sensitive behavior can be tested in one place.
            """

            def __init__(self, *args, cached_indices, **kwargs):
                object.__setattr__(self, "_cached_predecessor_indices", cached_indices)
                super().__init__(*args, **kwargs)

            def _compute_nn(self):
                self.nn_xinduce_idx = self._cached_predecessor_indices.to(
                    device=self.inducing_points.device,
                    dtype=torch.long,
                )
                return self

            def training_marginals(self, observation_indices):
                """Gather q(u_i) marginals by exact training index."""
                indices = observation_indices.to(
                    device=self.inducing_points.device,
                    dtype=torch.long,
                )
                mean = self._variational_distribution.variational_mean.index_select(
                    -1, indices
                )
                variance = self._variational_distribution._variational_stddev.index_select(
                    -1, indices
                ).square()
                return mean, variance

            def _stochastic_kl_helper(self, kl_indices):
                """GPyTorch conditional KL with memory-bounded neighbour gathers.

                Preserve the upstream conditional Gaussian formula, jitter,
                covariance solves and reduction order. Only variational-neighbour
                indexing differs from NNVariationalStrategy's implementation.
                Test values and all parameter gradients against the installed
                upstream helper, including shared/repeated neighbour indices.
                """
                kl_bs = len(kl_indices)
                variational_mean = self._variational_distribution.variational_mean
                variational_stddev = self._variational_distribution._variational_stddev
                logdet_q = (variational_stddev[..., kl_indices] ** 2).log().sum(dim=-1)

                inducing_points = self.inducing_points[..., kl_indices, :].expand(
                    *self._batch_shape, kl_bs, self.D
                )
                neighbor_indices = self.nn_xinduce_idx[..., kl_indices - self.k, :].to(
                    inducing_points.device
                )
                # Coordinates are fixed (no gradient); retain the upstream
                # coordinate broadcasting and covariance evaluation exactly.
                expanded_points = self.inducing_points.unsqueeze(-2).expand(
                    *self._batch_shape, self.M, self.k, self.D
                )
                expanded_indices = neighbor_indices.unsqueeze(-1).expand(
                    *self._batch_shape, kl_bs, self.k, self.D
                )
                neighbors = expanded_points.gather(-3, expanded_indices)
                neighbors_ = neighbors.permute(
                    (-3,) + tuple(range(len(self._batch_shape))) + (-2, -1)
                )
                inducing_points_ = inducing_points.permute(
                    (-2,) + tuple(range(len(self._batch_shape))) + (-1,)
                )
                full_output = self.model.forward(
                    torch.cat([neighbors_, inducing_points_.unsqueeze(-2)], dim=-2)
                )
                full_mean, full_covar = full_output.mean, full_output.covariance_matrix
                undo_permute = tuple(range(1, 1 + len(self._batch_shape))) + (0, -1)
                neighbor_prior_mean = full_mean[..., :self.k].permute(undo_permute)
                inducing_prior_mean = full_mean[..., self.k:].permute(undo_permute).squeeze(-1)
                neighbor_prior_cov = full_covar[..., :self.k, :self.k]
                neighbor_cross_cov = full_covar[..., :self.k, self.k:]
                interp_term = torch.linalg.solve(
                    neighbor_prior_cov
                    + self.jitter_val * torch.eye(self.k, device=self.inducing_points.device),
                    neighbor_cross_cov,
                ).squeeze(-1).permute(undo_permute)
                neighbor_cross_cov = neighbor_cross_cov.squeeze(-1).permute(undo_permute)
                invquad_for_conditional = (interp_term * neighbor_cross_cov).sum(dim=-1)
                inducing_prior_cov = self.model.covar_module.forward(
                    inducing_points, inducing_points, diag=True
                )
                conditional_variance = inducing_prior_cov - invquad_for_conditional
                conditional_variance = conditional_variance + self.jitter_val
                logdet_p = conditional_variance.log().sum(dim=-1)

                neighbor_variance = _gather_vnngp_neighbors(
                    variational_stddev, neighbor_indices
                ) ** 2
                weighted_neighbor_variance = (interp_term**2 * neighbor_variance).sum(dim=-1)
                inducing_variance = variational_stddev[..., kl_indices] ** 2
                trace_term = (
                    (1.0 / conditional_variance)
                    * (weighted_neighbor_variance + inducing_variance)
                ).sum(dim=-1)
                neighbor_mean = _gather_vnngp_neighbors(variational_mean, neighbor_indices)
                conditional_mean = (
                    interp_term * (neighbor_mean - neighbor_prior_mean)
                ).sum(dim=-1)
                invquad_term = (
                    (variational_mean[..., kl_indices] - inducing_prior_mean - conditional_mean) ** 2
                    / conditional_variance
                ).sum(dim=-1)
                kl = (logdet_p - logdet_q - kl_bs + trace_term + invquad_term) * 0.5
                if kl.shape != self._batch_shape:
                    raise RuntimeError(f"Unexpected conditional KL shape: {kl.shape}")
                return kl

            def local_kl(
                self,
                *,
                mode: str,
                batch_size: int,
                generator=None,
                conditional_indices=None,
            ):
                """Return the full KL or an unbiased conditional-KL estimate.

                ``conditional_indices`` are zero-based indices into the
                ``M-k`` conditional terms. Joint training supplies one slice
                of an epoch-level permutation per observation minibatch. The
                fallback samples with replacement in O(batch_size) time.
                """
                if mode not in {"stochastic", "full"}:
                    raise ValueError(f"Unsupported local KL mode: {mode}")
                batch_size = int(batch_size)
                if batch_size <= 0:
                    raise ValueError("local KL batch size must be positive")

                first_k = self._firstk_kl_helper()
                n_conditionals = int(self.M - self.k)
                if mode == "full" or batch_size >= n_conditionals:
                    result = first_k
                    all_indices = torch.arange(
                        self.k,
                        self.M,
                        device=self.inducing_points.device,
                    )
                    for indices in all_indices.split(batch_size):
                        result = result + self._stochastic_kl_helper(indices)
                    return result

                if conditional_indices is None:
                    relative_indices = torch.randint(
                        n_conditionals,
                        (min(batch_size, n_conditionals),),
                        device=self.inducing_points.device,
                        generator=generator,
                    )
                else:
                    relative_indices = conditional_indices.to(
                        device=self.inducing_points.device,
                        dtype=torch.long,
                    ).reshape(-1)
                    if relative_indices.numel() == 0:
                        raise ValueError("conditional_indices must not be empty")
                    if (
                        int(relative_indices.min()) < 0
                        or int(relative_indices.max()) >= n_conditionals
                    ):
                        raise ValueError("conditional_indices are out of range")

                conditional_kl = None
                for relative_batch in relative_indices.split(batch_size):
                    term = self._stochastic_kl_helper(relative_batch + self.k)
                    conditional_kl = term if conditional_kl is None else conditional_kl + term
                return first_k + (
                    float(n_conditionals) / float(relative_indices.numel())
                ) * conditional_kl

        class BatchedVNNGP(gpytorch.models.ApproximateGP):
            def __init__(self):
                strategy = CachedNNVariationalStrategy(
                    self,
                    coords_t,
                    variational_dist,
                    k=int(n_neighbors),
                    training_batch_size=int(training_batch_size),
                    compute_full_kl=False,
                    cached_indices=predecessor_indices_t,
                )
                super().__init__(strategy)
                self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(
                        nu=1.5,
                        batch_shape=batch_shape,
                        lengthscale_constraint=gpytorch.constraints.Interval(
                            float(min_length_scale),
                            float(max_length_scale),
                        ),
                    ),
                    batch_shape=batch_shape,
                )

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean_module(x),
                    self.covar_module(x),
                )

        return BatchedVNNGP()


def fit_vnngp_spatial_maps(
    expression: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    feature_names: np.ndarray,
    output_dir: str | Path,
    *,
    settings: GlobalLocalGPSettings,
    prediction_coords: np.ndarray | None = None,
    training_prediction_rows: np.ndarray | None = None,
    standardize: bool = True,
) -> dict:
    """Fit local VNNGP maps, optionally predicting held-out coordinates."""
    torch, gpytorch = _require_gpytorch()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_raw = _as_dense_float(expression).astype(np.float32, copy=False)
    if standardize:
        y_np, feature_means, feature_stds = _standardize_columns_with_stats(y_raw)
    else:
        y_np = y_raw
        feature_means = np.zeros(y_np.shape[1], dtype=np.float32)
        feature_stds = np.ones(y_np.shape[1], dtype=np.float32)
    coords_np = np.asarray(coords, dtype=np.float32)
    prediction_coords_np = coords_np if prediction_coords is None else np.asarray(prediction_coords, dtype=np.float32)
    feature_names = np.asarray(feature_names).astype(str)
    if y_np.shape[0] != coords_np.shape[0] or y_np.shape[1] != feature_names.size:
        raise ValueError("expression, coords, and feature_names dimensions do not match")
    if prediction_coords_np.ndim != 2 or prediction_coords_np.shape[1] != 2:
        raise ValueError("prediction_coords must have shape n_spots x 2")
    if training_prediction_rows is None:
        if prediction_coords_np.shape[0] != coords_np.shape[0]:
            raise ValueError("training_prediction_rows is required when prediction coordinates differ")
        training_prediction_rows = np.arange(coords_np.shape[0], dtype=np.int64)
    training_prediction_rows = np.asarray(training_prediction_rows, dtype=np.int64)
    if training_prediction_rows.shape != (coords_np.shape[0],):
        raise ValueError("training_prediction_rows must map every training row to a prediction row")

    dag_started = time.time()
    order, predecessor_indices, geometry_cache_audit = load_or_build_vnngp_geometry(
        coords_np,
        settings.local_neighbors,
        candidate_multiplier=settings.predecessor_candidate_multiplier,
        query_batch_size=settings.neighbor_query_batch_size,
        cache_dir=settings.geometry_cache_dir,
        cache_mode=settings.geometry_cache_mode,
    )
    ordered_coords = coords_np[order]
    ordered_y = y_np[order]
    ordered_prediction_rows = training_prediction_rows[order]
    dag_seconds = time.time() - dag_started

    torch_dtype = torch.float64 if settings.dtype == "float64" else torch.float32
    device = _choose_device(torch, settings.device)
    torch.manual_seed(int(settings.random_state))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(settings.random_state))
    coords_t = torch.as_tensor(ordered_coords, dtype=torch_dtype, device=device)
    predecessor_indices_t = torch.as_tensor(predecessor_indices, dtype=torch.long, device=device)
    y_cpu = torch.as_tensor(ordered_y.T, dtype=torch_dtype, device="cpu")
    prediction_coords_t = torch.as_tensor(prediction_coords_np, dtype=torch_dtype, device=device)
    output_dtype = np.float16 if settings.output_dtype == "float16" else np.float32
    n_train, n_features = y_np.shape
    n_prediction = prediction_coords_np.shape[0]
    history_records: list[dict] = []
    metadata_records: list[dict] = []
    chunk_records: list[dict] = []
    started = time.time()

    for chunk_id, start in enumerate(range(0, n_features, int(settings.peak_chunk_size))):
        stop = min(start + int(settings.peak_chunk_size), n_features)
        positions = np.arange(start, stop, dtype=np.int64)
        n_chunk = positions.size
        model = _BatchedCachedVNNGP.build(
            gpytorch,
            torch,
            coords_t,
            predecessor_indices_t,
            n_features=n_chunk,
            n_neighbors=settings.local_neighbors,
            training_batch_size=settings.batch_size_spots,
            min_length_scale=settings.local_min_length_scale,
            max_length_scale=settings.local_max_length_scale,
        ).to(device=device, dtype=torch_dtype)
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            batch_shape=torch.Size([n_chunk])
        ).to(device=device, dtype=torch_dtype)
        model.covar_module.base_kernel.lengthscale = float(settings.local_initial_length_scale)
        model.covar_module.outputscale = 0.25
        likelihood.noise = 0.25
        variational_parameters = list(
            model.variational_strategy._variational_distribution.parameters()
        )
        variational_ids = {id(parameter) for parameter in variational_parameters}
        hyperparameters = [
            parameter for parameter in model.parameters() if id(parameter) not in variational_ids
        ] + list(likelihood.parameters())
        optimizer = torch.optim.Adam(
            [
                {"params": variational_parameters, "lr": float(settings.local_learning_rate)},
                {
                    "params": hyperparameters,
                    "lr": float(settings.local_hyperparameter_learning_rate),
                },
            ]
        )
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=n_train)
        model.train()
        likelihood.train()

        for epoch in range(int(settings.local_epochs)):
            model.variational_strategy._set_training_iterator()
            epoch_loss = 0.0
            for _ in range(model.variational_strategy._total_training_batches):
                optimizer.zero_grad(set_to_none=True)
                output = model(None)
                indices = model.variational_strategy.current_training_indices
                target = y_cpu[positions][:, indices.cpu()].to(device=device, non_blocking=True)
                loss = (-mll(output, target)).sum()
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
            mean_loss = epoch_loss / max(model.variational_strategy._total_training_batches, 1)
            history_records.append(
                {
                    "component": "local_vnngp",
                    "chunk_id": chunk_id,
                    "epoch": epoch + 1,
                    "mean_negative_elbo": mean_loss,
                }
            )
            print(
                f"VNNGP chunk {chunk_id} features {start}:{stop} "
                f"epoch {epoch + 1}/{settings.local_epochs} mean_negative_elbo={mean_loss:.6g}",
                flush=True,
            )

        q_distribution = model.variational_strategy._variational_distribution()
        q_mean = q_distribution.mean.transpose(0, 1).detach().cpu().numpy()
        q_variance = q_distribution.variance.transpose(0, 1).detach().cpu().numpy()
        means = np.full((n_prediction, n_chunk), np.nan, dtype=np.float32)
        variances = np.full((n_prediction, n_chunk), np.nan, dtype=np.float32)
        means[ordered_prediction_rows] = q_mean
        variances[ordered_prediction_rows] = q_variance

        holdout_rows = np.flatnonzero(~np.isfinite(means[:, 0]))
        if holdout_rows.size:
            model.variational_strategy.nn_util.set_nn_idx(coords_t.float())
            model.eval()
            with torch.no_grad():
                for batch_start in range(0, holdout_rows.size, int(settings.eval_batch_size_spots)):
                    rows = holdout_rows[batch_start : batch_start + int(settings.eval_batch_size_spots)]
                    posterior = model(prediction_coords_t[rows])
                    means[rows] = posterior.mean.transpose(0, 1).detach().cpu().numpy()
                    variances[rows] = posterior.variance.transpose(0, 1).detach().cpu().numpy()
        if not np.isfinite(means).all() or not np.isfinite(variances).all():
            raise RuntimeError("VNNGP prediction produced non-finite maps")

        noise = likelihood.noise.detach().cpu().numpy().astype(np.float32).reshape(-1)
        chunk_name = f"vnngp_gp_maps_chunk_{chunk_id:04d}.npz"
        chunk_path = output_dir / chunk_name
        arrays = encode_gp_maps(settings, dict(
            posterior_mean=means, posterior_variance=variances,
            posterior_predictive_variance=variances + noise[None, :]))
        from needlepoint.posterior_blocks import atomic_savez_maps
        atomic_savez_maps(chunk_path, **arrays,
            gene=feature_names[positions], feature_position=positions)

        for local_column, feature_position in enumerate(positions):
            metadata_records.append(
                {
                    "gene": feature_names[feature_position],
                    "feature_position": int(feature_position),
                    "standardization_mean": float(feature_means[feature_position]),
                    "standardization_std": float(feature_stds[feature_position]),
                    "learned_length_scale": float(
                        model.covar_module.base_kernel.lengthscale.detach().cpu().reshape(-1)[local_column]
                    ),
                    "learned_outputscale": float(
                        model.covar_module.outputscale.detach().cpu().reshape(-1)[local_column]
                    ),
                    "learned_noise_variance": float(noise[local_column]),
                    "map_chunk_id": chunk_id,
                    "map_chunk_file": chunk_name,
                    "map_column": local_column,
                }
            )
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "file": chunk_name,
                "n_spots": n_prediction,
                "n_training_spots": n_train,
                "n_features": n_chunk,
                "feature_start": start,
                "feature_stop": stop,
                "file_size_bytes": int(chunk_path.stat().st_size),
                "file_sha256": sha256_file(chunk_path),
            }
        )

    pd.DataFrame(history_records).to_csv(output_dir / "vnngp_training_history.csv", index=False)
    pd.DataFrame(metadata_records).to_csv(output_dir / "vnngp_model_metadata.csv", index=False)
    pd.DataFrame(chunk_records).to_csv(output_dir / "vnngp_gp_map_chunks.csv", index=False)
    used_for_training = np.zeros(n_prediction, dtype=bool)
    used_for_training[np.asarray(training_prediction_rows, dtype=np.int64)] = True
    pd.DataFrame(
        {
            "map_row": np.arange(n_prediction, dtype=np.int64),
            "x": prediction_coords_np[:, 0],
            "y": prediction_coords_np[:, 1],
            "used_for_training": used_for_training,
            "prediction_only": ~used_for_training,
        }
    ).to_csv(output_dir / "vnngp_gp_map_spots.csv.gz", index=False)
    manifest = {
        "method": "local VNNGP with cached Morton predecessor DAG",
        "n_training_spots": n_train,
        "n_prediction_spots": n_prediction,
        "n_prediction_only_spots": int(n_prediction - n_train),
        "training_coverage_fraction": float(n_train / n_prediction),
        "full_training_coverage": bool(n_train == n_prediction),
        "n_features": n_features,
        "settings": asdict(settings),
        "storage": storage_manifest(settings),
        "output_dtype": settings.output_dtype,
        "neighbor_dag_seconds": dag_seconds,
        "vnngp_geometry_cache": geometry_cache_audit,
        "vnngp_conditional_kl_backend": "indexed-neighbor-gather-v1",
        "elapsed_seconds": time.time() - started,
        "chunks": chunk_records,
    }
    atomic_write_json(output_dir / "vnngp_gp_maps_manifest.json", manifest)
    return manifest


def fit_global_local_gp_spatial_maps(
    expression: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    feature_names: np.ndarray,
    output_dir: str | Path,
    *,
    settings: GlobalLocalGPSettings | None = None,
    prediction_coords: np.ndarray | None = None,
    prediction_spot_indices: np.ndarray | None = None,
    training_prediction_rows: np.ndarray | None = None,
) -> dict:
    """Fit global V-SKI, local residual VNNGP, and save additive maps."""
    settings = settings or GlobalLocalGPSettings()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    components_dir = output_dir / "components"
    global_dir = components_dir / "global_vski"
    local_dir = components_dir / "local_vnngp"
    y_raw = _as_dense_float(expression).astype(np.float32, copy=False)
    y_standardized, original_means, original_stds = _standardize_columns_with_stats(y_raw)
    coords_np = np.asarray(coords, dtype=np.float32)
    prediction_coords_np = coords_np if prediction_coords is None else np.asarray(prediction_coords, dtype=np.float32)
    if training_prediction_rows is None:
        training_prediction_rows = np.arange(coords_np.shape[0], dtype=np.int64)

    global_settings = VSKIFitSettings(
        grid_size=settings.global_grid_size,
        n_epochs=settings.global_epochs,
        batch_size_spots=settings.batch_size_spots,
        peak_chunk_size=settings.peak_chunk_size,
        eval_batch_size_spots=settings.eval_batch_size_spots,
        learning_rate=settings.global_learning_rate,
        variational_learning_rate=settings.global_learning_rate,
        hyperparameter_learning_rate=settings.global_hyperparameter_learning_rate,
        variational_distribution="meanfield",
        dtype=settings.dtype,
        device=settings.device,
        output_dtype="float32",
        retain_predictive_variance=True,
        random_state=settings.random_state,
        initial_length_scale=settings.global_initial_length_scale,
        kl_backend="structured_kron",
    )
    started = time.time()
    global_manifest = fit_vski_gp_spatial_maps(
        y_standardized,
        coords_np,
        feature_names,
        global_dir,
        settings=global_settings,
        standardization_mean=np.zeros(y_standardized.shape[1], dtype=np.float32),
        standardization_std=np.ones(y_standardized.shape[1], dtype=np.float32),
        prediction_coords=prediction_coords_np,
        prediction_spot_indices=prediction_spot_indices,
    )
    global_maps = _load_component_maps(global_dir, "vski_gp_map_chunks.csv")
    global_training_mean = global_maps["posterior_mean"][np.asarray(training_prediction_rows, dtype=np.int64)]
    residual = y_standardized - global_training_mean
    local_manifest = fit_vnngp_spatial_maps(
        residual,
        coords_np,
        feature_names,
        local_dir,
        settings=replace(settings, output_dtype="float32", retain_predictive_variance=True),
        prediction_coords=prediction_coords_np,
        training_prediction_rows=np.asarray(training_prediction_rows, dtype=np.int64),
        standardize=False,
    )
    local_maps = _load_component_maps(local_dir, "vnngp_gp_map_chunks.csv")

    n_prediction, n_features = global_maps["posterior_mean"].shape
    chunk_records: list[dict] = []
    for chunk_id, start in enumerate(range(0, n_features, settings.peak_chunk_size)):
        stop = min(start + settings.peak_chunk_size, n_features)
        positions = np.arange(start, stop, dtype=np.int64)
        global_mean = global_maps["posterior_mean"][:, positions]
        local_mean = local_maps["posterior_mean"][:, positions]
        global_variance = global_maps["posterior_variance"][:, positions]
        local_variance = local_maps["posterior_variance"][:, positions]
        predictive_variance = global_variance + local_maps["posterior_predictive_variance"][:, positions]
        chunk_name = f"vski_gp_maps_chunk_{chunk_id:04d}.npz"
        chunk_path = output_dir / chunk_name
        exported = encode_gp_maps(settings, dict(
            posterior_mean=global_mean + local_mean,
            posterior_variance=global_variance + local_variance,
            posterior_predictive_variance=predictive_variance,
            global_posterior_mean=global_mean, local_posterior_mean=local_mean,
            global_posterior_variance=global_variance, local_posterior_variance=local_variance))
        from needlepoint.posterior_blocks import atomic_savez_maps
        atomic_savez_maps(chunk_path, **exported,
            gene=np.asarray(feature_names).astype(str)[positions], feature_position=positions)
        chunk_records.append(
            {
                "chunk_id": chunk_id,
                "file": chunk_name,
                "n_spots": n_prediction,
                "n_training_spots": y_standardized.shape[0],
                "n_features": positions.size,
                "feature_start": start,
                "feature_stop": stop,
                "file_size_bytes": int(chunk_path.stat().st_size),
                "file_sha256": sha256_file(chunk_path),
            }
        )

    global_metadata = pd.read_csv(global_dir / "vski_gp_model_metadata.csv")
    local_metadata = pd.read_csv(local_dir / "vnngp_model_metadata.csv")
    metadata = global_metadata.merge(
        local_metadata[
            [
                "feature_position",
                "learned_length_scale",
                "learned_outputscale",
                "learned_noise_variance",
            ]
        ],
        on="feature_position",
        suffixes=("_global", "_local"),
    )
    metadata["standardization_mean"] = original_means
    metadata["standardization_std"] = original_stds
    metadata.to_csv(output_dir / "vski_gp_model_metadata.csv", index=False)
    pd.DataFrame(chunk_records).to_csv(output_dir / "vski_gp_map_chunks.csv", index=False)
    global_history = pd.read_csv(global_dir / "vski_training_history.csv")
    global_history.insert(0, "component", "global_vski")
    local_history = pd.read_csv(local_dir / "vnngp_training_history.csv")
    pd.concat([global_history, local_history], ignore_index=True, sort=False).to_csv(
        output_dir / "global_local_training_history.csv", index=False
    )
    shutil.copy2(global_dir / "vski_gp_map_spots.csv.gz", output_dir / "vski_gp_map_spots.csv.gz")
    manifest = {
        "method": "global structured V-SKI plus local VNNGP residual backfit",
        "model": "y = global_rbf_surface + local_matern32_surface + Gaussian_noise",
        "fit_order": ["global_vski", "local_vnngp_on_global_residual"],
        "n_training_spots": int(y_standardized.shape[0]),
        "n_prediction_spots": int(n_prediction),
        "n_prediction_only_spots": int(n_prediction - y_standardized.shape[0]),
        "training_coverage_fraction": float(y_standardized.shape[0] / n_prediction),
        "full_training_coverage": bool(y_standardized.shape[0] == n_prediction),
        "n_features": int(n_features),
        "grid_bounds": global_manifest.get("grid_bounds"),
        "inducing_grid_axes": global_manifest.get("inducing_grid_axes"),
        "settings": asdict(settings),
        "storage": storage_manifest(settings),
        "output_dtype": settings.output_dtype,
        "global_manifest": str(global_dir / "vski_gp_maps_manifest.json"),
        "local_manifest": str(local_dir / "vnngp_gp_maps_manifest.json"),
        "chunks": chunk_records,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output_dir / "vski_gp_maps_manifest.json", manifest)
    if not settings.keep_component_outputs:
        shutil.rmtree(components_dir)
    return manifest


def _load_component_maps(component_dir: Path, index_name: str) -> dict[str, np.ndarray]:
    index = pd.read_csv(component_dir / index_name).sort_values("chunk_id")
    keys = ("posterior_mean", "posterior_variance", "posterior_predictive_variance")
    blocks = {key: [] for key in keys}
    for row in index.itertuples(index=False):
        with np.load(component_dir / row.file) as data:
            for key in keys:
                blocks[key].append(np.asarray(data[key], dtype=np.float32))
    return {key: np.concatenate(value, axis=1) for key, value in blocks.items()}
