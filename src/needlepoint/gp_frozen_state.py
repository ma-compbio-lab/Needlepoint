"""Immutable, NumPy-only descriptions of previously fitted Gaussian priors.

Importing this module does not import Torch. Loading a schema-2 training export
uses its checkpoint and the installed interpolation/neighbor implementations,
but stores only plain arrays and scalars in the resulting snapshot. Checkpoints
must have a SHA256 in the completion receipt before they are deserialized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


CONVENTION_VERSION = "ski-vnngp-joint-frozen-v1"
QUERY_CONVENTION = "unique-query-independent-local-conditional-residual-v1"


def array_sha256(array):
    """The dtype/shape-aware hash used by the original component checkpoints."""
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _immutable_array(value, dtype=None):
    array = np.ascontiguousarray(value, dtype=dtype)
    # A bytes owner also prevents callers from re-enabling ndarray WRITEABLE.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class FrozenGaussianFit:
    """One feature's fixed SKI/VNNGP model, with no live training objects.

    All training arrays follow the saved predecessor DAG order. Query arrays
    follow the source map order. ``training_values`` are training-standardized
    observations, and may be absent for covariance-only calculations.
    ``target_identity`` excludes observations, learned q parameters and source
    filenames: changing a variational family cannot change a fixed prior target.
    """

    grid_axes: tuple[np.ndarray, np.ndarray]
    interpolation_indices: np.ndarray
    interpolation_weights: np.ndarray
    training_query_rows: np.ndarray
    training_coordinates: np.ndarray
    local_dag: np.ndarray
    query_coordinates: np.ndarray
    query_neighbor_indices: np.ndarray
    global_lengthscale: float | tuple[float, float]
    global_scale: float
    global_mean: float
    global_jitter: float
    axis_stabilization: float
    local_lengthscale: float
    local_scale: float
    local_jitter: float
    local_neighbors: int
    noise_variance: float
    feature_position: int = 0
    feature_name: str = "feature"
    standardization_mean: float = 0.0
    standardization_std: float = 1.0
    training_row_ids: np.ndarray | None = None
    query_row_ids: np.ndarray | None = None
    training_values: np.ndarray | None = None
    source_variational_mean: np.ndarray | None = None
    source_provenance: Mapping = field(default_factory=dict)
    construction_dtype: str = "float32"
    coordinate_units: str = "micrometers"
    source_commit: str = "unspecified"
    global_kernel: str = "rbf"
    local_kernel: str = "matern32"
    inducing_ordering: str = "x-fast-y-slow"
    local_initial_block_rule: str = "correlated-first-k-plus-jitter"
    query_convention: str = QUERY_CONVENTION
    schema_version: int = 1
    convention_version: str = CONVENTION_VERSION
    target_identity: str = field(init=False)

    def __post_init__(self):
        arrays = ("interpolation_indices", "interpolation_weights", "training_query_rows",
                  "training_coordinates", "local_dag", "query_coordinates",
                  "query_neighbor_indices", "training_row_ids", "query_row_ids",
                  "training_values", "source_variational_mean")
        for name in arrays:
            value = getattr(self, name)
            if value is not None:
                integer = name in {"interpolation_indices", "training_query_rows", "local_dag",
                                   "query_neighbor_indices", "training_row_ids", "query_row_ids"}
                object.__setattr__(self, name, _immutable_array(value, np.int64 if integer else None))
        object.__setattr__(self, "grid_axes", tuple(_immutable_array(a) for a in self.grid_axes))
        if not np.isscalar(self.global_lengthscale):
            object.__setattr__(self, "global_lengthscale", tuple(float(v) for v in self.global_lengthscale))
        object.__setattr__(self, "source_provenance", _freeze(self.source_provenance))
        n, p, k = len(self.training_coordinates), len(self.query_coordinates), self.local_neighbors
        if self.training_coordinates.shape != (n, 2) or self.query_coordinates.shape != (p, 2):
            raise ValueError("training and query coordinates must have shape (n, 2)")
        if not 0 < k < n or self.local_dag.shape != (n-k, k):
            raise ValueError("local DAG shape does not match training support/neighbors")
        if (self.local_dag < 0).any() or (self.local_dag >= np.arange(k, n)[:, None]).any():
            raise ValueError("local DAG must contain only earlier training indices")
        if any(len(np.unique(row)) != k for row in self.local_dag):
            raise ValueError("local predecessor rows must have unique indices")
        if self.training_query_rows.shape != (n,) or len(np.unique(self.training_query_rows)) != n:
            raise ValueError("training-to-query mapping must be unique")
        if (self.training_query_rows < 0).any() or (self.training_query_rows >= p).any():
            raise ValueError("training-to-query mapping is outside query support")
        if not np.array_equal(self.training_coordinates, self.query_coordinates[self.training_query_rows]):
            raise ValueError("training-to-query coordinates disagree")
        if len(np.unique(self.query_coordinates, axis=0)) != p:
            raise ValueError("duplicate query coordinates require explicit deduplication")
        if self.query_neighbor_indices.shape != (p, k):
            raise ValueError("query neighbor indices have incompatible dimensions")
        withheld = np.ones(p, dtype=bool)
        withheld[self.training_query_rows] = False
        nb = self.query_neighbor_indices[withheld]
        if ((nb < 0) | (nb >= n)).any():
            raise ValueError("withheld query neighbors must index training support")
        if len(self.grid_axes) != 2 or any(a.ndim != 1 or len(a) < 2 or (np.diff(a) <= 0).any()
                                               for a in self.grid_axes):
            raise ValueError("grid axes must be increasing one-dimensional arrays")
        if any(not np.isfinite(axis).all() for axis in self.grid_axes):
            raise ValueError("grid axes must be finite")
        m = len(self.grid_axes[0]) * len(self.grid_axes[1])
        if (self.interpolation_indices.shape != self.interpolation_weights.shape or
                self.interpolation_indices.ndim != 2 or self.interpolation_indices.shape[0] != p):
            raise ValueError("interpolation arrays must have matching query rows")
        if ((self.interpolation_indices < 0) | (self.interpolation_indices >= m)).any():
            raise ValueError("interpolation index outside inducing grid")
        for name in arrays:
            value = getattr(self, name)
            if value is not None and not np.isfinite(value).all():
                raise ValueError(f"nonfinite frozen field: {name}")
        for name in ("global_lengthscale", "global_scale", "local_lengthscale", "local_scale",
                     "noise_variance", "standardization_std"):
            value = np.asarray(getattr(self, name))
            if not np.isfinite(value).all() or (value <= 0).any():
                raise ValueError(f"{name} must be finite and positive")
        for name in ("global_jitter", "local_jitter", "axis_stabilization"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite([self.global_mean, self.standardization_mean]).all():
            raise ValueError("mean parameters must be finite")
        if self.training_values is not None and self.training_values.shape != (n,):
            raise ValueError("training values must follow ordered training rows")
        if self.source_variational_mean is not None and self.source_variational_mean.shape != (p,):
            raise ValueError("source mean must follow query rows")
        for name, count in (("training_row_ids", n), ("query_row_ids", p)):
            value = getattr(self, name)
            if value is not None and (value.shape != (count,) or len(np.unique(value)) != count):
                raise ValueError("original row identities must be unique")
        if (self.training_row_ids is not None and self.query_row_ids is not None and
                not np.array_equal(self.training_row_ids, self.query_row_ids[self.training_query_rows])):
            raise ValueError("training original row identities disagree with query mapping")
        if self.global_kernel != "rbf" or self.local_kernel != "matern32":
            raise ValueError("unsupported frozen kernel convention")
        if (self.inducing_ordering != "x-fast-y-slow" or
                self.local_initial_block_rule != "correlated-first-k-plus-jitter" or
                self.query_convention != QUERY_CONVENTION or
                self.convention_version != CONVENTION_VERSION or self.schema_version != 1):
            raise ValueError("unsupported frozen inference convention")
        identity = {name: getattr(self, name) for name in (
            "global_lengthscale", "global_scale", "global_mean", "global_jitter", "axis_stabilization",
            "local_lengthscale", "local_scale", "local_jitter", "local_neighbors", "noise_variance",
            "standardization_mean", "standardization_std", "construction_dtype", "coordinate_units",
            "global_kernel", "local_kernel", "inducing_ordering", "local_initial_block_rule",
            "query_convention", "schema_version", "convention_version")}
        identity["array_hashes"] = {name: array_sha256(getattr(self, name)) for name in arrays
                                    if name not in {"training_values", "source_variational_mean"}
                                    and getattr(self, name) is not None}
        identity["grid_axis_hashes"] = [array_sha256(a) for a in self.grid_axes]
        object.__setattr__(self, "target_identity", _json_hash(identity))


def _checked_file(folder, name, hashes, *, required=True):
    path = (folder / name).resolve()
    if not path.is_relative_to(folder.resolve()):
        raise ValueError("source artifact path escapes its directory")
    expected = hashes.get(name)
    if expected is None and required:
        raise ValueError(f"source completion receipt lacks hash for {name}")
    actual = _file_hash(path)
    if expected is not None and actual != expected:
        raise ValueError(f"source artifact hash mismatch: {name}")
    return path, actual


def _constraints_value(torch, state, key):
    raw = state[key]
    constraint = key + "_constraint"
    lo, hi = state[constraint + ".lower_bound"], state[constraint + ".upper_bound"]
    if torch.isfinite(hi).all():
        return raw.sigmoid() * (hi - lo) + lo
    return torch.nn.functional.softplus(raw) + lo


def _resolve_conventions(folder, manifest, checkpoint):
    explicit = manifest.get("frozen_inference_conventions")
    if explicit is not None:
        required = {"local_jitter", "nearest_neighbor_backend", "gpytorch_version",
                    "scikit_learn_version", "coordinate_units"}
        if not required <= explicit.keys():
            raise ValueError("incomplete explicit frozen inference conventions")
        return dict(explicit), {"resolution": "explicit source manifest"}
    # This registry is intentionally narrow. An absent jitter or a different
    # training implementation is never interpreted using current defaults.
    identity = checkpoint.get("pilot_run_identity", {})
    commit = identity.get("source", {}).get("commit")
    if (commit != "824a09c4f6ac11029422b6e9adb6b96ad2b21658" or
            manifest.get("vnngp_conditional_kl_backend") != "indexed-neighbor-gather-v1"):
        raise ValueError("missing verified source jitter/neighbor convention; require explicit conventions")
    env_path = folder.parents[1] / "provenance" / "execution_environment.json"
    environment = json.loads(env_path.read_text())
    packages = environment["packages"]
    if packages.get("gpytorch") != "1.15.2" or any("faiss" in k.lower() for k in packages):
        raise ValueError("unsupported source nearest-neighbor or local jitter convention")
    conventions = dict(local_jitter=0.001, nearest_neighbor_backend="sklearn-auto",
                       gpytorch_version=packages["gpytorch"],
                       scikit_learn_version=packages["scikit-learn"], coordinate_units="micrometers")
    return conventions, dict(resolution="audited schema-2 indexed-neighbor-gather-v1 / GPyTorch 1.15.2",
                             environment_sha256=_file_hash(env_path), environment_path=str(env_path))


def _verify_query_receipt(folder, spots, hashes, spot_name, identity):
    """Verify all prediction rows even for older receipts omitting the CSV."""
    if spot_name in hashes:
        return {"query_support_verification": "completion receipt spot-file SHA256"}
    if not identity.get("crop_hash") or not identity.get("crop"):
        raise ValueError("query spot table has neither a source hash nor a verified crop receipt")
    crop_path = folder.parents[1] / "provenance" / f"{identity['crop']}_crop.json"
    crop = json.loads(crop_path.read_text())
    if _json_hash(crop) != identity["crop_hash"]:
        raise ValueError("source crop receipt disagrees with checkpoint identity")
    if not np.array_equal(spots.spot_index.to_numpy(), crop["original_row_ids"]):
        raise ValueError("source query row IDs disagree with verified crop receipt")
    # Pilot v2 stores original acquired coordinates as float64 and hashes shape
    # text, unlike the package's checkpoint hash. This is a schema-specific
    # receipt adapter, not a guess at the checkpoint coordinate representation.
    coordinates = np.ascontiguousarray(spots[["x", "y"]].to_numpy(dtype=np.float64))
    digest = hashlib.sha256(str(coordinates.dtype).encode() + str(coordinates.shape).encode()
                            + coordinates.tobytes()).hexdigest()
    if digest != crop["coordinate_sha256"]:
        raise ValueError("source query coordinates disagree with verified crop receipt")
    return dict(query_support_verification="checkpoint-bound pilot-v2 crop receipt",
                source_crop_sha256=_file_hash(crop_path))


def _raw_training_values(raw_manifest, query_row_ids, training_query_rows, feature_names,
                         expected_coordinates, metadata_rows, expected_hash):
    """Replay the original float64 statistics then float32 standardization."""
    if raw_manifest is None:
        return None, None
    raw = json.loads(Path(raw_manifest).read_text()) if isinstance(raw_manifest, (str, Path)) else dict(raw_manifest)
    if raw.get("orientation") != "rows_by_features":
        raise ValueError("raw manifest must declare rows_by_features orientation")
    if raw.get("coordinate_units") != "micrometers":
        raise ValueError("raw manifest coordinate units are not micrometers")
    paths = raw["paths"]
    for key in ("values", "coords", "features", "row_ids"):
        expected = raw.get("expected_sha256", {}).get(Path(paths[key]).name)
        if expected is None:
            expected = raw.get("verified_sha256", {}).get(key)
        if expected is None or _file_hash(paths[key]) != expected:
            raise ValueError(f"raw input hash mismatch or missing: {key}")
    row_ids = np.load(paths["row_ids"], allow_pickle=False)
    features = np.load(paths["features"], allow_pickle=False).astype(str)
    if len(np.unique(row_ids)) != len(row_ids) or len(np.unique(features)) != len(features):
        raise ValueError("raw row IDs and feature names must be unique")
    row_lookup = {int(v): i for i, v in enumerate(row_ids)}
    feature_lookup = {str(v): i for i, v in enumerate(features)}
    try:
        query_source = np.array([row_lookup[int(v)] for v in query_row_ids])
        columns = np.array([feature_lookup[str(v)] for v in feature_names])
    except KeyError as error:
        raise ValueError("raw input lacks source rows or features") from error
    coordinates = np.load(paths["coords"], mmap_mode="r", allow_pickle=False)
    if not np.array_equal(np.asarray(coordinates[query_source], dtype=np.float32), expected_coordinates):
        raise ValueError("raw input query coordinates disagree with source maps")
    values = np.load(paths["values"], mmap_mode="r", allow_pickle=False)
    # The trainer first casts the raw feature chunk to float32, then computes
    # standardization statistics in float64; preserve both conversions.
    training = np.asarray(values[np.ix_(query_source[training_query_rows], columns)], dtype=np.float32).astype(np.float64)
    if not np.isfinite(training).all():
        raise ValueError("nonfinite training values cannot be replayed without a saved observation mask")
    means, stds = training.mean(axis=0), training.std(axis=0)
    stds[stds < 1e-8] = 1.0
    for column, row in enumerate(metadata_rows):
        if (np.float32(means[column]) != np.float32(row.standardization_mean) or
                np.float32(stds[column]) != np.float32(row.standardization_std)):
            raise ValueError("training-only standardization disagrees with metadata")
    standardized = ((training-means) / stds).astype(np.float32)
    if array_sha256(standardized) != expected_hash:
        raise ValueError("ordered standardized training values fail checkpoint hash")
    return standardized, {"manifest_sha256": _json_hash(raw), "transformation": raw.get("transformation"),
                          "training_value_sha256": expected_hash}


def load_frozen_gaussian_fit(source_dir, raw_input_manifest=None, feature_positions=None):
    """Load a verified schema-2 training export into immutable feature snapshots.

    Returns a tuple in source feature order. ``feature_positions`` are positions
    in the source map metadata, not positions in an external assay. With no raw
    manifest, only means are unavailable; the covariance target is identical.
    Metadata-only legacy exports currently fail closed because this exporter
    did not record the global mean and replayable local DAG in its metadata.
    """
    import pandas as pd
    import torch
    import gpytorch
    import sklearn
    from .gpytorch_global_local import build_predecessor_neighbor_dag

    folder = Path(source_dir).resolve()
    complete_path = folder / "complete.json"
    if not complete_path.is_file():
        raise ValueError("a hash-bearing source complete.json receipt is required")
    complete = json.loads(complete_path.read_text())
    hashes = complete.get("artifact_hashes", {})
    manifest_path, manifest_hash = _checked_file(folder, "vski_gp_maps_manifest.json", hashes)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("observation_model") != "gaussian":
        raise ValueError("frozen joint conditioning requires a Gaussian observation model")
    if manifest.get("model") != "y = global_rbf + local_matern32 + Gaussian_noise":
        raise ValueError("unsupported or missing source kernel-family declaration")
    metadata_path, metadata_hash = _checked_file(folder, manifest["metadata_file"], hashes)
    metadata = pd.read_csv(metadata_path)
    if metadata.feature_position.duplicated().any() or metadata.gene.duplicated().any():
        raise ValueError("source feature ordering is ambiguous")
    positions = metadata.feature_position.to_numpy(dtype=int)
    if not np.array_equal(positions, np.arange(len(positions))):
        raise ValueError("source feature positions must be contiguous and ordered")
    requested = set(positions) if feature_positions is None else set(map(int, feature_positions))
    if not requested <= set(positions) or not requested:
        raise ValueError("requested feature positions are missing or empty")
    spots_path, spots_hash = _checked_file(folder, manifest["spot_file"], hashes, required=False)
    spots = pd.read_csv(spots_path)
    if not np.array_equal(spots.map_row.to_numpy(), np.arange(len(spots))):
        raise ValueError("source query map rows must be contiguous and ordered")
    query = spots[["x", "y"]].to_numpy(dtype=np.float32)
    row_ids = spots.spot_index.to_numpy(dtype=np.int64)
    used = spots.used_for_training.astype(str).str.lower().isin(["true", "1"]).to_numpy()
    if int(used.sum()) != manifest["n_training_spots"] or len(query) != manifest["n_prediction_spots"]:
        raise ValueError("source training support disagrees with manifest")
    if len(np.unique(query, axis=0)) != len(query):
        raise ValueError("duplicate query coordinates require explicit deduplication")
    coordinate_lookup = {tuple(point): i for i, point in enumerate(query)}
    snapshots = []
    source_commit = None
    for chunk in manifest["chunks"]:
        chunk_rows = metadata[metadata.map_chunk_id == chunk["chunk_id"]]
        if not requested.intersection(chunk_rows.feature_position):
            continue
        state_file = chunk.get("state_dict_file")
        if not state_file:
            raise ValueError("legacy export lacks a checkpoint: global mean and original DAG cannot be verified")
        checkpoint_path, checkpoint_hash = _checked_file(folder, state_file, hashes)
        # Local user-supplied training state is trusted only after integrity
        # verification. RNG/optimizer fields require the legacy pickle reader.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_schema") != 2:
            raise ValueError("only schema-2 component checkpoints are supported")
        if _json_hash(checkpoint["settings"]) != _json_hash(manifest["settings"]):
            raise ValueError("checkpoint settings disagree with source manifest")
        covariance = manifest["global_covariance"]
        if (checkpoint.get("covariance_basis") != "inducing_values_unwhitened" or
                covariance.get("basis") != "inducing_values_unwhitened" or
                covariance.get("implementation_version") != "inducing-values-diag-plus-root-v1"):
            raise ValueError("unsupported source inducing-prior implementation convention")
        if covariance.get("prior_jitter") != checkpoint["settings"]["global_prior_jitter"]:
            raise ValueError("manifest global prior jitter disagrees with checkpoint settings")
        # This training implementation fixes the axis-factor stabilization in
        # code. A changed manifest value is a different prior, not a numerical
        # solver setting, and needs a separately versioned training convention.
        if covariance.get("axis_factor_stabilization") != 1e-10:
            raise ValueError("axis stabilization disagrees with audited source implementation")
        conventions, resolution = _resolve_conventions(folder, manifest, checkpoint)
        if conventions["gpytorch_version"] != gpytorch.__version__:
            raise ValueError("interpolation replay requires the recorded GPyTorch version")
        if (conventions["nearest_neighbor_backend"] != "sklearn-auto" or
                conventions["scikit_learn_version"] != sklearn.__version__):
            raise ValueError("query-neighbor replay requires the recorded scikit-learn backend/version")
        identity = checkpoint.get("pilot_run_identity", {})
        commit = identity.get("source", {}).get("commit", manifest.get("source_commit"))
        if not commit or (source_commit is not None and source_commit != commit):
            raise ValueError("missing or mixed source repository commits")
        source_commit = commit
        if ((folder / "run.json").is_file() and
                _json_hash(json.loads((folder / "run.json").read_text())) != _json_hash(identity)):
            raise ValueError("checkpoint identity disagrees with source run receipt")
        support_verification = _verify_query_receipt(folder, spots, hashes, manifest["spot_file"], identity)
        global_state, local_state = checkpoint["global_model"], checkpoint["local_model"]
        training = local_state["variational_strategy.inducing_points"].cpu().numpy().astype(np.float32)
        if array_sha256(training) != checkpoint["training_coordinate_sha256"]:
            raise ValueError("saved training coordinates fail checkpoint hash")
        if checkpoint["training_coordinate_sha256"] != manifest["training_coordinate_sha256"]:
            raise ValueError("manifest training coordinate hash disagrees with checkpoint")
        try:
            training_rows = np.array([coordinate_lookup[tuple(point)] for point in training], dtype=np.int64)
        except KeyError as error:
            raise ValueError("source training coordinate absent from prediction support") from error
        if set(training_rows) != set(np.flatnonzero(used)):
            raise ValueError("checkpoint training support disagrees with map spot table")
        settings = checkpoint["settings"]
        k = int(settings["local_neighbors"])
        dag = build_predecessor_neighbor_dag(training, k,
                  candidate_multiplier=settings["predecessor_candidate_multiplier"],
                  query_batch_size=settings["neighbor_query_batch_size"])
        if (array_sha256(dag) != checkpoint["neighbor_dag_sha256"] or
                checkpoint["neighbor_dag_sha256"] != manifest["neighbor_dag_sha256"]):
            raise ValueError("reconstructed predecessor DAG fails the original hash (including tie order)")
        grid = global_state["variational_strategy.grid"]
        axes = tuple(grid[:, axis].cpu().numpy() for axis in range(2))
        if not np.array_equal(np.asarray(manifest["inducing_grid_axes"]), np.asarray(axes)):
            raise ValueError("checkpoint inducing axes disagree with source manifest")
        x, y = np.meshgrid(*axes, indexing="xy")
        inducing = np.column_stack([x.ravel(), y.ravel()])
        if not np.array_equal(inducing, global_state["variational_strategy.inducing_points"].cpu().numpy()):
            raise ValueError("unrecognized inducing ordering; expected x-fast-y-slow")
        with torch.no_grad():
            indices, weights = gpytorch.utils.interpolation.Interpolation().interpolate(
                grid, torch.as_tensor(query.copy(), dtype=grid.dtype))
        indices, weights = indices.numpy(), weights.numpy()
        withheld = np.flatnonzero(~used)
        query_neighbors = np.full((len(query), k), -1, dtype=np.int64)
        if len(withheld):
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(training)
            query_neighbors[withheld] = neighbors.kneighbors(query[withheld], return_distance=False)
        map_path, map_hash = _checked_file(folder, chunk["file"], hashes)
        with np.load(map_path, allow_pickle=False) as maps:
            if not np.array_equal(maps["gene"].astype(str), chunk_rows.gene.to_numpy(dtype=str)):
                raise ValueError("source map feature ordering disagrees with metadata")
            source_means = maps["posterior_mean"].copy()
            global_means = maps["global_posterior_mean"].copy() if "global_posterior_mean" in maps else None
        raw_values, preprocessing = _raw_training_values(raw_input_manifest, row_ids, training_rows,
             chunk_rows.gene.to_numpy(dtype=str), query, list(chunk_rows.itertuples()),
             checkpoint["training_value_sha256"])
        if (preprocessing is not None and identity.get("input_hash") is not None and
                preprocessing["manifest_sha256"] != identity["input_hash"]):
            raise ValueError("raw input manifest identity disagrees with source checkpoint")
        lengths = _constraints_value(torch, global_state, "covar_module.base_kernel.raw_lengthscale")
        global_scales = _constraints_value(torch, global_state, "covar_module.raw_outputscale")
        local_lengths = _constraints_value(torch, local_state, "covar_module.base_kernel.raw_lengthscale")
        local_scales = _constraints_value(torch, local_state, "covar_module.raw_outputscale")
        noises = _constraints_value(torch, checkpoint["joint_likelihood"], "noise_covar.raw_noise")
        for column, row in enumerate(chunk_rows.itertuples()):
            if int(row.feature_position) not in requested:
                continue
            if int(row.map_column) != column:
                raise ValueError("metadata columns do not match checkpoint feature ordering")
            parameters = dict(global_lengthscale=float(lengths[column].item()),
                global_scale=float(global_scales[column].item()),
                global_mean=float(global_state["mean_module.raw_constant"][column].item()),
                local_lengthscale=float(local_lengths[column].item()),
                local_scale=float(local_scales[column].item()), noise_variance=float(noises[column].item()))
            metadata_names = {"global_lengthscale": "joint_global_length_scale", "global_scale": "joint_global_outputscale",
                              "local_lengthscale": "joint_local_length_scale", "local_scale": "joint_local_outputscale",
                              "noise_variance": "joint_noise_variance"}
            for name, metadata_name in metadata_names.items():
                if not np.isclose(parameters[name], float(getattr(row, metadata_name)), rtol=2e-7, atol=1e-9):
                    raise ValueError(f"checkpoint parameter disagrees with metadata: {name}")
            # The source strategy interpolates unwhitened q(u). Verify this
            # independent of q family; never add the prior constant twice.
            if global_means is not None:
                qmean = global_state["variational_strategy._variational_distribution.variational_mean"][column].numpy()
                reconstructed = (qmean[indices] * weights).sum(axis=1)
                tolerance = 2e-3 if global_means.dtype == np.float16 else 5e-6
                if not np.allclose(reconstructed, global_means[:, column], rtol=tolerance, atol=tolerance):
                    raise ValueError("replayed interpolation disagrees with saved global mean")
            covariance = manifest["global_covariance"]
            provenance = dict(source_dir=str(folder), source_commit=commit, checkpoint_schema=2,
                checkpoint_sha256=checkpoint_hash, manifest_sha256=manifest_hash, metadata_sha256=metadata_hash,
                source_map_sha256=map_hash, source_spots_sha256=spots_hash,
                completion_sha256=_file_hash(complete_path), source_family=checkpoint["covariance_family"],
                source_rank=checkpoint["covariance_rank"], source_fit_purpose=manifest.get("fit_purpose"),
                downstream_ready=manifest.get("downstream_ready", False),
                training_coordinate_sha256=checkpoint["training_coordinate_sha256"],
                neighbor_dag_sha256=checkpoint["neighbor_dag_sha256"],
                interpolation_indices_sha256=array_sha256(indices), interpolation_weights_sha256=array_sha256(weights),
                query_neighbors_sha256=array_sha256(query_neighbors), conventions=conventions,
                convention_resolution=resolution, preprocessing=preprocessing,
                preprocessing_receipt=dict(source_input_identity=identity.get("input_hash"),
                    training_standardization="raw float32; training population moments float64; standardized float32",
                    standardization_mean=float(row.standardization_mean),
                    standardization_std=float(row.standardization_std),
                    raw_values_verified=raw_values is not None),
                training_value_sha256=checkpoint["training_value_sha256"], source_run_identity=identity,
                **support_verification)
            snapshots.append(FrozenGaussianFit(grid_axes=axes, interpolation_indices=indices,
                interpolation_weights=weights, training_query_rows=training_rows, training_coordinates=training,
                local_dag=dag, query_coordinates=query, query_neighbor_indices=query_neighbors,
                global_jitter=float(covariance["prior_jitter"]),
                axis_stabilization=float(covariance["axis_factor_stabilization"]),
                local_jitter=float(conventions["local_jitter"]), local_neighbors=k,
                feature_position=int(row.feature_position), feature_name=str(row.gene),
                standardization_mean=float(row.standardization_mean), standardization_std=float(row.standardization_std),
                training_row_ids=row_ids[training_rows], query_row_ids=row_ids,
                training_values=None if raw_values is None else raw_values[:, column],
                source_variational_mean=source_means[:, column], source_provenance=provenance,
                construction_dtype=str(grid.numpy().dtype), coordinate_units=conventions["coordinate_units"],
                source_commit=commit, **parameters))
    if len(snapshots) != len(requested):
        raise ValueError("not all requested features had verified component checkpoints")
    return tuple(sorted(snapshots, key=lambda snapshot: snapshot.feature_position))
