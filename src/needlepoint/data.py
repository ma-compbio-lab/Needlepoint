from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix, load_npz, vstack


CHUNK_FILE_RE = re.compile(r"peak_features_chunk_(\d+)\.npz$")
SPOT_ID_RE = re.compile(r"Spot\s+(\d+)$")
H5AD_OBSM_COORD_CANDIDATES = (
    "spatial",
    "X_spatial",
    "spatial_raw",
    "spatial_aligned",
    "stereoseq",
    "StereoSeq",
)
H5AD_OBS_COORD_PAIR_CANDIDATES = (
    ("x", "y"),
    ("X", "Y"),
    ("coord_x", "coord_y"),
    ("x_coord", "y_coord"),
    ("bin_x", "bin_y"),
    ("array_col", "array_row"),
    ("imagecol", "imagerow"),
    ("col", "row"),
)
H5AD_LABEL_CANDIDATES = (
    "region_label",
    "tissue_region",
    "tissue_label",
    "anatomical_region",
    "anatomy",
    "annotation",
    "annotations",
    "domain",
    "spatial_domain",
    "layer",
    "truth",
    "ground_truth",
    "cell_type",
    "celltype",
    "Cell_type",
    "CellType",
    "subclass",
    "class",
)
H5AD_SLICE_CANDIDATES = (
    "slice_id",
    "section_id",
    "slide_id",
    "sample_id",
    "batch",
    "section",
    "slice",
    "slide",
)


@dataclass(frozen=True)
class FeatureChunk:
    chunk_index: int
    matrix_path: Path
    spot_ids_path: Path


@dataclass
class SpatialFeatureDataset:
    """Locations x features matrix plus spatial metadata.

    The proposal writes the matrix as features x locations. The executable code
    stores `X` as locations x features because that is the AnnData/scikit-learn
    convention. `feature_names` therefore index columns and `obs` indexes rows.
    """

    X: sparse.spmatrix | np.ndarray
    coords: np.ndarray
    feature_names: np.ndarray
    obs: pd.DataFrame | None = None
    feature_metadata: pd.DataFrame | None = None
    ground_truth: np.ndarray | None = None
    name: str = "dataset"


def discover_temi_chunks(
    feature_dir: str | Path,
    chunk_indices: list[int] | None = None,
    max_chunks: int | None = None,
) -> list[FeatureChunk]:
    feature_dir = Path(feature_dir)
    chunks: list[FeatureChunk] = []
    requested = set(chunk_indices) if chunk_indices is not None else None
    for matrix_path in feature_dir.glob("peak_features_chunk_*.npz"):
        match = CHUNK_FILE_RE.match(matrix_path.name)
        if match is None:
            continue
        chunk_index = int(match.group(1))
        if requested is not None and chunk_index not in requested:
            continue
        spot_ids_path = feature_dir / f"peak_features_chunk_{chunk_index}_spot_ids.npy"
        if not spot_ids_path.exists():
            raise FileNotFoundError(f"Missing spot id file for chunk {chunk_index}: {spot_ids_path}")
        chunks.append(FeatureChunk(chunk_index, matrix_path, spot_ids_path))
    chunks.sort(key=lambda item: item.chunk_index)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    if not chunks:
        raise FileNotFoundError(f"No TEMI feature chunks found in {feature_dir}")
    return chunks


def parse_spot_indices(spot_ids_path: str | Path) -> np.ndarray:
    raw = np.load(spot_ids_path, allow_pickle=True)
    indices = np.empty(raw.shape[0], dtype=np.int64)
    for idx, value in enumerate(raw):
        match = SPOT_ID_RE.match(str(value))
        if match is None:
            raise ValueError(f"Could not parse spot id {value!r} from {spot_ids_path}")
        indices[idx] = int(match.group(1)) - 1
    return indices


def load_temi_coordinates(coords_file: str | Path) -> pd.DataFrame:
    coords = pd.read_csv(coords_file, sep=";", comment="#")
    required = {"Spot index", "x", "y"}
    missing = required.difference(coords.columns)
    if missing:
        raise ValueError(f"Coordinate file is missing columns: {sorted(missing)}")
    coords = coords.rename(columns={"Spot index": "spot_index"})
    expected = np.arange(len(coords), dtype=np.int64)
    observed = coords["spot_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("TEMI coordinate rows are not in contiguous spot-index order")
    return coords


def load_temi_peak_metadata(path: str | Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_extraction_summary(path: str | Path | None) -> dict:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_transform_csr(matrix: sparse.spmatrix, target_sum: float, transform: str) -> csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    return normalize_transform_csr_with_row_sums(matrix, row_sums, target_sum=target_sum, transform=transform)


def normalize_transform_csr_with_row_sums(
    matrix: sparse.spmatrix,
    row_sums: np.ndarray,
    target_sum: float,
    transform: str,
) -> csr_matrix:
    """Normalize a sparse panel with externally computed per-spot totals.

    For selected-feature analyses, ``row_sums`` should normally be calculated
    from the complete detected spectrum before selecting columns. Normalizing
    by the selected panel silently makes every fitted peak depend on which
    other peaks happened to be selected.
    """
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    row_sums = np.asarray(row_sums, dtype=np.float32).reshape(-1)
    if row_sums.shape != (matrix.shape[0],):
        raise ValueError("row_sums must have one value per matrix row")
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    if matrix.nnz:
        row_counts = np.diff(matrix.indptr)
        row_ids = np.repeat(np.arange(matrix.shape[0], dtype=np.int32), row_counts)
        scale = np.ones(matrix.shape[0], dtype=np.float32)
        nonzero = row_sums > 0
        scale[nonzero] = np.float32(target_sum) / row_sums[nonzero]
        matrix.data *= scale[row_ids]
        if transform == "log1p":
            np.log1p(matrix.data, out=matrix.data)
        elif transform == "sqrt":
            np.sqrt(matrix.data, out=matrix.data)
        elif transform == "none":
            pass
        else:
            raise ValueError(f"Unsupported transform: {transform}")
    return matrix


def transform_csr(matrix: sparse.spmatrix, transform: str) -> csr_matrix:
    """Apply a variance-stabilizing transform without row normalization."""
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    if transform == "log1p":
        np.log1p(matrix.data, out=matrix.data)
    elif transform == "sqrt":
        np.sqrt(matrix.data, out=matrix.data)
    elif transform != "none":
        raise ValueError(f"Unsupported transform: {transform}")
    return matrix


def load_temi_selected_panel(
    chunks: list[FeatureChunk],
    selected_spots: np.ndarray,
    selected_feature_indices: np.ndarray,
    *,
    normalization_scope: str = "full-spectrum",
    target_sum: float = 1.0e4,
    transform: str = "sqrt",
) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    """Load selected TEMI columns while preserving full-spectrum spot totals.

    Returns the processed selected panel, source spot indices in row order, and
    the untransformed full-spectrum row sums. The implementation streams source
    chunks and never materializes the complete spot-by-peak matrix.
    """
    if normalization_scope not in {"full-spectrum", "selected-panel", "none"}:
        raise ValueError(
            "normalization_scope must be one of: full-spectrum, selected-panel, none"
        )
    selected_spots = np.asarray(selected_spots, dtype=np.int64)
    selected_feature_indices = np.asarray(selected_feature_indices, dtype=np.int64)
    if selected_spots.ndim != 1 or selected_feature_indices.ndim != 1:
        raise ValueError("selected_spots and selected_feature_indices must be one-dimensional")
    if selected_feature_indices.size == 0:
        raise ValueError("At least one selected feature is required")

    blocks: list[csr_matrix] = []
    spot_blocks: list[np.ndarray] = []
    total_blocks: list[np.ndarray] = []
    for chunk in chunks:
        chunk_spots = parse_spot_indices(chunk.spot_ids_path)
        keep = np.isin(chunk_spots, selected_spots, assume_unique=False)
        if not keep.any():
            continue
        full_block = load_npz(chunk.matrix_path).tocsr()[keep]
        if int(selected_feature_indices.max()) >= full_block.shape[1]:
            raise ValueError(
                f"Selected feature index exceeds columns in {chunk.matrix_path}"
            )
        full_totals = np.asarray(full_block.sum(axis=1), dtype=np.float32).ravel()
        panel = full_block[:, selected_feature_indices]
        if normalization_scope == "full-spectrum":
            panel = normalize_transform_csr_with_row_sums(
                panel,
                full_totals,
                target_sum=target_sum,
                transform=transform,
            )
        elif normalization_scope == "selected-panel":
            panel = normalize_transform_csr(panel, target_sum=target_sum, transform=transform)
        else:
            panel = transform_csr(panel, transform=transform)
        blocks.append(panel)
        spot_blocks.append(chunk_spots[keep])
        total_blocks.append(full_totals)

    if not blocks:
        raise RuntimeError("Selected spots did not overlap any TEMI feature chunks")
    panel = blocks[0] if len(blocks) == 1 else vstack(blocks, format="csr")
    loaded_spots = np.concatenate(spot_blocks)
    full_totals = np.concatenate(total_blocks)
    order = np.argsort(loaded_spots, kind="stable")
    return panel[order], loaded_spots[order], full_totals[order]


def infer_h5ad_label_key(obs_columns, preferred: tuple[str, ...] = H5AD_LABEL_CANDIDATES) -> str | None:
    columns = set(map(str, obs_columns))
    for key in preferred:
        if key in columns:
            return key
    lower_to_key = {str(key).lower(): str(key) for key in obs_columns}
    for key in preferred:
        observed = lower_to_key.get(key.lower())
        if observed is not None:
            return observed
    return None


def infer_h5ad_slice_key(obs_columns, preferred: tuple[str, ...] = H5AD_SLICE_CANDIDATES) -> str | None:
    columns = set(map(str, obs_columns))
    for key in preferred:
        if key in columns:
            return key
    lower_to_key = {str(key).lower(): str(key) for key in obs_columns}
    for key in preferred:
        observed = lower_to_key.get(key.lower())
        if observed is not None:
            return observed
    return None


def resolve_h5ad_coordinates(adata, coord_key: str = "spatial") -> tuple[np.ndarray, str]:
    if coord_key != "auto":
        if coord_key in adata.obsm:
            return np.asarray(adata.obsm[coord_key], dtype=np.float32), coord_key
        if coord_key in adata.obs and f"{coord_key}_y" in adata.obs:
            pair = (coord_key, f"{coord_key}_y")
            return adata.obs[list(pair)].to_numpy(dtype=np.float32), ",".join(pair)

    if coord_key == "auto":
        for key in H5AD_OBSM_COORD_CANDIDATES:
            if key in adata.obsm:
                return np.asarray(adata.obsm[key], dtype=np.float32), key
        for x_key, y_key in H5AD_OBS_COORD_PAIR_CANDIDATES:
            if {x_key, y_key}.issubset(adata.obs.columns):
                return adata.obs[[x_key, y_key]].to_numpy(dtype=np.float32), f"{x_key},{y_key}"

    if {"x", "y"}.issubset(adata.obs.columns):
        return adata.obs[["x", "y"]].to_numpy(dtype=np.float32), "x,y"
    if {"array_row", "array_col"}.issubset(adata.obs.columns):
        return adata.obs[["array_row", "array_col"]].to_numpy(dtype=np.float32), "array_row,array_col"
    raise KeyError(f"No coordinates found for coord_key={coord_key!r}")


def select_variable_features(
    sample_matrix: sparse.spmatrix,
    candidate_feature_indices: np.ndarray,
    n_features: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    sample_matrix = sample_matrix.tocsr()
    mean = np.asarray(sample_matrix.mean(axis=0)).ravel()
    mean_sq = np.asarray(sample_matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(mean_sq - mean**2, 0.0)
    dispersion = variance / (mean + 1.0e-6)
    keep_n = min(int(n_features), candidate_feature_indices.size)
    top_order = np.argsort(dispersion)[::-1][:keep_n]
    selected = candidate_feature_indices[top_order].astype(np.int32, copy=False)
    stats = pd.DataFrame(
        {
            "feature_index": candidate_feature_indices,
            "sample_mean": mean,
            "sample_variance": variance,
            "sample_dispersion": dispersion,
        }
    ).sort_values("sample_dispersion", ascending=False)
    return selected, stats


def build_temi_sample(
    chunks: list[FeatureChunk],
    coords: pd.DataFrame,
    candidate_feature_indices: np.ndarray | None = None,
    sample_per_chunk: int = 500,
    target_sum: float = 1.0e4,
    transform: str = "log1p",
    random_state: int = 0,
) -> tuple[csr_matrix, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(random_state)
    blocks: list[csr_matrix] = []
    sampled_spot_indices: list[np.ndarray] = []
    for chunk in chunks:
        matrix = load_npz(chunk.matrix_path).tocsr()
        n_rows = matrix.shape[0]
        sample_n = min(int(sample_per_chunk), n_rows)
        row_idx = np.arange(n_rows, dtype=np.int64)
        if sample_n < n_rows:
            row_idx = np.sort(rng.choice(row_idx, size=sample_n, replace=False))
        if candidate_feature_indices is not None:
            matrix = matrix[:, candidate_feature_indices]
        block = normalize_transform_csr(matrix[row_idx], target_sum=target_sum, transform=transform)
        blocks.append(block)
        chunk_spots = parse_spot_indices(chunk.spot_ids_path)
        sampled_spot_indices.append(chunk_spots[row_idx])
    X = blocks[0] if len(blocks) == 1 else vstack(blocks, format="csr")
    spot_indices = np.concatenate(sampled_spot_indices)
    obs = coords.iloc[spot_indices].copy().reset_index(drop=True)
    obs["source_spot_index"] = spot_indices
    obs["spot_id"] = [f"Spot {idx + 1}" for idx in spot_indices]
    sample_coords = obs[["x", "y"]].to_numpy(dtype=np.float32, copy=True)
    return X, sample_coords, obs


def load_h5ad_dataset(
    h5ad_path: str | Path,
    coord_key: str = "spatial",
    x_key: str | None = None,
    label_key: str | None = None,
    max_obs: int | None = None,
    max_vars: int | None = None,
    random_state: int = 0,
) -> SpatialFeatureDataset:
    h5ad_path = Path(h5ad_path)
    adata = ad.read_h5ad(h5ad_path)
    rng = np.random.default_rng(random_state)

    if max_obs is not None and adata.n_obs > max_obs:
        obs_idx = np.sort(rng.choice(adata.n_obs, size=max_obs, replace=False))
        adata = adata[obs_idx].copy()
    if max_vars is not None and adata.n_vars > max_vars:
        X_for_var = adata.X if x_key is None else adata.layers[x_key]
        if sparse.issparse(X_for_var):
            nnz = np.asarray((X_for_var > 0).sum(axis=0)).ravel()
            mean = np.asarray(X_for_var.mean(axis=0)).ravel()
        else:
            nnz = np.count_nonzero(X_for_var > 0, axis=0)
            mean = np.asarray(X_for_var).mean(axis=0)
        order = np.lexsort((-mean, -nnz))
        adata = adata[:, np.sort(order[:max_vars])].copy()

    coords, _ = resolve_h5ad_coordinates(adata, coord_key=coord_key)

    X = adata.X if x_key is None else adata.layers[x_key]
    if sparse.issparse(X):
        X = X.tocsr().astype(np.float32)
    else:
        X = np.asarray(X, dtype=np.float32)

    ground_truth = None
    if label_key == "auto":
        label_key = infer_h5ad_label_key(adata.obs.columns)
    if label_key is not None and label_key in adata.obs:
        ground_truth = adata.obs[label_key].to_numpy()

    feature_metadata = adata.var.copy()
    feature_metadata.insert(0, "feature_name", adata.var_names.astype(str))
    return SpatialFeatureDataset(
        X=X,
        coords=coords,
        feature_names=adata.var_names.astype(str).to_numpy(),
        obs=adata.obs.copy(),
        feature_metadata=feature_metadata.reset_index(drop=True),
        ground_truth=ground_truth,
        name=h5ad_path.stem,
    )
