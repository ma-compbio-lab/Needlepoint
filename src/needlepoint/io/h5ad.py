from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix


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
    ("x_um", "y_um"),
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


def normalize_transform_csr(
    matrix: sparse.spmatrix,
    target_sum: float,
    transform: str,
) -> csr_matrix:
    """Normalize each row by its full-library total and transform nonzero values."""

    matrix = matrix.tocsr(copy=True).astype(np.float32)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    if target_sum <= 0:
        raise ValueError("target_sum must be positive")
    if matrix.nnz:
        row_ids = np.repeat(
            np.arange(matrix.shape[0], dtype=np.int32),
            np.diff(matrix.indptr),
        )
        scale = np.ones(matrix.shape[0], dtype=np.float32)
        nonzero = row_sums > 0
        scale[nonzero] = np.float32(target_sum) / row_sums[nonzero]
        matrix.data *= scale[row_ids]
        if transform == "log1p":
            np.log1p(matrix.data, out=matrix.data)
        elif transform == "sqrt":
            np.sqrt(matrix.data, out=matrix.data)
        elif transform != "none":
            raise ValueError(f"Unsupported transform: {transform}")
    return matrix


def infer_h5ad_label_key(
    obs_columns,
    preferred: tuple[str, ...] = H5AD_LABEL_CANDIDATES,
) -> str | None:
    columns = set(str(column) for column in obs_columns)
    for key in preferred:
        if key in columns:
            return key
    return None


def resolve_h5ad_coordinates(adata, coord_key: str = "spatial") -> tuple[np.ndarray, str]:
    """Resolve two-dimensional spatial coordinates from an AnnData object."""

    if coord_key != "auto":
        if coord_key in adata.obsm:
            coords = np.asarray(adata.obsm[coord_key])
            return _validate_coordinates(coords, f"obsm:{coord_key}"), f"obsm:{coord_key}"
        obs_pair = _parse_obs_pair(coord_key)
        if obs_pair is not None and all(key in adata.obs for key in obs_pair):
            coords = adata.obs[list(obs_pair)].to_numpy()
            label = f"obs:{obs_pair[0]},{obs_pair[1]}"
            return _validate_coordinates(coords, label), label
        raise KeyError(
            f"Coordinate key {coord_key!r} was not found in obsm and is not an "
            "available comma-separated obs coordinate pair"
        )

    for key in H5AD_OBSM_COORD_CANDIDATES:
        if key in adata.obsm:
            coords = np.asarray(adata.obsm[key])
            return _validate_coordinates(coords, f"obsm:{key}"), f"obsm:{key}"
    for pair in H5AD_OBS_COORD_PAIR_CANDIDATES:
        if all(key in adata.obs for key in pair):
            coords = adata.obs[list(pair)].to_numpy()
            label = f"obs:{pair[0]},{pair[1]}"
            return _validate_coordinates(coords, label), label
    raise KeyError(
        "Could not infer H5AD coordinates; pass --coord-key with an obsm key or "
        "a comma-separated obs pair such as x_um,y_um"
    )


def _parse_obs_pair(value: str) -> tuple[str, str] | None:
    tokens = [token.strip() for token in value.split(",")]
    if len(tokens) == 2 and all(tokens):
        return tokens[0], tokens[1]
    return None


def _validate_coordinates(coords: np.ndarray, label: str) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Coordinates from {label} must have shape (n_spots, >=2)")
    coords = coords[:, :2]
    if not np.isfinite(coords).all():
        raise ValueError(f"Coordinates from {label} contain non-finite values")
    return coords
