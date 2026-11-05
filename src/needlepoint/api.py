"""Small stable entry points around the inherited scientific implementation."""
from importlib.resources import files
import tomllib


def default_workflow():
    """Return a fresh copy of the reference policy (not automatically applied to fits)."""
    return tomllib.loads(files('needlepoint').joinpath('resources/default_workflow.toml').read_text())


def normalize_counts(counts, *, target_sum=10000.0):
    """Copy counts to CSR, apply full-library CP10K/log1p, and retain zero rows.

    Supply the complete measured gene matrix, *before* selecting a gene panel.
    Coordinates and observation ordering are never changed. Negative or nonfinite
    values are rejected. Input is not modified.
    """
    import numpy as np
    from scipy import sparse
    from .io.h5ad import normalize_transform_csr
    if not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError('target_sum must be finite and positive')
    matrix = sparse.csr_matrix(counts, dtype=np.float32, copy=True)
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError('counts must be finite and nonnegative')
    return normalize_transform_csr(matrix, target_sum=target_sum, transform='log1p')
