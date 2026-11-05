from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.sparse.linalg import eigsh
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import kneighbors_graph


@dataclass
class SVGMethodResult:
    method: str
    table: pd.DataFrame
    basis_shape: tuple[int, int] | None = None


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = pvalues.size
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def _as_dense_float(matrix: sparse.spmatrix | np.ndarray) -> np.ndarray:
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


def standardize_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, ddof=0, keepdims=True)
    std[std < 1.0e-8] = 1.0
    return matrix / std


def scale_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    coords = coords[:, :2]
    span = coords.max(axis=0) - coords.min(axis=0)
    span[span < 1.0e-8] = 1.0
    return (coords - coords.min(axis=0)) / span


def default_length_scales(coords: np.ndarray, n_scales: int = 3) -> np.ndarray:
    coords = scale_coords(coords)
    rng = np.random.default_rng(0)
    if coords.shape[0] > 1000:
        idx = rng.choice(coords.shape[0], size=1000, replace=False)
        sample = coords[idx]
    else:
        sample = coords
    distances = pairwise_distances(sample)
    distances = distances[np.triu_indices_from(distances, k=1)]
    distances = distances[distances > 0]
    if distances.size == 0:
        return np.array([0.2], dtype=np.float64)
    quantiles = np.linspace(0.2, 0.8, n_scales)
    return np.unique(np.quantile(distances, quantiles))


def nystrom_rbf_basis(
    coords: np.ndarray,
    length_scale: float,
    n_components: int = 64,
    random_state: int = 0,
) -> np.ndarray:
    coords = scale_coords(coords)
    n = coords.shape[0]
    m = min(int(n_components), n)
    rng = np.random.default_rng(random_state)
    landmarks_idx = np.sort(rng.choice(n, size=m, replace=False))
    landmarks = coords[landmarks_idx]

    gamma = 1.0 / (2.0 * max(float(length_scale), 1.0e-6) ** 2)
    d_nm = pairwise_distances(coords, landmarks, squared=True)
    k_nm = np.exp(-gamma * d_nm)
    d_mm = pairwise_distances(landmarks, landmarks, squared=True)
    k_mm = np.exp(-gamma * d_mm)
    evals, evecs = np.linalg.eigh(k_mm + 1.0e-6 * np.eye(m))
    keep = evals > 1.0e-8
    basis = k_nm @ (evecs[:, keep] / np.sqrt(evals[keep]))
    return _orthonormal_basis(basis)


def knn_diffusion_basis(
    coords: np.ndarray,
    n_neighbors: int = 16,
    n_components: int = 64,
) -> np.ndarray:
    coords = scale_coords(coords)
    n = coords.shape[0]
    k = min(n_neighbors, n - 1)
    graph = kneighbors_graph(coords, n_neighbors=k, mode="distance", include_self=False)
    graph = graph.maximum(graph.T).tocsr()
    if graph.nnz:
        positive = graph.data[graph.data > 0]
        bandwidth = float(np.median(positive)) if positive.size else 1.0
        graph.data = np.exp(-(graph.data / max(bandwidth, 1.0e-8))).astype(np.float64)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    degree[degree <= 0] = 1.0
    inv_sqrt_degree = sparse.diags(1.0 / np.sqrt(degree))
    sym = inv_sqrt_degree @ graph @ inv_sqrt_degree
    usable = min(int(n_components) + 1, n - 1)
    if usable <= 1:
        return np.zeros((n, 0), dtype=np.float64)
    values, vectors = eigsh(sym, k=usable, which="LA")
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    # Drop the leading near-constant eigenvector.
    return _orthonormal_basis(vectors[:, 1 : min(usable, n_components + 1)])


def _orthonormal_basis(basis: np.ndarray) -> np.ndarray:
    basis = standardize_columns(np.asarray(basis, dtype=np.float64))
    if basis.size == 0:
        return basis.reshape(basis.shape[0], 0)
    q, r = np.linalg.qr(basis, mode="reduced")
    keep = np.abs(np.diag(r)) > 1.0e-8
    return q[:, keep]


def basis_f_test(
    expression: sparse.spmatrix | np.ndarray,
    basis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _as_dense_float(expression)
    basis = _orthonormal_basis(basis)
    n, n_genes = y.shape
    rank = basis.shape[1]
    if rank == 0:
        return np.ones(n_genes), np.zeros(n_genes), np.zeros(n_genes)
    y_centered = y - y.mean(axis=0, keepdims=True)
    total_ss = np.sum(y_centered**2, axis=0)
    projected = basis.T @ y_centered
    model_ss = np.sum(projected**2, axis=0)
    model_ss = np.minimum(model_ss, total_ss)
    residual_ss = np.maximum(total_ss - model_ss, 1.0e-12)
    df1 = rank
    df2 = max(n - rank - 1, 1)
    f_stat = (model_ss / df1) / (residual_ss / df2)
    pvalues = stats.f.sf(f_stat, df1, df2)
    pvalues = np.where(total_ss <= 1.0e-12, 1.0, pvalues)
    spatial_fraction = np.where(total_ss > 0, model_ss / total_ss, 0.0)
    return pvalues, f_stat, spatial_fraction


def run_spatialde_style(
    expression: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    gene_names: np.ndarray,
    length_scales: np.ndarray | None = None,
    n_components: int = 64,
    random_state: int = 0,
) -> SVGMethodResult:
    if length_scales is None:
        length_scales = default_length_scales(coords, n_scales=4)
    p_by_scale = []
    f_by_scale = []
    frac_by_scale = []
    for idx, scale in enumerate(length_scales):
        basis = nystrom_rbf_basis(
            coords,
            length_scale=float(scale),
            n_components=n_components,
            random_state=random_state + idx,
        )
        pvals, f_stats, fractions = basis_f_test(expression, basis)
        p_by_scale.append(pvals)
        f_by_scale.append(f_stats)
        frac_by_scale.append(fractions)
    p_mat = np.vstack(p_by_scale)
    f_mat = np.vstack(f_by_scale)
    frac_mat = np.vstack(frac_by_scale)
    best = np.argmin(p_mat, axis=0)
    raw_p = p_mat[best, np.arange(p_mat.shape[1])]
    p_bonf = np.minimum(raw_p * p_mat.shape[0], 1.0)
    table = pd.DataFrame(
        {
            "gene": gene_names,
            "pvalue": p_bonf,
            "qvalue": bh_fdr(p_bonf),
            "best_length_scale": np.asarray(length_scales)[best],
            "f_stat": f_mat[best, np.arange(f_mat.shape[1])],
            "spatial_fraction": frac_mat[best, np.arange(frac_mat.shape[1])],
        }
    ).sort_values(["qvalue", "pvalue", "spatial_fraction"], ascending=[True, True, False])
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return SVGMethodResult("SpatialDE_style_nystrom_gp", table, None)


def run_nnsvg_style(
    expression: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    gene_names: np.ndarray,
    n_neighbors: int = 16,
    n_components: int = 64,
) -> SVGMethodResult:
    basis = knn_diffusion_basis(coords, n_neighbors=n_neighbors, n_components=n_components)
    pvals, f_stats, fractions = basis_f_test(expression, basis)
    table = pd.DataFrame(
        {
            "gene": gene_names,
            "pvalue": pvals,
            "qvalue": bh_fdr(pvals),
            "f_stat": f_stats,
            "spatial_fraction": fractions,
            "n_neighbors": n_neighbors,
            "basis_rank": basis.shape[1],
        }
    ).sort_values(["qvalue", "pvalue", "spatial_fraction"], ascending=[True, True, False])
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return SVGMethodResult("nnSVG_style_knn_gp", table, basis.shape)


def pearson_count_residuals(counts: sparse.spmatrix | np.ndarray) -> np.ndarray:
    counts = _as_dense_float(counts)
    library = counts.sum(axis=1)
    gene_totals = counts.sum(axis=0)
    total = max(float(gene_totals.sum()), 1.0)
    p_gene = gene_totals / total
    expected = np.outer(library, p_gene)
    # Mild overdispersion guard keeps very high-count spots from dominating.
    variance = expected + 0.05 * expected**2
    residuals = (counts - expected) / np.sqrt(np.maximum(variance, 1.0e-8))
    return standardize_columns(residuals)


def cauchy_combine(pvalue_matrix: np.ndarray) -> np.ndarray:
    pvalue_matrix = np.asarray(pvalue_matrix, dtype=np.float64)
    pvalue_matrix = np.clip(pvalue_matrix, 1.0e-15, 1.0 - 1.0e-15)
    tans = np.tan((0.5 - pvalue_matrix) * np.pi)
    statistic = np.mean(tans, axis=0)
    combined = 0.5 - np.arctan(statistic) / np.pi
    return np.clip(combined, 0.0, 1.0)


def run_spark_style(
    counts: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    gene_names: np.ndarray,
    length_scales: np.ndarray | None = None,
    n_components: int = 48,
    random_state: int = 0,
) -> SVGMethodResult:
    residuals = pearson_count_residuals(counts)
    if length_scales is None:
        length_scales = default_length_scales(coords, n_scales=4)
    p_by_kernel = []
    frac_by_kernel = []
    kernel_names = []
    for idx, scale in enumerate(length_scales):
        basis = nystrom_rbf_basis(
            coords,
            length_scale=float(scale),
            n_components=n_components,
            random_state=random_state + idx,
        )
        pvals, _, fractions = basis_f_test(residuals, basis)
        p_by_kernel.append(pvals)
        frac_by_kernel.append(fractions)
        kernel_names.append(f"rbf_{scale:.4f}")
    # Add a broad linear-coordinate kernel surrogate.
    coord_basis = _orthonormal_basis(scale_coords(coords))
    pvals, _, fractions = basis_f_test(residuals, coord_basis)
    p_by_kernel.append(pvals)
    frac_by_kernel.append(fractions)
    kernel_names.append("linear_coords")

    p_mat = np.vstack(p_by_kernel)
    frac_mat = np.vstack(frac_by_kernel)
    combined = cauchy_combine(p_mat)
    best = np.argmin(p_mat, axis=0)
    table = pd.DataFrame(
        {
            "gene": gene_names,
            "pvalue": combined,
            "qvalue": bh_fdr(combined),
            "best_kernel": np.asarray(kernel_names)[best],
            "best_kernel_pvalue": p_mat[best, np.arange(p_mat.shape[1])],
            "max_spatial_fraction": frac_mat.max(axis=0),
        }
    ).sort_values(["qvalue", "pvalue", "max_spatial_fraction"], ascending=[True, True, False])
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return SVGMethodResult("SPARK_style_multikernel_count_residual", table, None)
