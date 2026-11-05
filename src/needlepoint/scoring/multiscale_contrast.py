#!/usr/bin/env python3
"""Select spatially variable genes using multiscale spatial contrast.

The score mirrors the contrast-first TEMI 202604 peak screen: each normalized,
transformed feature map is Gaussian-smoothed at several physical bandwidths,
and spatial contrast is the fraction of the raw feature variance retained by
the smoothed map.  The script prefilters genes by count dispersion, penalizes
simple library-size/axis/edge gradients, calibrates scores against
prevalence-matched permutation nulls, and renders raw-expression previews for
review before V-SKI fitting.  The former fixed-size, scale-balanced panel is
available as an explicit legacy selection mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from scipy import sparse
from scipy import fft as scipy_fft
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

from needlepoint.io.h5ad import resolve_h5ad_coordinates
from needlepoint.scoring.gpu_spatial_contrast import TorchFFTGridScorer


EPS = 1.0e-12
def load_cuda_fft_scorer_class():
    """Return the package-qualified CUDA backend.

    Kept as a function for compatibility with the original scorer tests and
    callers, without dynamic loading or ambiguous ``msflbm`` imports.
    """

    return TorchFFTGridScorer


@dataclass(frozen=True)
class GridGeometry:
    unique_x: np.ndarray
    unique_y: np.ndarray
    x_index: np.ndarray
    y_index: np.ndarray
    mask: np.ndarray
    spacing_x: float
    spacing_y: float


@dataclass(frozen=True)
class FFTGridSmoother:
    padded_shape: tuple[int, int]
    offset_y: int
    offset_x: int
    kernels: dict[float, np.ndarray]
    denominators: dict[float, np.ndarray]
    workers: int


@dataclass
class SpatialScoringContext:
    geometry: GridGeometry | None
    denominators: dict[float, np.ndarray]
    fft_smoother: FFTGridSmoother | None
    knn_kernels: dict[float, sparse.csr_matrix]
    x: np.ndarray
    y: np.ndarray
    edge_distance: np.ndarray
    library_reference: np.ndarray
    cuda_fft_scorer: object | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coord-key", default="auto")
    parser.add_argument("--coord-scale", type=float, default=1.0)
    parser.add_argument("--layer", default=None)
    parser.add_argument(
        "--bandwidths-um",
        type=float,
        nargs="+",
        default=[20.0, 40.0, 80.0, 160.0, 320.0, 640.0],
    )
    parser.add_argument("--min-spots", type=int, default=20)
    parser.add_argument(
        "--n-candidate-genes",
        type=int,
        default=0,
        help="Prefilter this many genes by raw-count dispersion; <=0 scores all eligible genes.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["empirical-fdr", "legacy-panel-size"],
        default="empirical-fdr",
        help="Use data-dependent empirical FDR (default) or the former fixed-size panel.",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=722,
        help="Number of genes to select only when --selection-mode=legacy-panel-size.",
    )
    parser.add_argument("--fdr-q", type=float, default=0.05)
    parser.add_argument("--max-artifact-like-score", type=float, default=0.35)
    parser.add_argument("--n-prevalence-strata", type=int, default=10)
    parser.add_argument("--null-profiles-per-stratum", type=int, default=2000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument(
        "--null-checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for atomic per-chunk empirical-null checkpoints. "
            "The directory is configuration-validated before reuse."
        ),
    )
    parser.add_argument(
        "--resume-null-checkpoints",
        action="store_true",
        help="Resume complete null chunks from --null-checkpoint-dir.",
    )
    parser.add_argument(
        "--minimum-selected-genes",
        type=int,
        default=20,
        help="Exit nonzero after writing audits if fewer genes pass selection.",
    )
    parser.add_argument("--top-per-scale", type=int, default=150)
    parser.add_argument("--target-sum", type=float, default=1.0e4)
    parser.add_argument("--transform", choices=["log1p", "sqrt", "none"], default="sqrt")
    parser.add_argument("--chunk-size-genes", type=int, default=32)
    parser.add_argument(
        "--smoothing-backend",
        choices=["auto", "grid-gaussian", "grid-fft", "grid-fft-cuda"],
        default="auto",
        help=(
            "Auto uses CPU FFT smoothing for large grids/bandwidths. "
            "grid-fft-cuda explicitly enables the parity-tested PyTorch CUDA backend."
        ),
    )
    parser.add_argument("--fft-workers", type=int, default=4)
    parser.add_argument(
        "--cuda-device",
        default="cuda",
        help="PyTorch CUDA device used only by --smoothing-backend=grid-fft-cuda.",
    )
    parser.add_argument(
        "--gpu-reduction-accumulator",
        choices=["float32", "float64"],
        default="float64",
        help=(
            "Accumulator for GPU SSE/variance reductions. float64 is the "
            "scientific-parity default; FFTs remain float32/complex64."
        ),
    )
    parser.add_argument("--plot-top-n", type=int, default=12)
    parser.add_argument("--clip-percentile", type=float, default=99.5)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--point-size", type=float, default=0.5)
    parser.add_argument(
        "--allow-irregular-scatter",
        action="store_true",
        help="Allow scatter previews only when coordinates cannot be represented as a regular grid.",
    )
    return parser.parse_args()


def read_count_matrix(adata, layer: str | None):
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"AnnData layer {layer!r} is absent; available: {list(adata.layers.keys())}")
        matrix = adata.layers[layer]
    elif adata.X is not None:
        matrix = adata.X
    elif "raw_count" in adata.layers:
        matrix = adata.layers["raw_count"]
    else:
        raise ValueError("AnnData has no X matrix; pass --layer with the count matrix")
    if sparse.issparse(matrix):
        return matrix.tocsr().astype(np.float32)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float32))


def compute_gene_statistics(matrix, genes: np.ndarray) -> pd.DataFrame:
    n_spots = matrix.shape[0]
    expressed = np.asarray(matrix.getnnz(axis=0)).ravel().astype(np.int64)
    total = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
    total_sq = np.asarray(matrix.power(2).sum(axis=0)).ravel().astype(np.float64)
    mean = total / max(n_spots, 1)
    variance = np.maximum(total_sq / max(n_spots, 1) - mean * mean, 0.0)
    dispersion = variance / np.maximum(mean, 1.0e-6)
    return pd.DataFrame(
        {
            "feature_index": np.arange(matrix.shape[1], dtype=np.int64),
            "gene": genes.astype(str),
            "n_spots_expressed": expressed,
            "relative_prevalence_raw": expressed / max(n_spots, 1),
            "total_counts": total,
            "mean_counts": mean,
            "raw_count_variance": variance,
            "dispersion": dispersion,
        }
    )


def choose_candidates(stats: pd.DataFrame, min_spots: int, n_candidates: int) -> pd.DataFrame:
    eligible = stats.loc[stats["n_spots_expressed"] >= int(min_spots)].copy()
    if eligible.empty:
        raise ValueError(f"No genes pass min_spots={min_spots}")
    eligible = eligible.sort_values(
        ["dispersion", "n_spots_expressed", "gene"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    if n_candidates > 0:
        eligible = eligible.head(int(n_candidates))
    eligible["dispersion_prefilter_rank"] = np.arange(1, eligible.shape[0] + 1)
    eligible = eligible.sort_values("feature_index").reset_index(drop=True)
    eligible.insert(0, "candidate_position", np.arange(eligible.shape[0], dtype=np.int64))
    return eligible


def normalize_transform_selected_csr(
    counts: sparse.spmatrix,
    selected_indices: np.ndarray,
    library_totals: np.ndarray,
    target_sum: float,
    transform: str,
) -> sparse.csc_matrix:
    """Normalize only selected columns using totals from the complete library.

    This is numerically equivalent to normalizing the complete matrix before
    selecting columns, but avoids copying and transforming ineligible genes.
    """

    if float(target_sum) <= 0:
        raise ValueError("target_sum must be positive")
    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    matrix = counts[:, selected_indices].tocsr(copy=True).astype(np.float32)
    totals = np.asarray(library_totals, dtype=np.float32)
    if totals.shape != (matrix.shape[0],):
        raise ValueError("library_totals must have one value per spot")
    if matrix.nnz:
        row_ids = np.repeat(
            np.arange(matrix.shape[0], dtype=np.int32),
            np.diff(matrix.indptr),
        )
        scale = np.ones(matrix.shape[0], dtype=np.float32)
        nonzero = totals > 0
        scale[nonzero] = np.float32(target_sum) / totals[nonzero]
        matrix.data *= scale[row_ids]
        if transform == "log1p":
            np.log1p(matrix.data, out=matrix.data)
        elif transform == "sqrt":
            np.sqrt(matrix.data, out=matrix.data)
        elif transform != "none":
            raise ValueError(f"Unsupported transform: {transform}")
    return matrix.tocsc()


def build_grid_geometry(coords: np.ndarray) -> GridGeometry | None:
    if coords.ndim != 2 or coords.shape[1] < 2 or not np.all(np.isfinite(coords[:, :2])):
        return None
    x = np.asarray(coords[:, 0], dtype=np.float64)
    y = np.asarray(coords[:, 1], dtype=np.float64)
    inferred_x = infer_grid_axis(x)
    inferred_y = infer_grid_axis(y)
    if inferred_x is None or inferred_y is None:
        return None
    unique_x, x_index, spacing_x = inferred_x
    unique_y, y_index, spacing_y = inferred_y
    grid_cells = int(unique_x.size * unique_y.size)
    if grid_cells > max(5 * coords.shape[0], 5_000_000):
        return None
    if np.unique(np.column_stack([y_index, x_index]), axis=0).shape[0] != coords.shape[0]:
        return None
    mask = np.zeros((unique_y.size, unique_x.size), dtype=np.float32)
    mask[y_index, x_index] = 1.0
    return GridGeometry(unique_x, unique_y, x_index, y_index, mask, spacing_x, spacing_y)


def coordinate_grid_audit(coords: np.ndarray, geometry: GridGeometry | None) -> dict:
    coordinate_pairs = np.asarray(coords[:, :2], dtype=np.float64)
    unique_pairs = np.unique(coordinate_pairs, axis=0)
    audit = {
        "n_observed_bins": int(coordinate_pairs.shape[0]),
        "n_unique_coordinate_pairs": int(unique_pairs.shape[0]),
        "n_duplicate_coordinate_pairs": int(coordinate_pairs.shape[0] - unique_pairs.shape[0]),
        "finite_coordinates": bool(np.all(np.isfinite(coordinate_pairs))),
        "regular_grid_inferred": geometry is not None,
    }
    if geometry is not None:
        grid_cells = int(geometry.mask.size)
        audit.update(
            {
                "grid_shape_yx": list(map(int, geometry.mask.shape)),
                "grid_spacing_xy": [float(geometry.spacing_x), float(geometry.spacing_y)],
                "grid_cells": grid_cells,
                "observed_grid_cells": int(np.count_nonzero(geometry.mask)),
                "observed_grid_coverage": float(np.count_nonzero(geometry.mask) / max(grid_cells, 1)),
                "preview_render_mode": "filled_grid_nearest",
            }
        )
    else:
        audit["preview_render_mode"] = "irregular_scatter_or_failure"
    return audit


def infer_grid_axis(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Infer a complete regular axis even when entire tissue rows/columns are absent."""

    values = np.asarray(values, dtype=np.float64)
    observed = np.unique(values)
    if observed.size < 2:
        return None
    differences = np.diff(observed)
    differences = differences[differences > 0]
    if not differences.size:
        return None
    # Missing full rows/columns create integer multiples of the base pitch.
    # Coordinates in these H5ADs are integer bin indices (possibly rescaled to
    # microns), so the smallest observed positive step recovers that pitch.
    spacing = float(np.min(differences))
    if not np.isfinite(spacing) or spacing <= 0:
        return None
    multiples = differences / spacing
    if not np.allclose(multiples, np.rint(multiples), rtol=0.0, atol=0.05):
        return None
    indices = np.rint((values - observed.min()) / spacing).astype(np.int64)
    reconstructed = observed.min() + indices * spacing
    if not np.allclose(values, reconstructed, rtol=0.0, atol=max(1.0e-5, spacing * 0.01)):
        return None
    axis = observed.min() + np.arange(int(indices.max()) + 1, dtype=np.float64) * spacing
    return axis, indices.astype(np.int32, copy=False), spacing


def precompute_grid_denominators(
    geometry: GridGeometry,
    bandwidths: list[float],
) -> dict[float, np.ndarray]:
    output: dict[float, np.ndarray] = {}
    for bandwidth in bandwidths:
        sigma = (
            max(float(bandwidth) / geometry.spacing_y, 1.0e-6),
            max(float(bandwidth) / geometry.spacing_x, 1.0e-6),
        )
        denominator = gaussian_filter(geometry.mask, sigma=sigma, mode="constant", cval=0.0)
        output[float(bandwidth)] = np.maximum(denominator, 1.0e-6).astype(np.float32)
    return output


def smooth_grid_block(
    block: np.ndarray,
    geometry: GridGeometry,
    denominator: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    canvas = np.zeros((*geometry.mask.shape, block.shape[1]), dtype=np.float32)
    canvas[geometry.y_index, geometry.x_index, :] = block
    sigma = (
        max(float(bandwidth) / geometry.spacing_y, 1.0e-6),
        max(float(bandwidth) / geometry.spacing_x, 1.0e-6),
        0.0,
    )
    smoothed = gaussian_filter(canvas, sigma=sigma, mode="constant", cval=0.0)
    smoothed /= denominator[:, :, None]
    return smoothed[geometry.y_index, geometry.x_index, :].astype(np.float32, copy=False)


def resolve_smoothing_backend(
    requested: str,
    geometry: GridGeometry | None,
    bandwidths: list[float],
    n_spots: int,
) -> str:
    if geometry is None:
        if requested != "auto":
            raise ValueError(f"{requested} requires regular-grid coordinates")
        return "knn-gaussian"
    if requested != "auto":
        return requested
    max_sigma = max(
        max(float(value) / geometry.spacing_x, float(value) / geometry.spacing_y)
        for value in bandwidths
    )
    return "grid-fft" if n_spots >= 100_000 or max_sigma >= 40 else "grid-gaussian"


def build_fft_grid_smoother(
    geometry: GridGeometry,
    bandwidths: list[float],
    workers: int,
) -> FFTGridSmoother:
    max_sigma_y = max(float(value) / geometry.spacing_y for value in bandwidths)
    max_sigma_x = max(float(value) / geometry.spacing_x for value in bandwidths)
    offset_y = int(np.ceil(4.0 * max_sigma_y))
    offset_x = int(np.ceil(4.0 * max_sigma_x))
    padded_shape = (
        int(scipy_fft.next_fast_len(geometry.mask.shape[0] + 2 * offset_y)),
        int(scipy_fft.next_fast_len(geometry.mask.shape[1] + 2 * offset_x)),
    )
    padded_mask = np.zeros(padded_shape, dtype=np.float32)
    y_slice = slice(offset_y, offset_y + geometry.mask.shape[0])
    x_slice = slice(offset_x, offset_x + geometry.mask.shape[1])
    padded_mask[y_slice, x_slice] = geometry.mask
    workers = max(int(workers), 1)
    mask_fft = scipy_fft.rfftn(padded_mask, axes=(0, 1), workers=workers)
    frequency_y = scipy_fft.fftfreq(padded_shape[0])[:, None]
    frequency_x = scipy_fft.rfftfreq(padded_shape[1])[None, :]
    kernels: dict[float, np.ndarray] = {}
    denominators: dict[float, np.ndarray] = {}
    for bandwidth in bandwidths:
        sigma_y = float(bandwidth) / geometry.spacing_y
        sigma_x = float(bandwidth) / geometry.spacing_x
        kernel = np.exp(
            -2.0
            * np.pi**2
            * (sigma_y * sigma_y * frequency_y * frequency_y + sigma_x * sigma_x * frequency_x * frequency_x)
        ).astype(np.float32)
        denominator_padded = scipy_fft.irfftn(
            mask_fft * kernel,
            s=padded_shape,
            axes=(0, 1),
            workers=workers,
        )
        kernels[float(bandwidth)] = kernel
        denominators[float(bandwidth)] = np.maximum(
            denominator_padded[y_slice, x_slice],
            1.0e-6,
        ).astype(np.float32)
    return FFTGridSmoother(
        padded_shape=padded_shape,
        offset_y=offset_y,
        offset_x=offset_x,
        kernels=kernels,
        denominators=denominators,
        workers=workers,
    )


def smooth_grid_block_fft(
    block: np.ndarray,
    geometry: GridGeometry,
    smoother: FFTGridSmoother,
    bandwidth: float,
    block_fft: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if block_fft is None:
        canvas = np.zeros((*smoother.padded_shape, block.shape[1]), dtype=np.float32)
        canvas[
            geometry.y_index + smoother.offset_y,
            geometry.x_index + smoother.offset_x,
            :,
        ] = block
        block_fft = scipy_fft.rfftn(canvas, axes=(0, 1), workers=smoother.workers)
    smoothed_padded = scipy_fft.irfftn(
        block_fft * smoother.kernels[float(bandwidth)][:, :, None],
        s=smoother.padded_shape,
        axes=(0, 1),
        workers=smoother.workers,
    )
    y_slice = slice(smoother.offset_y, smoother.offset_y + geometry.mask.shape[0])
    x_slice = slice(smoother.offset_x, smoother.offset_x + geometry.mask.shape[1])
    smoothed_grid = smoothed_padded[y_slice, x_slice, :]
    smoothed_grid /= smoother.denominators[float(bandwidth)][:, :, None]
    values = smoothed_grid[geometry.y_index, geometry.x_index, :].astype(np.float32, copy=False)
    return values, block_fft


def build_knn_kernel(coords: np.ndarray, bandwidth: float, n_neighbors: int = 64):
    n_neighbors = min(int(n_neighbors), coords.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1).fit(coords)
    distances, indices = nn.kneighbors(coords)
    weights = np.exp(-0.5 * (distances / float(bandwidth)) ** 2).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-12)
    rows = np.repeat(np.arange(coords.shape[0], dtype=np.int64), n_neighbors)
    return sparse.csr_matrix(
        (weights.ravel(), (rows, indices.ravel())),
        shape=(coords.shape[0], coords.shape[0]),
    )


def safe_correlations(block: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    values = np.asarray(block, dtype=np.float64)
    ref_centered = ref - ref.mean()
    value_centered = values - values.mean(axis=0, keepdims=True)
    numerator = ref_centered @ value_centered
    denominator = np.sqrt(
        np.dot(ref_centered, ref_centered)
        * np.einsum("ij,ij->j", value_centered, value_centered)
    )
    output = np.zeros(values.shape[1], dtype=np.float32)
    valid = denominator > EPS
    output[valid] = (numerator[valid] / denominator[valid]).astype(np.float32)
    return np.clip(output, -1.0, 1.0)


def compute_scale_metrics(
    block: np.ndarray,
    smoothed: np.ndarray,
    tss: np.ndarray,
    variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual = block - smoothed
    sse = np.einsum("ij,ij->j", residual, residual, dtype=np.float64)
    r2 = np.zeros(block.shape[1], dtype=np.float32)
    valid_tss = tss > EPS
    r2[valid_tss] = (1.0 - sse[valid_tss] / tss[valid_tss]).astype(np.float32)
    r2 = np.clip(r2, 0.0, 1.0)

    smooth_mean = smoothed.mean(axis=0, dtype=np.float64)
    smooth_var = np.maximum(
        np.einsum("ij,ij->j", smoothed, smoothed, dtype=np.float64) / smoothed.shape[0]
        - smooth_mean * smooth_mean,
        0.0,
    )
    contrast = np.zeros(block.shape[1], dtype=np.float32)
    valid_var = variance > EPS
    contrast[valid_var] = (smooth_var[valid_var] / variance[valid_var]).astype(np.float32)
    return r2, np.clip(contrast, 0.0, 1.0)


def build_spatial_scoring_context(
    coords: np.ndarray,
    library_totals: np.ndarray,
    bandwidths: list[float],
    smoothing_backend: str,
    fft_workers: int,
    geometry: GridGeometry | None = None,
    cuda_device: str = "cuda",
    gpu_reduction_accumulator: str = "float64",
) -> SpatialScoringContext:
    if geometry is None:
        geometry = build_grid_geometry(coords)
    denominators = (
        precompute_grid_denominators(geometry, bandwidths)
        if geometry is not None and smoothing_backend == "grid-gaussian"
        else {}
    )
    fft_smoother = (
        build_fft_grid_smoother(geometry, bandwidths, fft_workers)
        if geometry is not None and smoothing_backend in {"grid-fft", "grid-fft-cuda"}
        else None
    )
    x = np.asarray(coords[:, 0], dtype=np.float64)
    y = np.asarray(coords[:, 1], dtype=np.float64)
    edge_distance = np.minimum.reduce([x - x.min(), x.max() - x, y - y.min(), y.max() - y])
    context = SpatialScoringContext(
        geometry=geometry,
        denominators=denominators,
        fft_smoother=fft_smoother,
        knn_kernels={},
        x=x,
        y=y,
        edge_distance=edge_distance,
        library_reference=np.log1p(np.asarray(library_totals, dtype=np.float64)),
    )
    if smoothing_backend == "grid-fft-cuda":
        if geometry is None or fft_smoother is None:
            raise ValueError("grid-fft-cuda requires regular-grid coordinates")
        TorchFFTGridScorer = load_cuda_fft_scorer_class()
        context.cuda_fft_scorer = TorchFFTGridScorer(
            geometry,
            fft_smoother,
            device=cuda_device,
            reduction_accumulator=gpu_reduction_accumulator,
        )
    return context


def score_dense_feature_block(
    block: np.ndarray,
    coords: np.ndarray,
    bandwidths: list[float],
    smoothing_backend: str,
    context: SpatialScoringContext,
    candidate_positions: np.ndarray,
) -> pd.DataFrame:
    """Score one dense feature block using precomputed spatial resources."""

    block = np.asarray(block, dtype=np.float32)
    if block.ndim != 2 or block.shape[0] != coords.shape[0]:
        raise ValueError("Feature block must have shape (n_spots, n_features)")
    if candidate_positions.shape != (block.shape[1],):
        raise ValueError("candidate_positions must have one entry per feature")

    n_spots = block.shape[0]
    value_sum = block.sum(axis=0, dtype=np.float64)
    value_sum_sq = np.einsum("ij,ij->j", block, block, dtype=np.float64)
    mean = value_sum / max(n_spots, 1)
    variance = np.maximum(value_sum_sq / max(n_spots, 1) - mean * mean, 0.0)
    tss = np.maximum(value_sum_sq - value_sum * value_sum / max(n_spots, 1), 0.0)
    frame = pd.DataFrame(
        {
            "candidate_position": np.asarray(candidate_positions, dtype=np.int64),
            "normalized_mean": mean,
            "normalized_variance": variance,
            "prevalence": (block > 0).sum(axis=0) / max(n_spots, 1),
            "library_size_correlation": safe_correlations(block, context.library_reference),
            "x_correlation": safe_correlations(block, context.x),
            "y_correlation": safe_correlations(block, context.y),
            "edge_distance_correlation": safe_correlations(block, context.edge_distance),
        }
    )
    block_fft = None
    for bandwidth in bandwidths:
        if context.geometry is not None and smoothing_backend == "grid-gaussian":
            smoothed = smooth_grid_block(
                block,
                context.geometry,
                context.denominators[float(bandwidth)],
                float(bandwidth),
            )
        elif context.geometry is not None and smoothing_backend == "grid-fft":
            assert context.fft_smoother is not None
            smoothed, block_fft = smooth_grid_block_fft(
                block,
                context.geometry,
                context.fft_smoother,
                float(bandwidth),
                block_fft,
            )
        elif context.geometry is not None and smoothing_backend == "grid-fft-cuda":
            if context.cuda_fft_scorer is None:
                raise RuntimeError("CUDA FFT scorer was not initialized")
            scale_metrics = context.cuda_fft_scorer.score_scale_metrics(
                block,
                bandwidths,
                tss=tss,
                variance=variance,
            )
            for cuda_bandwidth in bandwidths:
                r2, contrast = scale_metrics[float(cuda_bandwidth)]
                tag = f"{int(round(cuda_bandwidth))}um"
                frame[f"scale_r2_{tag}"] = r2
                frame[f"scale_contrast_{tag}"] = contrast
            break
        else:
            if float(bandwidth) not in context.knn_kernels:
                context.knn_kernels[float(bandwidth)] = build_knn_kernel(coords, float(bandwidth))
            smoothed = context.knn_kernels[float(bandwidth)].dot(block).astype(np.float32, copy=False)
        r2, contrast = compute_scale_metrics(block, smoothed, tss, variance)
        tag = f"{int(round(bandwidth))}um"
        frame[f"scale_r2_{tag}"] = r2
        frame[f"scale_contrast_{tag}"] = contrast
    return frame


def finalize_spatial_scores(scores: pd.DataFrame, bandwidths: list[float]) -> pd.DataFrame:
    """Add nuisance penalty, best scale, and the combined spatial score."""

    scores = scores.copy()
    nuisance = np.maximum.reduce(
        [
            np.abs(scores["library_size_correlation"].to_numpy()),
            np.abs(scores["x_correlation"].to_numpy()),
            np.abs(scores["y_correlation"].to_numpy()),
            np.abs(scores["edge_distance_correlation"].to_numpy()),
        ]
    )
    scores["artifact_like_score"] = np.clip(nuisance, 0.0, 1.0)
    contrast_columns = [f"scale_contrast_{int(round(value))}um" for value in bandwidths]
    contrast_matrix = scores[contrast_columns].to_numpy(dtype=np.float32)
    best = np.argmax(contrast_matrix, axis=1)
    scores["max_spatial_contrast"] = contrast_matrix[np.arange(scores.shape[0]), best]
    scores["best_contrast_bandwidth_um"] = np.asarray(bandwidths, dtype=np.float32)[best]
    prevalence_weight = np.clip(
        np.sqrt(scores["prevalence"].to_numpy(dtype=np.float64) / 0.01),
        0.0,
        1.0,
    )
    scores["spatial_contrast_score"] = np.clip(
        scores["max_spatial_contrast"].to_numpy(dtype=np.float64)
        * (1.0 - scores["artifact_like_score"].to_numpy(dtype=np.float64))
        * prevalence_weight,
        0.0,
        1.0,
    )
    return scores


def score_candidate_matrix(
    matrix,
    coords: np.ndarray,
    library_totals: np.ndarray,
    bandwidths: list[float],
    chunk_size: int,
    smoothing_backend: str,
    fft_workers: int,
    context: SpatialScoringContext | None = None,
) -> pd.DataFrame:
    if context is None:
        context = build_spatial_scoring_context(
            coords,
            library_totals,
            bandwidths,
            smoothing_backend,
            fft_workers,
        )

    frames: list[pd.DataFrame] = []
    for start in range(0, matrix.shape[1], int(chunk_size)):
        stop = min(start + int(chunk_size), matrix.shape[1])
        print(f"Scoring genes {start + 1:,}-{stop:,} of {matrix.shape[1]:,}", flush=True)
        block = matrix[:, start:stop].toarray().astype(np.float32, copy=False)
        frames.append(
            score_dense_feature_block(
                block,
                coords,
                bandwidths,
                smoothing_backend,
                context,
                np.arange(start, stop, dtype=np.int64),
            )
        )
    return finalize_spatial_scores(pd.concat(frames, ignore_index=True), bandwidths)


def assign_prevalence_strata(scores: pd.DataFrame, n_strata: int) -> pd.DataFrame:
    """Assign deterministic, approximately equal-sized prevalence strata."""

    if int(n_strata) < 1:
        raise ValueError("n_strata must be positive")
    if scores.empty:
        raise ValueError("Cannot stratify an empty score table")
    required = {"prevalence", "candidate_position"}
    missing = required.difference(scores.columns)
    if missing:
        raise KeyError(f"Score table lacks prevalence-stratification columns: {sorted(missing)}")

    actual_strata = min(int(n_strata), scores.shape[0])
    tie_columns = ["prevalence"]
    if "gene" in scores.columns:
        tie_columns.append("gene")
    tie_columns.append("candidate_position")
    ordered = scores.sort_values(tie_columns, kind="mergesort")
    labels = np.floor(
        np.arange(scores.shape[0], dtype=np.float64) * actual_strata / scores.shape[0]
    ).astype(np.int16)
    output = scores.copy()
    output["prevalence_stratum"] = pd.Series(labels, index=ordered.index).loc[output.index].to_numpy()
    return output


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write a JSON checkpoint through an exclusive sibling staging file."""

    staging = path.with_name(path.name + ".partial")
    if path.exists() or staging.exists():
        raise FileExistsError(f"Refusing existing checkpoint output: {path} or {staging}")
    staging.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, path)


def _atomic_write_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    """Write a gzip CSV checkpoint atomically without relying on suffix inference."""

    staging = path.with_name(path.name + ".partial")
    if path.exists() or staging.exists():
        raise FileExistsError(f"Refusing existing checkpoint output: {path} or {staging}")
    frame.to_csv(staging, index=False, compression="gzip")
    os.replace(staging, path)


def generate_permutation_null_scores(
    candidate_matrix,
    scored_candidates: pd.DataFrame,
    coords: np.ndarray,
    library_totals: np.ndarray,
    bandwidths: list[float],
    chunk_size: int,
    smoothing_backend: str,
    fft_workers: int,
    null_profiles_per_stratum: int,
    seed: int,
    context: SpatialScoringContext | None = None,
    checkpoint_dir: Path | None = None,
    resume_checkpoints: bool = False,
) -> pd.DataFrame:
    """Score prevalence-matched profiles after randomizing their spatial locations.

    A source gene is sampled with replacement within each prevalence stratum.
    Every output profile is a permutation of the source values.  To avoid
    materializing one ``n_spots`` permutation vector per feature, a uniformly
    random base permutation is drawn per scoring block and independently
    cyclically shifted for each column.  Each marginal permutation remains
    uniform while memory stays bounded by the feature chunk size.
    """

    required = {"candidate_position", "gene", "prevalence", "prevalence_stratum"}
    missing = required.difference(scored_candidates.columns)
    if missing:
        raise KeyError(f"Scored candidates lack null-calibration columns: {sorted(missing)}")
    if int(null_profiles_per_stratum) < 1:
        raise ValueError("null_profiles_per_stratum must be positive")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    if context is None:
        context = build_spatial_scoring_context(
            coords,
            library_totals,
            bandwidths,
            smoothing_backend,
            fft_workers,
        )

    if resume_checkpoints and checkpoint_dir is None:
        raise ValueError("resume_checkpoints requires checkpoint_dir")
    rng = np.random.default_rng(int(seed))
    n_spots = int(candidate_matrix.shape[0])
    lookup = scored_candidates.set_index("candidate_position", drop=False)
    frames: list[pd.DataFrame] = []
    null_profile_id = 0
    strata = sorted(scored_candidates["prevalence_stratum"].astype(int).unique().tolist())
    checkpoint_manifest = {
        "format_version": 1,
        "seed": int(seed),
        "n_spots": n_spots,
        "n_candidate_genes": int(candidate_matrix.shape[1]),
        "chunk_size": int(chunk_size),
        "null_profiles_per_stratum": int(null_profiles_per_stratum),
        "strata": strata,
        "bandwidths_um": list(map(float, bandwidths)),
        "smoothing_backend": str(smoothing_backend),
        "coordinates_sha256": hashlib.sha256(
            np.ascontiguousarray(coords, dtype=np.float32).tobytes()
        ).hexdigest(),
        "candidate_signature_sha256": hashlib.sha256(
            np.ascontiguousarray(
                scored_candidates["candidate_position"].to_numpy(dtype=np.int64)
            ).tobytes()
            + "\0".join(scored_candidates["gene"].astype(str)).encode("utf-8")
        ).hexdigest(),
    }
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = checkpoint_dir / "null_checkpoint_manifest.json"
        if manifest_path.exists():
            if not resume_checkpoints:
                raise FileExistsError(
                    f"Null checkpoint manifest already exists; pass --resume-null-checkpoints: {manifest_path}"
                )
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing_manifest != checkpoint_manifest:
                raise ValueError("Null checkpoint manifest does not match the current run configuration")
        else:
            if resume_checkpoints and any(checkpoint_dir.iterdir()):
                raise RuntimeError("Checkpoint directory is nonempty but lacks its manifest")
            _atomic_write_json(manifest_path, checkpoint_manifest)

    for stratum in strata:
        source_rows = scored_candidates.loc[
            scored_candidates["prevalence_stratum"].astype(int) == int(stratum)
        ].sort_values(["candidate_position"], kind="mergesort")
        source_positions = source_rows["candidate_position"].to_numpy(dtype=np.int64)
        if not source_positions.size:
            raise RuntimeError(f"Prevalence stratum {stratum} has no source genes")
        plan_path = None
        if checkpoint_dir is not None:
            plan_path = checkpoint_dir / f"stratum_{int(stratum):03d}_plan.json"
        if plan_path is not None and plan_path.exists():
            if not resume_checkpoints:
                raise FileExistsError(f"Null checkpoint plan already exists: {plan_path}")
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            sampled_positions = np.asarray(plan["sampled_positions"], dtype=np.int64)
            if sampled_positions.shape != (int(null_profiles_per_stratum),):
                raise ValueError(f"Invalid sampled-position checkpoint: {plan_path}")
            rng.bit_generator.state = plan["rng_state_after_sampling"]
        else:
            sampled_positions = rng.choice(
                source_positions,
                size=int(null_profiles_per_stratum),
                replace=True,
            ).astype(np.int64, copy=False)
            if plan_path is not None:
                _atomic_write_json(
                    plan_path,
                    {
                        "prevalence_stratum": int(stratum),
                        "sampled_positions": sampled_positions.tolist(),
                        "rng_state_after_sampling": rng.bit_generator.state,
                    },
                )
        for start in range(0, sampled_positions.size, int(chunk_size)):
            stop = min(start + int(chunk_size), sampled_positions.size)
            chosen = sampled_positions[start:stop]
            chunk_index = start // int(chunk_size)
            chunk_frame_path = None
            chunk_state_path = None
            chunk_marker_path = None
            if checkpoint_dir is not None:
                prefix = checkpoint_dir / (
                    f"stratum_{int(stratum):03d}_chunk_{int(chunk_index):05d}"
                )
                chunk_frame_path = prefix.with_suffix(".csv.gz")
                chunk_state_path = prefix.with_suffix(".rng.json")
                chunk_marker_path = prefix.with_suffix(".complete")
                if chunk_marker_path.exists():
                    if not resume_checkpoints:
                        raise FileExistsError(f"Null checkpoint chunk already exists: {chunk_marker_path}")
                    if not chunk_frame_path.is_file() or not chunk_state_path.is_file():
                        raise RuntimeError(f"Incomplete checkpoint behind marker: {chunk_marker_path}")
                    frame = pd.read_csv(chunk_frame_path)
                    if frame.shape[0] != chosen.size:
                        raise ValueError(f"Checkpoint row count mismatch: {chunk_frame_path}")
                    rng.bit_generator.state = json.loads(
                        chunk_state_path.read_text(encoding="utf-8")
                    )
                    frames.append(frame)
                    null_profile_id += chosen.size
                    print(
                        f"Resumed null profiles for prevalence stratum {stratum}: "
                        f"{start + 1:,}-{stop:,} of {sampled_positions.size:,}",
                        flush=True,
                    )
                    continue
                partials = [
                    path
                    for path in (chunk_frame_path, chunk_state_path)
                    if path.exists()
                ]
                if partials:
                    raise RuntimeError(
                        "Incomplete null checkpoint without completion marker: "
                        + ", ".join(map(str, partials))
                    )
            print(
                f"Scoring null profiles for prevalence stratum {stratum}: "
                f"{start + 1:,}-{stop:,} of {sampled_positions.size:,}",
                flush=True,
            )
            source_block = candidate_matrix[:, chosen].toarray().astype(np.float32, copy=False)
            base_permutation = rng.permutation(n_spots)
            offsets = rng.integers(0, max(n_spots, 1), size=source_block.shape[1])
            permuted = np.empty_like(source_block)
            for column, offset in enumerate(offsets):
                row_order = np.roll(base_permutation, int(offset))
                permuted[:, column] = source_block[row_order, column]

            profile_ids = np.arange(
                null_profile_id,
                null_profile_id + chosen.size,
                dtype=np.int64,
            )
            null_profile_id += chosen.size
            frame = score_dense_feature_block(
                permuted,
                coords,
                bandwidths,
                smoothing_backend,
                context,
                profile_ids,
            )
            frame = finalize_spatial_scores(frame, bandwidths)
            source_metadata = lookup.loc[chosen]
            frame = frame.rename(columns={"candidate_position": "null_profile_id"})
            frame.insert(1, "prevalence_stratum", int(stratum))
            frame.insert(2, "null_index_within_stratum", np.arange(start, stop, dtype=np.int64))
            frame.insert(3, "source_candidate_position", chosen)
            frame.insert(4, "source_gene", source_metadata["gene"].astype(str).to_numpy())
            frame.insert(5, "source_prevalence", source_metadata["prevalence"].to_numpy(dtype=float))
            if chunk_frame_path is not None:
                _atomic_write_csv_gzip(chunk_frame_path, frame)
                assert chunk_state_path is not None and chunk_marker_path is not None
                _atomic_write_json(chunk_state_path, rng.bit_generator.state)
                chunk_marker_path.touch(exist_ok=False)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return deterministic Benjamini-Hochberg adjusted p-values."""

    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("p_values must be a finite one-dimensional array")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("p_values must lie in [0, 1]")
    if not values.size:
        return values.copy()
    order = np.argsort(values, kind="mergesort")
    ranks = np.arange(1, values.size + 1, dtype=np.float64)
    ranked_q = values[order] * values.size / ranks
    ranked_q = np.minimum.accumulate(ranked_q[::-1])[::-1]
    output = np.empty_like(ranked_q)
    output[order] = np.clip(ranked_q, 0.0, 1.0)
    return output


def calibrate_empirical_pvalues(
    scores: pd.DataFrame,
    null_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate stratum-specific empirical p-values and global BH q-values."""

    required_observed = {"prevalence_stratum", "spatial_contrast_score"}
    required_null = {"prevalence_stratum", "spatial_contrast_score"}
    if missing := required_observed.difference(scores.columns):
        raise KeyError(f"Observed scores lack calibration columns: {sorted(missing)}")
    if missing := required_null.difference(null_scores.columns):
        raise KeyError(f"Null scores lack calibration columns: {sorted(missing)}")

    output = scores.copy()
    empirical_p = np.ones(output.shape[0], dtype=np.float64)
    empirical_null_n = np.zeros(output.shape[0], dtype=np.int64)
    for stratum in sorted(output["prevalence_stratum"].astype(int).unique().tolist()):
        observed_mask = output["prevalence_stratum"].astype(int).to_numpy() == int(stratum)
        null_values = null_scores.loc[
            null_scores["prevalence_stratum"].astype(int) == int(stratum),
            "spatial_contrast_score",
        ].to_numpy(dtype=np.float64)
        if not null_values.size or not np.all(np.isfinite(null_values)):
            raise RuntimeError(f"Prevalence stratum {stratum} has no finite null scores")
        # Pandas copy-on-write and Arrow-backed columns can expose read-only
        # views. Sort a copy; preserve the caller's null table as well.
        null_values = np.sort(null_values)
        observed_values = output.loc[observed_mask, "spatial_contrast_score"].to_numpy(dtype=np.float64)
        # A +1 correction prevents zero p-values under a finite permutation null.
        n_at_least_observed = null_values.size - np.searchsorted(
            null_values,
            observed_values,
            side="left",
        )
        empirical_p[observed_mask] = (n_at_least_observed + 1.0) / (null_values.size + 1.0)
        empirical_null_n[observed_mask] = null_values.size
    output["empirical_p_value"] = empirical_p
    output["empirical_q_value"] = benjamini_hochberg(empirical_p)
    output["empirical_null_n"] = empirical_null_n
    return output


def select_empirical_fdr(
    scores: pd.DataFrame,
    q_threshold: float,
    max_artifact_like_score: float,
) -> pd.DataFrame:
    """Select every gene passing FDR and artifact gates without filling or truncating."""

    if not 0.0 < float(q_threshold) <= 1.0:
        raise ValueError("q_threshold must lie in (0, 1]")
    if not 0.0 <= float(max_artifact_like_score) <= 1.0:
        raise ValueError("max_artifact_like_score must lie in [0, 1]")
    required = {"empirical_q_value", "artifact_like_score", "spatial_contrast_score", "gene"}
    if missing := required.difference(scores.columns):
        raise KeyError(f"Score table lacks empirical-FDR columns: {sorted(missing)}")
    selected = scores.loc[
        (scores["empirical_q_value"] <= float(q_threshold))
        & (scores["artifact_like_score"] <= float(max_artifact_like_score))
    ].copy()
    selected = selected.sort_values(
        ["empirical_q_value", "spatial_contrast_score", "gene"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    selected.insert(0, "selection_rank", np.arange(1, selected.shape[0] + 1))
    selected.insert(1, "selection_reason", "empirical_fdr_spatial_contrast")
    return selected


def build_empirical_fdr_manifest(
    scores: pd.DataFrame,
    null_scores: pd.DataFrame,
    *,
    requested_strata: int,
    null_profiles_per_stratum: int,
    seed: int,
    q_threshold: float,
    max_artifact_like_score: float,
) -> dict:
    strata_summary: list[dict] = []
    for stratum in sorted(scores["prevalence_stratum"].astype(int).unique().tolist()):
        observed = scores.loc[scores["prevalence_stratum"].astype(int) == int(stratum)]
        null = null_scores.loc[null_scores["prevalence_stratum"].astype(int) == int(stratum)]
        null_values = null["spatial_contrast_score"].to_numpy(dtype=np.float64)
        strata_summary.append(
            {
                "prevalence_stratum": int(stratum),
                "n_observed_genes": int(observed.shape[0]),
                "observed_prevalence_min": float(observed["prevalence"].min()),
                "observed_prevalence_max": float(observed["prevalence"].max()),
                "n_null_profiles": int(null.shape[0]),
                "null_score_quantiles": {
                    str(quantile): float(np.quantile(null_values, quantile))
                    for quantile in (0.5, 0.9, 0.95, 0.99, 0.999)
                },
            }
        )
    minimum_p = 1.0 / (int(null_profiles_per_stratum) + 1.0)
    minimum_bh_rank = int(np.ceil(minimum_p * scores.shape[0] / float(q_threshold)))
    return {
        "method": "prevalence-stratified spatial permutation empirical FDR",
        "permutation_scheme": "source genes sampled within stratum; uniform base permutation per chunk with per-profile cyclic offsets",
        "requested_prevalence_strata": int(requested_strata),
        "actual_prevalence_strata": int(len(strata_summary)),
        "null_profiles_per_stratum": int(null_profiles_per_stratum),
        "total_null_profiles": int(null_scores.shape[0]),
        "permutation_seed": int(seed),
        "empirical_p_value_correction": "(1 + null scores >= observed) / (1 + n_null)",
        "multiple_testing": "global Benjamini-Hochberg",
        "q_threshold": float(q_threshold),
        "max_artifact_like_score": float(max_artifact_like_score),
        "calibration_resolution": {
            "empirical_p_value_step": float(minimum_p),
            "minimum_attainable_empirical_p_value": float(minimum_p),
            "number_of_global_hypotheses": int(scores.shape[0]),
            "minimum_bh_rank_at_which_minimum_p_can_pass_q_threshold": minimum_bh_rank,
            "interpretation": (
                "With finite permutation calibration, multiple genes may need to share the "
                "smallest attainable p-value before global BH can reject them."
            ),
        },
        "strata": strata_summary,
    }


def build_empirical_fdr_work_audit(
    *,
    n_spots: int,
    n_candidate_genes: int,
    n_bandwidths: int,
    actual_strata: int,
    null_profiles_per_stratum: int,
    chunk_size: int,
    smoothing_backend: str,
    transform: str,
    null_checkpointing_enabled: bool = False,
) -> dict:
    """Describe logical calibration work and peak dense-block memory."""

    n_null_profiles = int(actual_strata) * int(null_profiles_per_stratum)
    observed_chunks = int(np.ceil(n_candidate_genes / max(int(chunk_size), 1)))
    null_chunks = int(actual_strata) * int(
        np.ceil(null_profiles_per_stratum / max(int(chunk_size), 1))
    )
    dense_chunk_features = min(int(chunk_size), max(n_candidate_genes, n_null_profiles, 1))
    dense_chunk_bytes = int(n_spots) * dense_chunk_features * np.dtype(np.float32).itemsize
    return {
        "n_spots": int(n_spots),
        "n_candidate_genes": int(n_candidate_genes),
        "n_bandwidths": int(n_bandwidths),
        "actual_prevalence_strata": int(actual_strata),
        "null_profiles_per_stratum": int(null_profiles_per_stratum),
        "total_null_profiles": n_null_profiles,
        "observed_feature_chunks": observed_chunks,
        "null_feature_chunks": null_chunks,
        "logical_profile_scale_smoothing_evaluations": int(
            (int(n_candidate_genes) + n_null_profiles) * int(n_bandwidths)
        ),
        "approximate_single_dense_float32_block_bytes": dense_chunk_bytes,
        "approximate_single_dense_float32_block_gib": float(dense_chunk_bytes / 2**30),
        "smoothing_backend": smoothing_backend,
        "normalization": {
            "transform": transform,
            "strategy": (
                "transform selected eligible columns once using complete-library row totals; "
                "reuse the transformed matrix for observed scoring, null calibration, and previews"
            ),
        },
        "checkpointing": {
            "work_estimate_written_before_null_scoring": True,
            "final_null_scores_written_after_calibration": True,
            "null_chunk_resume_supported": bool(null_checkpointing_enabled),
            "note": (
                "Completed null chunks and exact legacy RNG states are atomically resumable."
                if null_checkpointing_enabled
                else "Without --null-checkpoint-dir, a failed null stage restarts from seed 0."
            ),
        },
    }


def rank_contrast_by_scale(
    scores: pd.DataFrame,
    bandwidths: list[float],
) -> pd.DataFrame:
    """Return complete adjusted-contrast rankings at every bandwidth."""

    frames: list[pd.DataFrame] = []
    penalty = (
        (1.0 - scores["artifact_like_score"].to_numpy(dtype=np.float64))
        * np.clip(np.sqrt(scores["prevalence"].to_numpy(dtype=np.float64) / 0.01), 0.0, 1.0)
    )
    for bandwidth in bandwidths:
        column = f"scale_contrast_{int(round(bandwidth))}um"
        ranked = scores.copy()
        ranked["adjusted_contrast_score"] = ranked[column].to_numpy(dtype=np.float64) * penalty
        ranked = ranked.sort_values(
            ["adjusted_contrast_score", column, "gene"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        ranked.insert(0, "contrast_bandwidth_um", float(bandwidth))
        ranked.insert(1, "contrast_rank", np.arange(1, ranked.shape[0] + 1))
        ranked["contrast_score"] = ranked[column]
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def top_contrast_by_scale(
    scores: pd.DataFrame,
    bandwidths: list[float],
    top_per_scale: int,
) -> pd.DataFrame:
    complete = rank_contrast_by_scale(scores, bandwidths)
    return truncate_scale_rankings(complete, top_per_scale)


def truncate_scale_rankings(complete_rankings: pd.DataFrame, top_per_scale: int) -> pd.DataFrame:
    if int(top_per_scale) < 1:
        raise ValueError("top_per_scale must be positive")
    return complete_rankings.loc[
        complete_rankings["contrast_rank"] <= int(top_per_scale)
    ].copy().reset_index(drop=True)


def fdr_selected_preview_rows(selected: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Choose preview rows exclusively from the empirical-FDR selected table."""

    if int(top_n) < 1 or selected.empty:
        return selected.head(0).copy()
    required = {"gene", "candidate_position", "empirical_q_value", "spatial_contrast_score"}
    if missing := required.difference(selected.columns):
        raise KeyError(f"FDR-selected table lacks preview columns: {sorted(missing)}")
    ranked = selected.sort_values(
        ["empirical_q_value", "spatial_contrast_score", "gene"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    return ranked.head(int(top_n)).copy().reset_index(drop=True)


def select_scale_balanced_panel(
    scores: pd.DataFrame,
    top_by_scale: pd.DataFrame,
    bandwidths: list[float],
    panel_size: int,
) -> pd.DataFrame:
    target = min(int(panel_size), scores.shape[0])
    ordered_per_scale = {
        float(bandwidth): top_by_scale.loc[
            top_by_scale["contrast_bandwidth_um"] == float(bandwidth)
        ].sort_values("contrast_rank")
        for bandwidth in bandwidths
    }
    selected: list[str] = []
    provenance: dict[str, tuple[float, int]] = {}
    seen: set[str] = set()
    max_rank = max((frame.shape[0] for frame in ordered_per_scale.values()), default=0)
    for rank_index in range(max_rank):
        for bandwidth in bandwidths:
            frame = ordered_per_scale[float(bandwidth)]
            if rank_index >= frame.shape[0]:
                continue
            row = frame.iloc[rank_index]
            gene = str(row["gene"])
            if gene in seen:
                continue
            seen.add(gene)
            selected.append(gene)
            provenance[gene] = (float(bandwidth), int(row["contrast_rank"]))
            if len(selected) >= target:
                break
        if len(selected) >= target:
            break

    if len(selected) < target:
        overall = scores.sort_values(
            ["spatial_contrast_score", "max_spatial_contrast", "gene"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        for _, row in overall.iterrows():
            gene = str(row["gene"])
            if gene in seen:
                continue
            seen.add(gene)
            selected.append(gene)
            provenance[gene] = (float(row["best_contrast_bandwidth_um"]), -1)
            if len(selected) >= target:
                break

    lookup = scores.set_index("gene")
    panel = lookup.loc[selected].reset_index()
    panel.insert(0, "selection_rank", np.arange(1, panel.shape[0] + 1))
    panel.insert(1, "selection_reason", "scale_balanced_spatial_contrast")
    panel["selection_bandwidth_um"] = [provenance[gene][0] for gene in selected]
    panel["selection_within_scale_rank"] = [provenance[gene][1] for gene in selected]
    return panel


def plot_gene_panel(
    selected: pd.DataFrame,
    candidate_matrix,
    coords: np.ndarray,
    geometry: GridGeometry | None,
    output_path: Path,
    title: str,
    clip_percentile: float,
    gamma: float,
    point_size: float,
) -> None:
    if selected.empty:
        return
    positions = selected["candidate_position"].to_numpy(dtype=np.int64)
    values = candidate_matrix[:, positions].toarray().astype(np.float32, copy=False)
    n_cols = min(4, selected.shape[0])
    n_rows = int(np.ceil(selected.shape[0] / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.2 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=15)
    for axis in axes.ravel():
        axis.set_visible(False)
    for index, (_, row) in enumerate(selected.iterrows()):
        ax = axes.ravel()[index]
        ax.set_visible(True)
        vector = values[:, index]
        positive = vector[vector > 0]
        vmax = float(np.percentile(positive, clip_percentile)) if positive.size else 1.0
        norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=max(vmax, 1.0e-6))
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad("white")
        if geometry is not None:
            canvas = np.full(geometry.mask.shape, np.nan, dtype=np.float32)
            canvas[geometry.y_index, geometry.x_index] = vector
            image = ax.imshow(canvas, origin="lower", cmap=cmap, norm=norm, interpolation="nearest")
        else:
            image = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=vector,
                s=point_size,
                cmap=cmap,
                norm=norm,
                linewidths=0,
                rasterized=True,
            )
            ax.set_aspect("equal", adjustable="box")
        score = row.get("adjusted_contrast_score", row.get("spatial_contrast_score", np.nan))
        ax.set_title(f"{row['gene']}\ncontrast={float(score):.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bandwidths = sorted({float(value) for value in args.bandwidths_um if float(value) > 0})
    if not bandwidths:
        raise ValueError("At least one positive bandwidth is required")
    if args.chunk_size_genes < 1 or args.top_per_scale < 1:
        raise ValueError("chunk size and top-per-scale must be positive")
    if args.selection_mode == "legacy-panel-size" and args.panel_size < 1:
        raise ValueError("panel size must be positive in legacy-panel-size mode")
    if args.selection_mode == "empirical-fdr":
        if args.n_prevalence_strata < 1:
            raise ValueError("n-prevalence-strata must be positive")
        if args.null_profiles_per_stratum < 2000:
            raise ValueError("empirical-fdr requires at least 2,000 null profiles per stratum")
        if not 0.0 < args.fdr_q <= 1.0:
            raise ValueError("fdr-q must lie in (0, 1]")
        if not 0.0 <= args.max_artifact_like_score <= 1.0:
            raise ValueError("max-artifact-like-score must lie in [0, 1]")
        if args.minimum_selected_genes < 1:
            raise ValueError("minimum-selected-genes must be positive")
        if args.resume_null_checkpoints and args.null_checkpoint_dir is None:
            raise ValueError("--resume-null-checkpoints requires --null-checkpoint-dir")

    adata = ad.read_h5ad(args.h5ad)
    try:
        adata.var_names_make_unique()
        coords_raw, resolved_coord_key = resolve_h5ad_coordinates(adata, args.coord_key)
        coords = np.asarray(coords_raw[:, :2], dtype=np.float32) * np.float32(args.coord_scale)
        counts = read_count_matrix(adata, args.layer)
        genes = adata.var_names.astype(str).to_numpy()
        library_totals = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
        stats = compute_gene_statistics(counts, genes)
        candidates = choose_candidates(stats, args.min_spots, args.n_candidate_genes)
        candidate_indices = candidates["feature_index"].to_numpy(dtype=np.int64)
        candidate_matrix = normalize_transform_selected_csr(
            counts,
            candidate_indices,
            library_totals,
            args.target_sum,
            args.transform,
        )

        geometry = build_grid_geometry(coords)
        grid_audit = coordinate_grid_audit(coords, geometry)
        grid_audit["irregular_scatter_explicitly_allowed"] = bool(args.allow_irregular_scatter)
        grid_audit["status"] = (
            "ready"
            if geometry is not None or args.allow_irregular_scatter
            else "failed_regular_grid_inference"
        )
        grid_audit_path = args.output_dir / "coordinate_grid_audit.json"
        grid_audit_path.write_text(
            json.dumps(json_safe(grid_audit), indent=2) + "\n",
            encoding="utf-8",
        )
        if geometry is None and not args.allow_irregular_scatter:
            raise ValueError(
                "Coordinates could not be represented as a regular grid; audit written to "
                f"{grid_audit_path}. Pass --allow-irregular-scatter only for intentionally "
                "irregular data."
            )
        smoothing_backend = resolve_smoothing_backend(
            args.smoothing_backend,
            geometry,
            bandwidths,
            counts.shape[0],
        )
        print(f"Using {smoothing_backend} smoothing", flush=True)
        scoring_context = build_spatial_scoring_context(
            coords,
            library_totals,
            bandwidths,
            smoothing_backend,
            args.fft_workers,
            geometry,
            args.cuda_device,
            args.gpu_reduction_accumulator,
        )

        scores = score_candidate_matrix(
            candidate_matrix,
            coords,
            library_totals,
            bandwidths,
            args.chunk_size_genes,
            smoothing_backend,
            args.fft_workers,
            scoring_context,
        )
        scores = candidates.merge(scores, on="candidate_position", how="inner", validate="one_to_one")
        scores = scores.sort_values("feature_index").reset_index(drop=True)
        null_scores_path: Path | None = None
        calibration_path: Path | None = None
        calibration_manifest: dict | None = None
        work_audit_path: Path | None = None
        work_audit: dict | None = None
        complete_by_scale: pd.DataFrame | None = None
        top_by_scale: pd.DataFrame | None = None
        if args.selection_mode == "empirical-fdr":
            scores = assign_prevalence_strata(scores, args.n_prevalence_strata)
            actual_strata = int(scores["prevalence_stratum"].nunique())
            work_audit = build_empirical_fdr_work_audit(
                n_spots=counts.shape[0],
                n_candidate_genes=scores.shape[0],
                n_bandwidths=len(bandwidths),
                actual_strata=actual_strata,
                null_profiles_per_stratum=args.null_profiles_per_stratum,
                chunk_size=args.chunk_size_genes,
                smoothing_backend=smoothing_backend,
                transform=args.transform,
                null_checkpointing_enabled=args.null_checkpoint_dir is not None,
            )
            work_audit_path = args.output_dir / "empirical_fdr_work_estimate.json"
            work_audit_path.write_text(
                json.dumps(json_safe(work_audit), indent=2) + "\n",
                encoding="utf-8",
            )
            null_scores = generate_permutation_null_scores(
                candidate_matrix,
                scores,
                coords,
                library_totals,
                bandwidths,
                args.chunk_size_genes,
                smoothing_backend,
                args.fft_workers,
                args.null_profiles_per_stratum,
                args.permutation_seed,
                scoring_context,
                args.null_checkpoint_dir,
                args.resume_null_checkpoints,
            )
            null_scores_path = args.output_dir / "empirical_fdr_null_scores.csv.gz"
            null_scores.to_csv(null_scores_path, index=False, compression="gzip")
            scores = calibrate_empirical_pvalues(scores, null_scores)
            panel = select_empirical_fdr(
                scores,
                args.fdr_q,
                args.max_artifact_like_score,
            )
            calibration_manifest = build_empirical_fdr_manifest(
                scores,
                null_scores,
                requested_strata=args.n_prevalence_strata,
                null_profiles_per_stratum=args.null_profiles_per_stratum,
                seed=args.permutation_seed,
                q_threshold=args.fdr_q,
                max_artifact_like_score=args.max_artifact_like_score,
            )
            calibration_path = args.output_dir / "empirical_fdr_calibration.json"
            calibration_path.write_text(
                json.dumps(json_safe(calibration_manifest), indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            # This mode intentionally preserves the former fixed-size behavior.
            complete_by_scale = rank_contrast_by_scale(scores, bandwidths)
            top_by_scale = truncate_scale_rankings(complete_by_scale, args.top_per_scale)
            panel = select_scale_balanced_panel(
                scores,
                top_by_scale,
                bandwidths,
                args.panel_size,
            )

        scores_path = args.output_dir / "multiscale_gene_contrast_scores.csv"
        scores.to_csv(scores_path, index=False)
        if complete_by_scale is None:
            complete_by_scale = rank_contrast_by_scale(scores, bandwidths)
        if top_by_scale is None:
            top_by_scale = truncate_scale_rankings(complete_by_scale, args.top_per_scale)
        complete_rankings_path = args.output_dir / "contrast_gene_rankings_all_scales.csv.gz"
        complete_by_scale.to_csv(complete_rankings_path, index=False, compression="gzip")
        top_path = args.output_dir / "top_contrast_genes_by_scale.csv"
        top_by_scale.to_csv(top_path, index=False)
        panel_path = args.output_dir / "selected_genes.csv"
        panel.to_csv(panel_path, index=False)

        downstream_ready = (
            args.selection_mode != "empirical-fdr"
            or panel.shape[0] >= int(args.minimum_selected_genes)
        )
        selection_status = {
            "status": "ready" if downstream_ready else "insufficient_selected_genes",
            "downstream_ready": bool(downstream_ready),
            "selection_mode": args.selection_mode,
            "n_selected_genes": int(panel.shape[0]),
            "minimum_selected_genes": (
                int(args.minimum_selected_genes)
                if args.selection_mode == "empirical-fdr"
                else None
            ),
            "exit_code": 0 if downstream_ready else 3,
            "selected_genes": str(panel_path),
        }
        status_path = args.output_dir / "selection_status.json"
        status_path.write_text(
            json.dumps(json_safe(selection_status), indent=2) + "\n",
            encoding="utf-8",
        )

        preview_paths: dict[str, str] = {}
        preview_genes_path: Path | None = None
        if args.plot_top_n > 0:
            if args.selection_mode == "empirical-fdr":
                preview_genes = fdr_selected_preview_rows(panel, args.plot_top_n)
                preview_genes_path = args.output_dir / "fdr_selected_preview_genes.csv"
                preview_genes.to_csv(preview_genes_path, index=False)
                path = args.output_dir / "top_fdr_selected_genes.png"
                plot_gene_panel(
                    preview_genes,
                    candidate_matrix,
                    coords,
                    geometry,
                    path,
                    "Top empirical-FDR selected spatial-contrast genes",
                    args.clip_percentile,
                    args.gamma,
                    args.point_size,
                )
                if not preview_genes.empty:
                    preview_paths["fdr_selected"] = str(path)
            else:
                for bandwidth in bandwidths:
                    selected = top_by_scale.loc[
                        top_by_scale["contrast_bandwidth_um"] == float(bandwidth)
                    ].sort_values("contrast_rank").head(args.plot_top_n)
                    path = args.output_dir / f"top_contrast_{int(round(bandwidth))}um.png"
                    plot_gene_panel(
                        selected,
                        candidate_matrix,
                        coords,
                        geometry,
                        path,
                        f"Top adjusted spatial-contrast genes at {bandwidth:g} µm",
                        args.clip_percentile,
                        args.gamma,
                        args.point_size,
                    )
                    preview_paths[f"{bandwidth:g}um"] = str(path)

        summary = {
            "method": "multiscale spatial contrast gene selection",
            "source_h5ad": str(args.h5ad),
            "layer": args.layer,
            "coord_key": resolved_coord_key,
            "coord_scale": float(args.coord_scale),
            "n_spots": int(counts.shape[0]),
            "n_genes": int(counts.shape[1]),
            "n_eligible_genes": int((stats["n_spots_expressed"] >= args.min_spots).sum()),
            "n_candidate_genes": int(candidates.shape[0]),
            "n_selected_genes": int(panel.shape[0]),
            "candidate_prefilter": (
                "none beyond min_spots eligibility"
                if args.n_candidate_genes <= 0
                else f"top {int(args.n_candidate_genes)} by raw-count dispersion"
            ),
            "selection_mode": args.selection_mode,
            "selection": (
                "all genes with empirical_q_value <= fdr_q and artifact_like_score <= maximum; no fill or truncation"
                if args.selection_mode == "empirical-fdr"
                else "legacy round-robin per-bandwidth ranks followed by overall-score fill"
            ),
            "selection_status": selection_status,
            "fdr_q": float(args.fdr_q) if args.selection_mode == "empirical-fdr" else None,
            "max_artifact_like_score": (
                float(args.max_artifact_like_score)
                if args.selection_mode == "empirical-fdr"
                else None
            ),
            "calibration": calibration_manifest,
            "null_checkpointing": {
                "directory": (
                    str(args.null_checkpoint_dir)
                    if args.null_checkpoint_dir is not None
                    else None
                ),
                "resume_requested": bool(args.resume_null_checkpoints),
                "chunk_completion_protocol": (
                    "atomic csv.gz plus atomic RNG-state JSON followed by completion marker"
                    if args.null_checkpoint_dir is not None
                    else None
                ),
            },
            "work_estimate": work_audit,
            "bandwidths_um": bandwidths,
            "min_spots": int(args.min_spots),
            "target_sum": float(args.target_sum),
            "transform": args.transform,
            "smoothing_backend": smoothing_backend,
            "fft_workers": int(args.fft_workers) if smoothing_backend == "grid-fft" else None,
            "cuda_fft": (
                scoring_context.cuda_fft_scorer.manifest()
                if scoring_context.cuda_fft_scorer is not None
                else None
            ),
            "grid_shape": None if geometry is None else list(map(int, geometry.mask.shape)),
            "grid_spacing": None if geometry is None else [geometry.spacing_y, geometry.spacing_x],
            "preview_render_mode": (
                "filled_grid_nearest" if geometry is not None else "irregular_scatter"
            ),
            "coordinate_grid_audit": grid_audit,
            "outputs": {
                "scores": str(scores_path),
                "complete_rankings_by_scale": str(complete_rankings_path),
                "top_by_scale": str(top_path),
                "selected_genes": str(panel_path),
                "selection_status": str(status_path),
                "empirical_fdr_null_scores": (
                    str(null_scores_path) if null_scores_path is not None else None
                ),
                "empirical_fdr_calibration": (
                    str(calibration_path) if calibration_path is not None else None
                ),
                "empirical_fdr_work_estimate": (
                    str(work_audit_path) if work_audit_path is not None else None
                ),
                "fdr_selected_preview_genes": (
                    str(preview_genes_path) if preview_genes_path is not None else None
                ),
                "coordinate_grid_audit": str(grid_audit_path),
                "previews": preview_paths,
            },
            "runtime_seconds": float(time.time() - started),
        }
        summary_path = args.output_dir / "multiscale_gene_contrast_summary.json"
        summary_path.write_text(json.dumps(json_safe(summary), indent=2) + "\n", encoding="utf-8")
        print(f"Scored {scores.shape[0]:,} candidate genes across {counts.shape[0]:,} spots")
        print(f"Selected {panel.shape[0]:,} genes using {args.selection_mode}: {panel_path}")
        print(f"Summary: {summary_path}")
        if not downstream_ready:
            print(
                f"Selection gate failed: {panel.shape[0]} genes passed, fewer than "
                f"minimum-selected-genes={args.minimum_selected_genes}",
                file=sys.stderr,
            )
            raise SystemExit(3)
    finally:
        del adata


if __name__ == "__main__":
    main()
