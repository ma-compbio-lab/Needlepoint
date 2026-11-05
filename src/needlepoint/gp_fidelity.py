from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import ndtr
from scipy.stats import rankdata


@dataclass(frozen=True)
class SpatialHoldouts:
    """Fixed spot masks used to distinguish interpolation from reconstruction."""

    random: np.ndarray
    blocked: np.ndarray

    @property
    def combined(self) -> np.ndarray:
        return self.random | self.blocked


def scale_coordinates_by_spot_spacing(coords: np.ndarray, max_reference_spots: int = 200_000) -> tuple[np.ndarray, float]:
    """Express coordinates in units of median nearest-neighbor spacing."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape n_spots x 2")
    if coords.shape[0] < 2:
        raise ValueError("At least two coordinates are required")
    if coords.shape[0] > max_reference_spots:
        reference = np.linspace(0, coords.shape[0] - 1, max_reference_spots, dtype=np.int64)
        query = coords[reference]
    else:
        query = coords
    distances, _ = cKDTree(coords).query(query, k=2, workers=-1)
    spacing = float(np.median(distances[:, 1][distances[:, 1] > 0]))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Could not infer a positive nearest-neighbor spacing")
    center = np.median(coords, axis=0)
    return ((coords - center[None, :]) / spacing).astype(np.float32), spacing


def make_spatial_holdouts(
    coords: np.ndarray,
    *,
    random_fraction: float = 0.02,
    blocked_fraction: float = 0.02,
    block_size: int = 256,
    random_state: int = 0,
) -> SpatialHoldouts:
    """Create disjoint random and compact blocked interpolation holdouts."""
    coords = np.asarray(coords, dtype=np.float64)
    n_spots = coords.shape[0]
    if n_spots < 4:
        raise ValueError("At least four spots are required for holdouts")
    if not (0 <= random_fraction < 1 and 0 <= blocked_fraction < 1):
        raise ValueError("Holdout fractions must be in [0, 1)")
    if random_fraction + blocked_fraction >= 0.5:
        raise ValueError("Combined holdout fraction must be below 0.5")
    rng = np.random.default_rng(random_state)
    random_mask = np.zeros(n_spots, dtype=bool)
    blocked_mask = np.zeros(n_spots, dtype=bool)

    n_random = int(round(random_fraction * n_spots))
    if n_random:
        random_mask[rng.choice(n_spots, size=n_random, replace=False)] = True

    n_blocked = int(round(blocked_fraction * n_spots))
    if n_blocked:
        tree = cKDTree(coords)
        available = np.flatnonzero(~random_mask)
        n_centers = max(1, int(np.ceil(n_blocked / max(int(block_size), 1))))
        center_candidates = rng.choice(available, size=min(n_centers * 5, available.size), replace=False)
        for center in center_candidates:
            if int(blocked_mask.sum()) >= n_blocked:
                break
            remaining = n_blocked - int(blocked_mask.sum())
            query_k = min(max(8, min(int(block_size), remaining)), n_spots)
            _, neighbors = tree.query(coords[center], k=query_k)
            neighbors = np.atleast_1d(neighbors).astype(np.int64)
            neighbors = neighbors[~random_mask[neighbors] & ~blocked_mask[neighbors]]
            blocked_mask[neighbors[:remaining]] = True
        if int(blocked_mask.sum()) < n_blocked:
            remaining = np.flatnonzero(~random_mask & ~blocked_mask)
            blocked_mask[rng.choice(remaining, size=n_blocked - int(blocked_mask.sum()), replace=False)] = True
    return SpatialHoldouts(random=random_mask, blocked=blocked_mask)


def evaluate_gp_fidelity(
    observed: np.ndarray,
    posterior_mean: np.ndarray,
    predictive_variance: np.ndarray,
    coords: np.ndarray,
    feature_names: np.ndarray,
    *,
    evaluation_mask: np.ndarray | None = None,
    neighborhood_sizes: tuple[int, ...] = (1, 4, 16, 64, 256),
    hotspot_quantiles: tuple[float, ...] = (0.90, 0.95, 0.99),
    max_spatial_samples: int = 20_000,
    random_state: int = 0,
    grid_size: int | None = None,
    grid_axes: list[list[float]] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute transparent predictive and spatial reconstruction diagnostics."""
    observed = np.asarray(observed, dtype=np.float64)
    posterior_mean = np.asarray(posterior_mean, dtype=np.float64)
    predictive_variance = np.asarray(predictive_variance, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    feature_names = np.asarray(feature_names).astype(str)
    if observed.shape != posterior_mean.shape or observed.shape != predictive_variance.shape:
        raise ValueError("observed, posterior_mean, and predictive_variance must have identical shapes")
    if observed.ndim != 2 or observed.shape[0] != coords.shape[0]:
        raise ValueError("maps must have shape n_spots x n_features")
    if feature_names.size != observed.shape[1]:
        raise ValueError("feature_names must match the number of columns")
    if evaluation_mask is None:
        evaluation_mask = np.ones(observed.shape[0], dtype=bool)
    evaluation_mask = np.asarray(evaluation_mask, dtype=bool)
    if evaluation_mask.shape != (observed.shape[0],) or not evaluation_mask.any():
        raise ValueError("evaluation_mask must select at least one spot")

    eval_idx = np.flatnonzero(evaluation_mask)
    sample_idx, neighbor_idx = _spatial_neighbor_sample(
        coords,
        evaluation_mask,
        max(neighborhood_sizes),
        max_spatial_samples,
        random_state,
    )
    rows: list[dict] = []
    for feature_pos, feature_name in enumerate(feature_names):
        y = observed[:, feature_pos]
        mean = posterior_mean[:, feature_pos]
        variance = np.maximum(predictive_variance[:, feature_pos], 1.0e-8)
        y_eval = y[eval_idx]
        mean_eval = mean[eval_idx]
        var_eval = variance[eval_idx]
        residual = y_eval - mean_eval
        std = np.sqrt(var_eval)
        z = residual / std
        row = {
            "feature_position": int(feature_pos),
            "gene": feature_name,
            "n_evaluation_spots": int(eval_idx.size),
            "rmse": float(np.sqrt(np.mean(np.square(residual)))),
            "mae": float(np.mean(np.abs(residual))),
            "pearson": _safe_correlation(y_eval, mean_eval),
            "spearman": _safe_spearman(y_eval, mean_eval),
            "mean_negative_log_predictive_density": float(
                np.mean(0.5 * (np.log(2.0 * np.pi * var_eval) + np.square(residual) / var_eval))
            ),
            "mean_crps": float(np.mean(_gaussian_crps(y_eval, mean_eval, std))),
            "residual_moran_i": _sample_moran_i(
                y - mean,
                sample_idx,
                neighbor_idx[
                    :, 1 : min(9, neighbor_idx.shape[1])
                ] if neighbor_idx.shape[1] > 1 else neighbor_idx,
            ),
        }
        nearest = neighbor_idx[:, 1] if neighbor_idx.shape[1] > 1 else neighbor_idx[:, 0]
        observed_increment = y[sample_idx] - y[nearest]
        fitted_increment = mean[sample_idx] - mean[nearest]
        observed_roughness = float(np.mean(np.square(observed_increment)))
        fitted_roughness = float(np.mean(np.square(fitted_increment)))
        row["nearest_neighbor_increment_correlation"] = _safe_correlation(
            observed_increment, fitted_increment
        )
        row["nearest_neighbor_roughness_ratio"] = fitted_roughness / max(
            observed_roughness, 1.0e-12
        )
        row["nearest_neighbor_log_roughness_error"] = float(
            abs(np.log(max(row["nearest_neighbor_roughness_ratio"], 1.0e-12)))
        )
        for level in (0.50, 0.80, 0.90, 0.95):
            cutoff = _normal_interval_z(level)
            row[f"coverage_{int(level * 100)}"] = float(np.mean(np.abs(z) <= cutoff))
        for quantile in hotspot_quantiles:
            threshold_y = np.quantile(y_eval, quantile)
            threshold_mean = np.quantile(mean_eval, quantile)
            truth = y_eval >= threshold_y
            fitted = mean_eval >= threshold_mean
            intersection = int(np.count_nonzero(truth & fitted))
            row[f"hotspot_dice_q{int(quantile * 100)}"] = float(
                2.0 * intersection / max(int(truth.sum() + fitted.sum()), 1)
            )
            row[f"hotspot_jaccard_q{int(quantile * 100)}"] = float(
                intersection / max(int(np.count_nonzero(truth | fitted)), 1)
            )
        y_neighbors = y[neighbor_idx]
        mean_neighbors = mean[neighbor_idx]
        for size in neighborhood_sizes:
            k = min(int(size), neighbor_idx.shape[1])
            y_smooth = y_neighbors[:, :k].mean(axis=1)
            mean_smooth = mean_neighbors[:, :k].mean(axis=1)
            row[f"multiscale_correlation_k{k}"] = _safe_correlation(y_smooth, mean_smooth)
            row[f"contrast_retention_k{k}"] = float(
                np.var(mean_smooth) / max(float(np.var(y_smooth)), 1.0e-12)
            )
        if grid_axes is not None:
            row["grid_distance_error_correlation"] = _grid_distance_error_correlation(
                coords[eval_idx], residual, grid_axes=np.asarray(grid_axes)
            )
            row["grid_distance_source"] = "fitted_inducing_grid"
        elif grid_size is not None:
            row["grid_distance_error_correlation"] = _grid_distance_error_correlation(
                coords[eval_idx], residual, grid_size=int(grid_size)
            )
            row["grid_distance_source"] = "estimated_from_evaluation_extent"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_fidelity(table: pd.DataFrame) -> dict:
    numeric = table.select_dtypes(include=[np.number]).drop(columns=["feature_position"], errors="ignore")
    return {
        "n_features": int(table.shape[0]),
        "median": {column: float(value) for column, value in numeric.median().items()},
        "mean": {column: float(value) for column, value in numeric.mean().items()},
    }


def _spatial_neighbor_sample(
    coords: np.ndarray,
    evaluation_mask: np.ndarray,
    max_neighbors: int,
    max_samples: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    candidates = np.flatnonzero(evaluation_mask)
    if candidates.size > max_samples:
        candidates = np.sort(rng.choice(candidates, size=max_samples, replace=False))
    k = min(max(int(max_neighbors), 1), coords.shape[0])
    _, neighbors = cKDTree(coords).query(coords[candidates], k=k, workers=-1)
    if k == 1:
        neighbors = np.asarray(neighbors, dtype=np.int64)[:, None]
    return candidates, np.asarray(neighbors, dtype=np.int64)


def _gaussian_crps(observed: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = (observed - mean) / std
    density = np.exp(-0.5 * np.square(z)) / np.sqrt(2.0 * np.pi)
    return std * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * density - 1.0 / np.sqrt(np.pi))


def _normal_interval_z(level: float) -> float:
    values = {0.50: 0.67448975, 0.80: 1.28155157, 0.90: 1.64485363, 0.95: 1.95996398}
    return values[level]


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.sqrt(np.square(left).sum() * np.square(right).sum())
    return float(np.dot(left, right) / denominator) if denominator > 1.0e-12 else 0.0


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _safe_correlation(rankdata(left), rankdata(right))


def _sample_moran_i(residual: np.ndarray, sample_idx: np.ndarray, neighbors: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=np.float64)
    centered = residual - residual.mean()
    focal = centered[sample_idx]
    neighbor_values = centered[neighbors]
    numerator = float(np.sum(focal[:, None] * neighbor_values))
    denominator = float(np.sum(np.square(focal)))
    if denominator <= 1.0e-12:
        return 0.0
    return float(sample_idx.size / max(neighbors.size, 1) * numerator / denominator)


def _grid_distance_error_correlation(
    coords: np.ndarray,
    residual: np.ndarray,
    *,
    grid_axes: np.ndarray | None = None,
    grid_size: int | None = None,
) -> float:
    if grid_axes is not None:
        axes = np.asarray(grid_axes, dtype=np.float64)
        if axes.ndim != 2 or axes.shape[0] != coords.shape[1] or axes.shape[1] < 2:
            raise ValueError("grid_axes must have shape n_dimensions x grid_size")
        squared_distance = np.zeros(coords.shape[0], dtype=np.float64)
        for dimension, axis in enumerate(axes):
            axis = np.sort(axis)
            insertion = np.searchsorted(axis, coords[:, dimension])
            lower = axis[np.clip(insertion - 1, 0, axis.size - 1)]
            upper = axis[np.clip(insertion, 0, axis.size - 1)]
            nearest_distance = np.minimum(
                np.abs(coords[:, dimension] - lower),
                np.abs(coords[:, dimension] - upper),
            )
            squared_distance += np.square(nearest_distance)
        return _safe_correlation(np.sqrt(squared_distance), np.abs(residual))
    if grid_size is None or grid_size < 2:
        return 0.0
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    spacing = np.maximum((hi - lo) / (grid_size - 1), 1.0e-8)
    phase = (coords - lo[None, :]) / spacing[None, :]
    distance = np.sqrt(np.square(phase - np.rint(phase)).sum(axis=1))
    return _safe_correlation(distance, np.abs(residual))
