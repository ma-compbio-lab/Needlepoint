from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse, stats

from needlepoint.gpytorch_vski import _choose_device, _require_gpytorch
from needlepoint.svg import _as_dense_float, bh_fdr


ScreenBackend = Literal["auto", "exact_dense", "ski"]
GridIntervalReduction = Literal["max", "geometric_mean", "min"]
LengthScalePolicy = Literal["physical_multiscale_v1", "grid_aware_log2"]


def interior_lengthscale_initialization(
    values: np.ndarray, lower_bound: float, *, relative_margin: float = 0.05
) -> tuple[np.ndarray, dict]:
    """Move screened scales into a learnable interior of a softplus constraint.

    A nextafter-sized gap is mathematically valid but has an almost zero raw
    parameter gradient. This explicit initialization policy changes the start,
    not the kernel's scientific lower bound. The returned audit records it.
    """
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Initial length scales must be finite and positive")
    if not np.isfinite(lower_bound) or lower_bound <= 0:
        raise ValueError("lower_bound must be finite and positive")
    if not np.isfinite(relative_margin) or relative_margin <= 0:
        raise ValueError("relative_margin must be finite and positive")
    floor = float(lower_bound) * (1.0 + float(relative_margin))
    return np.maximum(values, floor).astype(np.float32), {
        "policy": "softplus-interior-v1",
        "relative_margin": float(relative_margin),
        "effective_initialization_floor": floor,
        "n_values_moved_to_interior": int(np.count_nonzero(values < floor)),
        "n_values_floored_above_global_minimum": int(
            np.count_nonzero(values <= lower_bound)
        ),
    }


def scale_loglik_column(scale: float) -> str:
    """Return the stable CSV column name for one candidate-scale likelihood."""
    token = np.format_float_positional(float(scale), trim="-").replace(".", "p")
    return f"scale_{token}_loglik"


def scale_fraction_column(scale: float) -> str:
    """Return the CSV column name for the best spatial fraction at one scale."""
    token = np.format_float_positional(float(scale), trim="-").replace(".", "p")
    return f"scale_{token}_spatial_fraction"


@dataclass
class InitialLengthScaleScreenSettings:
    """Settings for the pre-fit GP scale screen preceding V-SKI fitting."""

    # Explicit values remain supported for controlled comparisons. Otherwise
    # physical_multiscale_v1 is the default; grid_aware_log2 reproduces the
    # original inducing-grid-relative candidate construction exactly.
    length_scales: tuple[float, ...] | None = None
    auto_scale_start_grid_intervals: float = 1.0
    auto_scale_stop_grid_intervals: float = 16.0
    auto_scale_steps_per_octave: int = 4
    auto_scale_interval_reduction: GridIntervalReduction = "max"
    spatial_fractions: tuple[float, ...] = (
        1.0e-4,
        3.0e-4,
        1.0e-3,
        3.0e-3,
        1.0e-2,
        3.0e-2,
        0.1,
        0.2,
        0.4,
        0.6,
        0.8,
        0.95,
    )
    backend: ScreenBackend = "auto"
    exact_max_spots: int = 5_000
    feature_chunk_size: int = 16
    grid_size: int = 128
    max_cg_iterations: int = 1_000
    cg_tolerance: float = 1.0e-3
    max_preconditioner_size: int = 15
    jitter: float = 1.0e-5
    dtype: str = "float32"
    device: str | None = None
    random_state: int = 0
    candidate_scale_offset: int = 0
    length_scale_policy: LengthScalePolicy = "physical_multiscale_v1"
    sampling_interval_um: float | None = None
    physical_local_floor_um: float = 10.0
    physical_global_floor_um: float = 40.0
    physical_global_to_local_ratio: float = 4.0
    physical_upper_fov_fraction: float = 0.5
    physical_max_candidates: int = 8


def approximate_inducing_grid_intervals(
    coords: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    """Approximate x/y intervals of the requested interpolation grid.

    GPyTorch pads its interpolation grid slightly beyond the supplied bounds,
    so these are reference intervals rather than exact saved inducing axes.
    They are sufficient for avoiding global-kernel candidates far below the
    numerical resolution of the requested grid.
    """
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] < 2:
        raise ValueError("coords must have shape n_spots x 2 with at least two rows")
    if int(grid_size) <= 1:
        raise ValueError("grid_size must exceed one")
    spans = np.ptp(values, axis=0)
    if not np.isfinite(spans).all() or np.any(spans <= 0.0):
        raise ValueError("Both coordinate axes must have positive finite spans")
    return spans / float(int(grid_size) - 1)


def make_grid_aware_length_scales(
    coords: np.ndarray,
    *,
    grid_size: int,
    start_grid_intervals: float = 1.0,
    stop_grid_intervals: float = 16.0,
    steps_per_octave: int = 4,
    interval_reduction: GridIntervalReduction = "max",
) -> tuple[np.ndarray, dict]:
    """Build dense log-spaced global GP scales from an inducing-grid interval."""
    start = float(start_grid_intervals)
    stop = float(stop_grid_intervals)
    steps = int(steps_per_octave)
    if not np.isfinite([start, stop]).all() or start <= 0.0 or stop <= start:
        raise ValueError("Require 0 < start_grid_intervals < stop_grid_intervals")
    if steps <= 0:
        raise ValueError("steps_per_octave must be positive")
    intervals = approximate_inducing_grid_intervals(coords, grid_size)
    if interval_reduction == "max":
        reference = float(np.max(intervals))
    elif interval_reduction == "geometric_mean":
        reference = float(np.sqrt(np.prod(intervals)))
    elif interval_reduction == "min":
        reference = float(np.min(intervals))
    else:
        raise ValueError(
            "interval_reduction must be max, geometric_mean, or min"
        )

    octave_span = math.log2(stop / start)
    n_regular_steps = int(math.floor(octave_span * steps + 1.0e-12))
    multipliers = start * np.exp2(
        np.arange(n_regular_steps + 1, dtype=np.float64) / steps
    )
    if multipliers[-1] < stop * (1.0 - 1.0e-12):
        multipliers = np.append(multipliers, stop)
    else:
        multipliers[-1] = stop
    scales = reference * multipliers
    provenance = {
        "mode": "grid_aware_log2",
        "approximate_grid_intervals": intervals.tolist(),
        "interval_reduction": interval_reduction,
        "reference_grid_interval": reference,
        "start_grid_intervals": start,
        "stop_grid_intervals": stop,
        "steps_per_octave": steps,
        "multipliers": multipliers.tolist(),
        "length_scales": scales.tolist(),
        "grid_interval_note": (
            "Approximate from coordinate spans/(grid_size-1); GPyTorch pads "
            "the actual interpolation grid slightly beyond these bounds."
        ),
    }
    return scales, provenance


def estimate_sampling_interval_um(
    coords: np.ndarray, *, max_query_spots: int = 8192
) -> float:
    """Median positive nearest-neighbor spacing in physical micrometres.

    Build one spatial index on *all* available locations and query a bounded
    deterministic subset, rather than subsampling the reference locations:
    subsampling the reference would inflate the pitch on dense assays.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise ValueError("coords must have shape n_spots x 2 with at least two rows")
    if not np.isfinite(points).all():
        raise ValueError("coords must be finite for physical length-scale selection")
    if int(max_query_spots) < 1:
        raise ValueError("max_query_spots must be positive")
    count = min(points.shape[0], int(max_query_spots))
    indices = np.linspace(0, points.shape[0] - 1, count, dtype=np.int64)
    # Query more than one neighbor to handle occasional identical coordinates.
    distances, _ = cKDTree(points).query(
        points[indices], k=min(points.shape[0], 8), workers=1
    )
    distances = np.atleast_2d(distances)
    distances = np.where(distances > 0.0, distances, np.inf)
    nearest = np.min(distances, axis=1)
    usable = nearest[np.isfinite(nearest)]
    if usable.size == 0:
        raise ValueError("Cannot estimate sampling interval: no distinct coordinates")
    return float(np.median(usable))


def make_physical_multiscale_length_scales(
    coords: np.ndarray,
    *,
    grid_size: int,
    sampling_interval_um: float | None = None,
    local_floor_um: float = 10.0,
    global_floor_um: float = 40.0,
    global_to_local_ratio: float = 4.0,
    upper_fov_fraction: float = 0.5,
    max_candidates: int = 8,
) -> tuple[np.ndarray, dict]:
    """Coarse global scales independent of inducing-grid spacing.

    The local length scale is fixed separately during physical-policy fitting;
    only the global RBF length scale is screened. Grid spacing is audited, not
    silently used to redefine the desired biological length scales.
    """
    parameters = [local_floor_um, global_floor_um, global_to_local_ratio, upper_fov_fraction]
    if not np.isfinite(parameters).all() or min(parameters) <= 0.0:
        raise ValueError("Physical length-scale policy requires finite positive parameters")
    if int(max_candidates) < 2:
        raise ValueError("physical_max_candidates must be at least two")
    if sampling_interval_um is not None and (
        not np.isfinite(sampling_interval_um) or sampling_interval_um <= 0.0
    ):
        raise ValueError("sampling_interval_um must be finite and positive")
    points = np.asarray(coords, dtype=np.float64)
    intervals = approximate_inducing_grid_intervals(points, grid_size)
    sampling_pitch = (
        estimate_sampling_interval_um(points)
        if sampling_interval_um is None
        else float(sampling_interval_um)
    )
    local_scale = max(float(local_floor_um), sampling_pitch)
    lower = max(float(global_floor_um), float(global_to_local_ratio) * local_scale)
    fov_spans = np.ptp(points, axis=0)
    # Always offer two candidates, while flagging a field of view too small to
    # distinguish them in the returned audit.
    upper = max(2.0 * lower, float(upper_fov_fraction) * float(max(fov_spans)))
    n_candidates = min(
        int(max_candidates),
        max(2, int(math.ceil(math.log2(upper / lower))) + 1),
    )
    scales = np.geomspace(lower, upper, num=n_candidates, dtype=np.float64)
    reference_grid_spacing = float(np.max(intervals))
    provenance = {
        "mode": "physical_multiscale_v1",
        "sampling_interval_um": sampling_pitch,
        "sampling_interval_source": (
            "estimated_positive_nearest_neighbor" if sampling_interval_um is None else "user_supplied"
        ),
        "local_length_scale_um": local_scale,
        "global_min_length_scale_um": lower,
        "global_upper_length_scale_um": upper,
        "global_to_local_ratio": float(global_to_local_ratio),
        "physical_local_floor_um": float(local_floor_um),
        "physical_global_floor_um": float(global_floor_um),
        "upper_fov_fraction": float(upper_fov_fraction),
        "max_candidates": int(max_candidates),
        "coordinate_spans_um": fov_spans.tolist(),
        "approximate_grid_intervals_um": intervals.tolist(),
        "inducing_grid_underresolved_at_global_min": bool(
            reference_grid_spacing > lower / 2.0
        ),
        "fov_too_small_for_global_min": bool(float(max(fov_spans)) < 2.0 * lower),
        "length_scales": scales.tolist(),
    }
    return scales, provenance


def resolve_initial_length_scales(
    coords: np.ndarray,
    settings: InitialLengthScaleScreenSettings,
) -> tuple[np.ndarray, dict]:
    """Resolve explicit, physically informed, or legacy grid-aware scales."""
    if settings.length_scales is not None:
        scales = np.asarray(settings.length_scales, dtype=np.float64)
        return scales, {
            "mode": "explicit",
            "length_scales": scales.tolist(),
        }
    if settings.length_scale_policy == "physical_multiscale_v1":
        return make_physical_multiscale_length_scales(
            coords,
            grid_size=settings.grid_size,
            sampling_interval_um=settings.sampling_interval_um,
            local_floor_um=settings.physical_local_floor_um,
            global_floor_um=settings.physical_global_floor_um,
            global_to_local_ratio=settings.physical_global_to_local_ratio,
            upper_fov_fraction=settings.physical_upper_fov_fraction,
            max_candidates=settings.physical_max_candidates,
        )
    if settings.length_scale_policy == "grid_aware_log2":
        return make_grid_aware_length_scales(
            coords,
            grid_size=settings.grid_size,
            start_grid_intervals=settings.auto_scale_start_grid_intervals,
            stop_grid_intervals=settings.auto_scale_stop_grid_intervals,
            steps_per_octave=settings.auto_scale_steps_per_octave,
            interval_reduction=settings.auto_scale_interval_reduction,
        )
    raise ValueError(f"Unsupported length_scale_policy: {settings.length_scale_policy}")


def screen_initial_length_scales(
    expression: sparse.spmatrix | np.ndarray,
    coords: np.ndarray,
    feature_names: np.ndarray,
    settings: InitialLengthScaleScreenSettings | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Select a peak-specific RBF scale by approximate Gaussian likelihood.

    Every feature is standardized using only the supplied rows. For each scale,
    the best spatial fraction is found first; the best and runner-up *scales*
    are then retained. This makes the likelihood gap a scale-identification
    diagnostic rather than merely a comparison of adjacent fraction values.
    """
    torch, gpytorch = _require_gpytorch()
    settings = settings or InitialLengthScaleScreenSettings()
    coords_np = np.asarray(coords, dtype=np.float32)
    scales, scale_generation = resolve_initial_length_scales(coords_np, settings)
    _validate_screen_inputs(
        expression, coords_np, feature_names, settings, scales=scales
    )
    feature_names = np.asarray(feature_names).astype(str)
    n_spots, n_features = expression.shape
    fractions = np.asarray(settings.spatial_fractions, dtype=np.float64)
    device = _choose_device(torch, settings.device)
    torch.manual_seed(int(settings.random_state))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(settings.random_state))
    torch_dtype = torch.float64 if settings.dtype == "float64" else torch.float32
    coords_t = torch.as_tensor(coords_np, dtype=torch_dtype, device=device)
    backend = _choose_screen_backend(
        settings.backend, n_spots, settings.exact_max_spots
    )
    started = time.time()

    eigensystems = None
    ski_log_determinants = None
    if backend == "exact_dense":
        eigensystems = []
        for scale in scales:
            kernel = _dense_rbf_kernel(
                torch, gpytorch, coords_t, float(scale), settings.jitter
            )
            eigenvalues, eigenvectors = torch.linalg.eigh(kernel)
            eigensystems.append(
                (float(scale), eigenvalues.clamp_min(settings.jitter), eigenvectors)
            )
    else:
        # The covariance determinant is identical for every feature because the
        # screen standardizes each feature and uses a shared coordinate kernel.
        # Cache it once per (scale, spatial fraction), rather than repeating a
        # stochastic Lanczos log-determinant for every feature chunk.
        with torch.no_grad(), gpytorch.settings.max_cg_iterations(
            int(settings.max_cg_iterations)
        ), gpytorch.settings.cg_tolerance(
            float(settings.cg_tolerance)
        ), gpytorch.settings.max_preconditioner_size(
            int(settings.max_preconditioner_size)
        ):
            _advance_ski_probe_rng(
                torch,
                coords_t,
                n_candidate_pairs=(
                    int(settings.candidate_scale_offset) * int(fractions.size)
                ),
            )
            ski_log_determinants = _precompute_ski_log_determinants(
                torch,
                gpytorch,
                coords_t,
                scales,
                fractions,
                settings.grid_size,
                settings.jitter,
            )

    records: list[pd.DataFrame] = []
    for start in range(0, n_features, int(settings.feature_chunk_size)):
        stop = min(start + int(settings.feature_chunk_size), n_features)
        raw = _as_dense_float(expression[:, start:stop]).astype(
            np.float32, copy=False
        )
        standardized, means, stds = _standardize_columns(raw)
        y_t = torch.as_tensor(standardized, dtype=torch_dtype, device=device)
        null_ll = _null_loglik(torch, y_t)
        width = stop - start
        best_ll = torch.full((width,), -torch.inf, dtype=torch_dtype, device=device)
        second_ll = torch.full_like(best_ll, -torch.inf)
        best_scale = torch.zeros_like(best_ll)
        second_scale = torch.zeros_like(best_ll)
        best_fraction = torch.zeros_like(best_ll)
        scale_loglik_profile: dict[float, np.ndarray] = {}
        scale_fraction_profile: dict[float, np.ndarray] = {}

        scale_models = (
            eigensystems
            if eigensystems is not None
            else [(float(scale), None, None) for scale in scales]
        )
        # Screening evaluates fixed covariance candidates; no parameters are
        # optimized here. Disabling autograd prevents each CG solve from
        # retaining a graph across the scale/fraction grid on large slides.
        with torch.no_grad(), gpytorch.settings.max_cg_iterations(
            int(settings.max_cg_iterations)
        ), gpytorch.settings.cg_tolerance(
            float(settings.cg_tolerance)
        ), gpytorch.settings.max_preconditioner_size(
            int(settings.max_preconditioner_size)
        ):
            for scale, eigenvalues, eigenvectors in scale_models:
                scale_best_ll = torch.full_like(best_ll, -torch.inf)
                scale_best_fraction = torch.zeros_like(best_ll)
                if backend == "exact_dense":
                    rotated = eigenvectors.transpose(-2, -1) @ y_t
                else:
                    kernel_operator = _ski_rbf_operator(
                        torch, gpytorch, coords_t, scale, settings.grid_size
                    )
                for fraction in fractions:
                    if backend == "exact_dense":
                        ll = _eigen_loglik(
                            torch, rotated, eigenvalues, float(fraction)
                        )
                    else:
                        ll = _ski_loglik(
                            torch,
                            kernel_operator,
                            y_t,
                            float(fraction),
                            settings.jitter,
                            precomputed_log_determinant=ski_log_determinants[
                                (float(scale), float(fraction))
                            ],
                        )
                    improves_fraction = ll > scale_best_ll
                    scale_best_ll = torch.where(
                        improves_fraction, ll, scale_best_ll
                    )
                    scale_best_fraction = torch.where(
                        improves_fraction,
                        torch.as_tensor(
                            float(fraction), dtype=torch_dtype, device=device
                        ),
                        scale_best_fraction,
                    )

                scale_loglik_profile[float(scale)] = (
                    scale_best_ll.detach().cpu().numpy().astype(np.float64)
                )
                scale_fraction_profile[float(scale)] = (
                    scale_best_fraction.detach().cpu().numpy().astype(np.float64)
                )

                improves_best = scale_best_ll > best_ll
                improves_second = (~improves_best) & (scale_best_ll > second_ll)
                second_ll = torch.where(
                    improves_best,
                    best_ll,
                    torch.where(improves_second, scale_best_ll, second_ll),
                )
                second_scale = torch.where(
                    improves_best,
                    best_scale,
                    torch.where(
                        improves_second,
                        torch.as_tensor(scale, dtype=torch_dtype, device=device),
                        second_scale,
                    ),
                )
                best_ll = torch.where(improves_best, scale_best_ll, best_ll)
                best_scale = torch.where(
                    improves_best,
                    torch.as_tensor(scale, dtype=torch_dtype, device=device),
                    best_scale,
                )
                best_fraction = torch.where(
                    improves_best, scale_best_fraction, best_fraction
                )

        best_ll_np = best_ll.detach().cpu().numpy().astype(np.float64)
        second_ll_np = second_ll.detach().cpu().numpy().astype(np.float64)
        null_ll_np = null_ll.detach().cpu().numpy().astype(np.float64)
        best_scale_np = best_scale.detach().cpu().numpy().astype(np.float64)
        second_scale_np = second_scale.detach().cpu().numpy().astype(np.float64)
        fraction_np = best_fraction.detach().cpu().numpy().astype(np.float64)
        llr = np.clip(2.0 * (best_ll_np - null_ll_np), 0.0, None)
        chunk_values = {
            "gene": feature_names[start:stop],
            "feature_position": np.arange(start, stop, dtype=np.int64),
            "standardization_mean": means,
            "standardization_std": stds,
            "best_length_scale": best_scale_np,
            "second_best_length_scale": second_scale_np,
            "spatial_fraction": fraction_np,
            "spatial_loglik": best_ll_np,
            "second_best_scale_loglik": second_ll_np,
            "length_scale_loglik_gap": best_ll_np - second_ll_np,
            "null_loglik": null_ll_np,
            "LLR": llr,
            "pvalue": stats.chi2.sf(llr, df=1),
        }
        for scale in scales:
            chunk_values[scale_loglik_column(float(scale))] = scale_loglik_profile[
                float(scale)
            ]
            chunk_values[scale_fraction_column(float(scale))] = (
                scale_fraction_profile[float(scale)]
            )
        chunk = pd.DataFrame(chunk_values)
        records.append(chunk)

    table = pd.concat(records, ignore_index=True)
    table["qvalue"] = bh_fdr(table["pvalue"].to_numpy(dtype=np.float64))
    table["at_lower_scale_boundary"] = np.isclose(
        table["best_length_scale"], scales.min()
    )
    table["at_upper_scale_boundary"] = np.isclose(
        table["best_length_scale"], scales.max()
    )
    table["backend"] = backend
    table["device"] = str(device)
    table["n_screen_spots"] = int(n_spots)
    table["rank"] = (
        table["LLR"].rank(method="first", ascending=False).astype(np.int64)
    )
    table["elapsed_seconds"] = float(time.time() - started)
    table = table.sort_values("feature_position", kind="mergesort").reset_index(
        drop=True
    )

    spans = np.ptp(coords_np.astype(np.float64), axis=0)
    approximate_grid_spacing = spans / max(int(settings.grid_size) - 1, 1)
    summary = {
        "n_spots": int(n_spots),
        "n_features": int(n_features),
        "backend": backend,
        "device": str(device),
        "length_scales": scales.tolist(),
        "length_scale_generation": scale_generation,
        "spatial_fractions": fractions.tolist(),
        "settings": {
            key: value
            for key, value in vars(settings).items()
            if key not in {"length_scales", "spatial_fractions"}
        },
        "coordinate_bounds": [
            [float(coords_np[:, axis].min()), float(coords_np[:, axis].max())]
            for axis in range(2)
        ],
        "approximate_grid_spacing": approximate_grid_spacing.tolist(),
        "scale_loglik_columns": [
            scale_loglik_column(float(scale)) for scale in scales
        ],
        "n_ski_logdet_evaluations": (
            int(scales.size * fractions.size) if backend == "ski" else 0
        ),
        "ski_logdet_probe_strategy": (
            "serial_random_probe_stream_with_candidate_offset"
            if backend == "ski"
            else None
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    return table, summary


def merge_initial_length_scale_screen_tables(
    tables: list[pd.DataFrame],
    *,
    length_scales: np.ndarray | tuple[float, ...] | list[float],
    elapsed_seconds: float | None = None,
    device: str | None = None,
) -> pd.DataFrame:
    """Merge exhaustive candidate-scale shards into one screen result.

    Scale likelihoods are independent once expression standardization and the
    null model are fixed.  This permits exact candidate sharding across GPUs:
    each worker evaluates a disjoint scale subset, and this function restores
    the same best/runner-up selection contract as a single exhaustive scan.
    """
    if not tables:
        raise ValueError("At least one scale-screen table is required")
    scales = np.asarray(length_scales, dtype=np.float64)
    if scales.ndim != 1 or scales.size < 2 or not np.isfinite(scales).all():
        raise ValueError("length_scales must contain at least two finite values")
    if np.any(scales <= 0.0) or np.unique(scales).size != scales.size:
        raise ValueError("length_scales must be positive and unique")

    aligned: list[pd.DataFrame] = []
    reference_keys = None
    for table in tables:
        required = {
            "gene",
            "feature_position",
            "standardization_mean",
            "standardization_std",
            "null_loglik",
            "backend",
            "n_screen_spots",
        }
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"Scale-screen shard lacks columns: {sorted(missing)}")
        current = table.sort_values("feature_position", kind="stable").reset_index(drop=True)
        if current["feature_position"].duplicated().any() or current["gene"].duplicated().any():
            raise ValueError("Scale-screen shard contains duplicate features")
        keys = current[["feature_position", "gene"]]
        if reference_keys is None:
            reference_keys = keys
        elif not keys.equals(reference_keys):
            raise ValueError("Scale-screen shards have different feature order or identifiers")
        aligned.append(current)

    reference = aligned[0]
    invariant_numeric = [
        "standardization_mean",
        "standardization_std",
        "null_loglik",
        "n_screen_spots",
    ]
    for shard in aligned[1:]:
        for column in invariant_numeric:
            left = reference[column].to_numpy()
            right = shard[column].to_numpy()
            if not np.allclose(left, right, rtol=1.0e-6, atol=1.0e-7, equal_nan=True):
                raise ValueError(f"Scale-screen shards disagree in {column}")
        if not np.array_equal(reference["backend"].astype(str), shard["backend"].astype(str)):
            raise ValueError("Scale-screen shards use different backends")

    owners: dict[float, pd.DataFrame] = {}
    for scale in scales:
        loglik_column = scale_loglik_column(float(scale))
        fraction_column = scale_fraction_column(float(scale))
        matching = [
            shard
            for shard in aligned
            if loglik_column in shard.columns or fraction_column in shard.columns
        ]
        if len(matching) != 1:
            raise ValueError(
                f"Candidate scale {float(scale):g} must appear in exactly one shard; "
                f"found {len(matching)}"
            )
        owner = matching[0]
        if loglik_column not in owner or fraction_column not in owner:
            raise ValueError(f"Candidate scale {float(scale):g} has an incomplete profile")
        owners[float(scale)] = owner

    expected_columns = {
        scale_loglik_column(float(scale)) for scale in scales
    } | {
        scale_fraction_column(float(scale)) for scale in scales
    }
    observed_columns = {
        column
        for shard in aligned
        for column in shard.columns
        if column.startswith("scale_")
        and (column.endswith("_loglik") or column.endswith("_spatial_fraction"))
    }
    unexpected = observed_columns.difference(expected_columns)
    if unexpected:
        raise ValueError(f"Scale-screen shards contain unexpected profiles: {sorted(unexpected)}")

    loglik_profile = np.column_stack(
        [
            owners[float(scale)][scale_loglik_column(float(scale))].to_numpy(np.float64)
            for scale in scales
        ]
    )
    fraction_profile = np.column_stack(
        [
            owners[float(scale)][scale_fraction_column(float(scale))].to_numpy(np.float64)
            for scale in scales
        ]
    )
    if not np.isfinite(loglik_profile).all() or not np.isfinite(fraction_profile).all():
        raise ValueError("Scale-screen profiles contain non-finite values")
    order = np.argsort(-loglik_profile, axis=1, kind="stable")
    rows = np.arange(reference.shape[0])
    best_index = order[:, 0]
    second_index = order[:, 1]
    best_loglik = loglik_profile[rows, best_index]
    second_loglik = loglik_profile[rows, second_index]
    null_loglik = reference["null_loglik"].to_numpy(np.float64)
    llr = np.clip(2.0 * (best_loglik - null_loglik), 0.0, None)

    output = pd.DataFrame(
        {
            "gene": reference["gene"].astype(str),
            "feature_position": reference["feature_position"].to_numpy(np.int64),
            "standardization_mean": reference["standardization_mean"].to_numpy(np.float64),
            "standardization_std": reference["standardization_std"].to_numpy(np.float64),
            "best_length_scale": scales[best_index],
            "second_best_length_scale": scales[second_index],
            "spatial_fraction": fraction_profile[rows, best_index],
            "spatial_loglik": best_loglik,
            "second_best_scale_loglik": second_loglik,
            "length_scale_loglik_gap": best_loglik - second_loglik,
            "null_loglik": null_loglik,
            "LLR": llr,
            "pvalue": stats.chi2.sf(llr, df=1),
        }
    )
    for column in sorted(expected_columns):
        for shard in aligned:
            if column in shard:
                output[column] = shard[column].to_numpy()
                break
    output["qvalue"] = bh_fdr(output["pvalue"].to_numpy(np.float64))
    output["at_lower_scale_boundary"] = np.isclose(output["best_length_scale"], scales.min())
    output["at_upper_scale_boundary"] = np.isclose(output["best_length_scale"], scales.max())
    output["backend"] = reference["backend"].astype(str).to_numpy()
    output["device"] = str(device) if device is not None else "scale_sharded"
    output["n_screen_spots"] = reference["n_screen_spots"].to_numpy(np.int64)
    output["rank"] = output["LLR"].rank(method="first", ascending=False).astype(np.int64)
    output["elapsed_seconds"] = (
        float(elapsed_seconds)
        if elapsed_seconds is not None
        else max(float(shard["elapsed_seconds"].max()) for shard in aligned)
    )

    recomputed = set(output.columns) | expected_columns
    reserved = {
        "best_length_scale",
        "second_best_length_scale",
        "spatial_fraction",
        "spatial_loglik",
        "second_best_scale_loglik",
        "length_scale_loglik_gap",
        "LLR",
        "pvalue",
        "qvalue",
        "at_lower_scale_boundary",
        "at_upper_scale_boundary",
        "weak_length_scale_identification",
        "recommended_for_multistart",
        "rank",
        "device",
        "elapsed_seconds",
    }
    extra_columns = [
        column
        for column in reference.columns
        if column not in recomputed and column not in reserved
    ]
    for column in extra_columns:
        expected = reference[column].reset_index(drop=True)
        for shard in aligned[1:]:
            if column not in shard or not shard[column].reset_index(drop=True).equals(expected):
                raise ValueError(f"Scale-screen shards disagree in metadata column {column}")
        output[column] = expected
    return output.sort_values("feature_position", kind="stable").reset_index(drop=True)


def mark_uncertain_initial_scales(
    table: pd.DataFrame,
    *,
    minimum_loglik_gap: float = 2.0,
) -> pd.DataFrame:
    """Flag peaks whose discrete scale choice merits a multistart sensitivity fit."""
    result = table.copy()
    result["weak_length_scale_identification"] = (
        result["length_scale_loglik_gap"] < float(minimum_loglik_gap)
    )
    result["recommended_for_multistart"] = (
        result["at_lower_scale_boundary"]
        | result["at_upper_scale_boundary"]
        | result["weak_length_scale_identification"]
    )
    return result


def _validate_screen_inputs(
    expression,
    coords,
    feature_names,
    settings,
    *,
    scales: np.ndarray,
) -> None:
    if expression.ndim != 2:
        raise ValueError("expression must have shape n_spots x n_features")
    if np.asarray(coords).shape != (expression.shape[0], 2):
        raise ValueError("coords must have shape n_spots x 2")
    if np.asarray(feature_names).shape != (expression.shape[1],):
        raise ValueError("feature_names must have one value per feature")
    if scales.size < 2 or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("length_scales must contain at least two positive values")
    if np.unique(scales).size != scales.size:
        raise ValueError("length_scales must be unique")
    fractions = np.asarray(settings.spatial_fractions, dtype=np.float64)
    if fractions.size == 0 or np.any((fractions <= 0) | (fractions >= 1)):
        raise ValueError("spatial_fractions must lie strictly between zero and one")
    if int(settings.feature_chunk_size) <= 0 or int(settings.grid_size) <= 1:
        raise ValueError("feature_chunk_size and grid_size must be positive")
    if int(settings.max_preconditioner_size) < 0:
        raise ValueError("max_preconditioner_size must be nonnegative")
    if int(settings.candidate_scale_offset) < 0:
        raise ValueError("candidate_scale_offset must be nonnegative")
    if settings.dtype not in {"float32", "float64"}:
        raise ValueError("dtype must be float32 or float64")


def _standardize_columns(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    stds[stds < 1.0e-8] = 1.0
    standardized = ((matrix - means) / stds).astype(np.float32, copy=False)
    return standardized, means.astype(np.float32), stds.astype(np.float32)


def _choose_screen_backend(
    backend: ScreenBackend, n_spots: int, exact_max_spots: int
) -> Literal["exact_dense", "ski"]:
    if backend == "auto":
        return "exact_dense" if n_spots <= int(exact_max_spots) else "ski"
    if backend not in {"exact_dense", "ski"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if backend == "exact_dense" and n_spots > int(exact_max_spots):
        warnings.warn(
            "Exact dense scale screening exceeds exact_max_spots",
            RuntimeWarning,
            stacklevel=2,
        )
    return backend


def _null_loglik(torch, target):
    n_spots = target.shape[0]
    variance = target.square().mean(dim=0).clamp_min(1.0e-8)
    return -0.5 * (
        n_spots * (torch.log(2.0 * torch.pi * variance) + 1.0)
    )


def _dense_rbf_kernel(torch, gpytorch, coords, scale: float, jitter: float):
    kernel = gpytorch.kernels.RBFKernel().to(
        device=coords.device, dtype=coords.dtype
    )
    kernel.lengthscale = torch.as_tensor(
        scale, dtype=coords.dtype, device=coords.device
    )
    covariance = kernel(coords).to_dense()
    covariance = covariance / covariance.diagonal().mean().clamp_min(1.0e-8)
    return covariance + float(jitter) * torch.eye(
        covariance.shape[0], dtype=coords.dtype, device=coords.device
    )


def _eigen_loglik(torch, rotated, eigenvalues, fraction: float):
    n_spots = rotated.shape[0]
    diagonal = (
        float(fraction) * eigenvalues + (1.0 - float(fraction))
    ).clamp_min(1.0e-8)
    quadratic = (rotated.square() / diagonal[:, None]).sum(dim=0)
    log_determinant = diagonal.log().sum()
    return -0.5 * (
        quadratic + log_determinant + n_spots * math.log(2.0 * math.pi)
    )


def _ski_rbf_operator(torch, gpytorch, coords, scale: float, grid_size: int):
    base_kernel = gpytorch.kernels.RBFKernel().to(
        device=coords.device, dtype=coords.dtype
    )
    base_kernel.lengthscale = torch.as_tensor(
        scale, dtype=coords.dtype, device=coords.device
    )
    bounds = tuple(
        (float(coords[:, axis].min()), float(coords[:, axis].max()))
        for axis in range(coords.shape[1])
    )
    kernel = gpytorch.kernels.GridInterpolationKernel(
        base_kernel,
        grid_size=int(grid_size),
        num_dims=coords.shape[1],
        grid_bounds=bounds,
    ).to(device=coords.device, dtype=coords.dtype)
    return kernel(coords)


def _ski_covariance(torch, kernel, target, fraction: float, jitter: float):
    n_spots = target.shape[0]
    diagonal = torch.full(
        (n_spots,),
        1.0 - float(fraction) + float(jitter),
        dtype=target.dtype,
        device=target.device,
    )
    return kernel.mul(float(fraction)).add_diagonal(diagonal)


def _precompute_ski_log_determinants(
    torch,
    gpytorch,
    coords,
    scales,
    fractions,
    grid_size: int,
    jitter: float,
) -> dict[tuple[float, float], float]:
    result: dict[tuple[float, float], float] = {}
    for scale in scales:
        kernel = _ski_rbf_operator(
            torch, gpytorch, coords, float(scale), int(grid_size)
        )
        for fraction in fractions:
            covariance = _ski_covariance(
                torch, kernel, coords, float(fraction), float(jitter)
            )
            result[(float(scale), float(fraction))] = float(
                covariance.logdet().detach().cpu()
            )
    return result


def _advance_ski_probe_rng(torch, coords, *, n_candidate_pairs: int) -> None:
    """Advance to a contiguous shard's place in the serial SLQ probe stream.

    LinearOperator draws ``num_trace_samples x n_spots`` standard-normal
    probes once per covariance log-determinant. Drawing and discarding the
    preceding blocks is inexpensive relative to their CG solves and preserves
    the historical seeded serial screen exactly across scale shards.
    """
    if int(n_candidate_pairs) < 0:
        raise ValueError("n_candidate_pairs must be nonnegative")
    if int(n_candidate_pairs) == 0:
        return
    from linear_operator import settings as linear_operator_settings

    n_trace_samples = int(linear_operator_settings.num_trace_samples.value())
    for _ in range(int(n_candidate_pairs)):
        torch.randn(
            n_trace_samples,
            int(coords.shape[0]),
            dtype=coords.dtype,
            device=coords.device,
        )


def _ski_loglik(
    torch,
    kernel,
    target,
    fraction: float,
    jitter: float,
    *,
    precomputed_log_determinant: float | None = None,
):
    n_spots = target.shape[0]
    covariance = _ski_covariance(
        torch, kernel, target, float(fraction), float(jitter)
    )
    if precomputed_log_determinant is None:
        inverse_quadratic, log_determinant = covariance.inv_quad_logdet(
            inv_quad_rhs=target,
            logdet=True,
            reduce_inv_quad=False,
        )
    else:
        inverse_quadratic = covariance.inv_quad(
            target, reduce_inv_quad=False
        )
        log_determinant = torch.as_tensor(
            precomputed_log_determinant,
            dtype=target.dtype,
            device=target.device,
        )
    return -0.5 * (
        inverse_quadratic
        + log_determinant
        + n_spots * math.log(2.0 * math.pi)
    )
