"""Reusable domain-by-peak statistics from streamed fitted TEMI maps.

This module deliberately keeps the statistic tensor independent from the
differential-testing workflow. For a standardized-Gaussian posterior mean
``m``, training center ``mu`` and scale ``sigma``, the input-scale fitted value
is ``x = m * sigma + mu``. The common-panel relative abundance for spot ``i``
and peak ``p`` is

``max(x[i, p], 0) / sum_q max(x[i, q], 0)``,

where the denominator spans the complete common feature panel.  Domain means
then use represented spot area times the saved soft-domain probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .domain_peak_differential import validate_joint_axes


AVERAGE_RELATIVE_ABUNDANCE = "average_relative_abundance"


@dataclass(frozen=True)
class DomainPeakStatisticsResult:
    """A labeled section x domain x peak x statistic tensor."""

    values: np.ndarray
    section_ids: np.ndarray
    domain_ids: np.ndarray
    peak_ids: np.ndarray
    statistic_ids: np.ndarray
    zero_denominator_counts: np.ndarray
    domain_weight_sums: np.ndarray
    metadata: dict[str, object]


def compute_average_relative_abundance_tensor(
    store,
    joint_spots: pd.DataFrame,
    domain_probabilities: np.ndarray,
    section_ids: Sequence[str],
    *,
    spot_areas: np.ndarray | Sequence[float] | None = None,
    spot_pitches: Sequence[float] | None = None,
    zero_denominator_tolerance: float = 0.0,
) -> DomainPeakStatisticsResult:
    """Compute common-panel average relative abundance in two streamed passes.

    The first pass inverse-transforms standardized-Gaussian posterior means,
    clips them at zero, and sums across all common peaks for every spot. The
    second pass normalizes each feature block by those totals and accumulates
    area- and soft-domain-weighted means. Spots whose common-panel total is at
    or below ``zero_denominator_tolerance`` are excluded from both numerator
    and domain weight denominator and are reported explicitly.

    Parameters
    ----------
    store
        A ``MultiDatasetVSKIMapStore``-compatible object.  Each iterator block
        must contain every row of one section and one nonoverlapping subset of
        the shared feature axis.
    joint_spots
        Saved joint spot table aligned exactly to ``store.spots``.
    domain_probabilities
        Saved soft responsibilities with shape ``(n_spots, n_domains)``.
    section_ids
        Section identifiers in their joint-row order.
    spot_areas
        Optional positive area for every joint spot.  Mutually exclusive with
        ``spot_pitches``.
    spot_pitches
        Optional positive pitch per section; its square is used as spot area.
        When neither area input is supplied, every spot has unit area.
    zero_denominator_tolerance
        Nonnegative threshold below which a clipped common-panel total is
        treated as zero.
    """

    names = [str(value) for value in section_ids]
    if not names or len(set(names)) != len(names):
        raise ValueError("section_ids must be nonempty and unique")
    if (
        not np.isfinite(zero_denominator_tolerance)
        or zero_denominator_tolerance < 0.0
    ):
        raise ValueError("zero_denominator_tolerance must be finite and nonnegative")

    probabilities = np.asarray(domain_probabilities)
    validate_joint_axes(store, joint_spots, probabilities, names)
    areas, area_source = _resolve_spot_areas(
        joint_spots,
        names,
        spot_areas=spot_areas,
        spot_pitches=spot_pitches,
    )
    section_rows = _section_rows(joint_spots, names)
    peak_ids = np.asarray(store.genes, dtype=str)
    transform_center, transform_scale, transform_provenance = (
        _standardized_gaussian_inverse_transforms(store, names, peak_ids)
    )

    # Pass 1: the relative-abundance denominator spans every common peak, so it
    # must be complete before any feature block can be normalized.
    spot_totals = np.zeros(store.n_spots, dtype=np.float64)
    for _, rows, _, clipped in _iter_validated_mean_blocks(
        store,
        section_rows,
        peak_ids,
        transform_center,
        transform_scale,
    ):
        spot_totals[rows] += clipped.sum(axis=1)

    zero_denominator = spot_totals <= float(zero_denominator_tolerance)
    zero_counts = np.asarray(
        [np.count_nonzero(zero_denominator[rows]) for rows in section_rows],
        dtype=np.int64,
    )

    d_count = len(names)
    r_count = probabilities.shape[1]
    p_count = peak_ids.size
    numerators = np.zeros((d_count, r_count, p_count), dtype=np.float64)
    domain_weight_sums = np.zeros((d_count, r_count), dtype=np.float64)
    section_weights: list[np.ndarray] = []
    for d, rows in enumerate(section_rows):
        weights = areas[rows, None] * np.asarray(
            probabilities[rows], dtype=np.float64
        )
        weights[zero_denominator[rows], :] = 0.0
        section_weights.append(weights)
        domain_weight_sums[d] = weights.sum(axis=0)

    # Pass 2: normalize each block and accumulate its domain sufficient
    # statistics without ever materializing the spot x peak panel.
    for d, rows, positions, clipped in _iter_validated_mean_blocks(
        store,
        section_rows,
        peak_ids,
        transform_center,
        transform_scale,
    ):
        relative = np.divide(
            clipped,
            spot_totals[rows, None],
            out=np.zeros_like(clipped),
            where=~zero_denominator[rows, None],
        )
        numerators[d][:, positions] = section_weights[d].T @ relative

    means = np.divide(
        numerators,
        domain_weight_sums[:, :, None],
        out=np.full_like(numerators, np.nan),
        where=domain_weight_sums[:, :, None] > 0.0,
    )
    values = means[..., None]
    section_ids_array = np.asarray(names, dtype=str)
    domain_ids = np.arange(r_count, dtype=np.int64)
    statistic_ids = np.asarray([AVERAGE_RELATIVE_ABUNDANCE], dtype=str)
    metadata: dict[str, object] = {
        "method": "soft-domain average of per-spot common-panel relative abundance",
        "axis_order": ["section", "domain", "peak", "statistic"],
        "shape": [int(value) for value in values.shape],
        "statistic_definition": (
            "over spots i with positive common-panel total: sum_i(area_i * "
            "B_i,r * max(mean_i,p * standardization_std_p + "
            "standardization_mean_p, 0) / sum_q max(mean_i,q * "
            "standardization_std_q + standardization_mean_q, 0)) / "
            "sum_i(area_i * B_i,r)"
        ),
        "relative_abundance_denominator": "all common-panel peaks within each spot",
        "denominator_scope": f"{p_count}_common_peak_panel",
        "posterior_mean_inverse_transform": transform_provenance,
        "negative_fitted_mean_policy": (
            "inverse-transform standardized Gaussian posterior mean, then clip to zero"
        ),
        "zero_denominator_policy": "exclude spot from all domain averages",
        "zero_denominator_tolerance": float(zero_denominator_tolerance),
        "zero_denominator_counts_by_section": {
            name: int(count) for name, count in zip(names, zero_counts)
        },
        "valid_spot_counts_by_section": {
            name: int(rows.size - count)
            for name, rows, count in zip(names, section_rows, zero_counts)
        },
        "spot_area_source": area_source,
        "map_standardization": getattr(store, "map_standardization", None),
        "n_common_peaks": int(p_count),
        "n_domains": int(r_count),
        "n_sections": int(d_count),
    }
    return DomainPeakStatisticsResult(
        values=values,
        section_ids=section_ids_array,
        domain_ids=domain_ids,
        peak_ids=peak_ids.copy(),
        statistic_ids=statistic_ids,
        zero_denominator_counts=zero_counts,
        domain_weight_sums=domain_weight_sums,
        metadata=metadata,
    )


def _resolve_spot_areas(
    joint_spots: pd.DataFrame,
    section_ids: Sequence[str],
    *,
    spot_areas: np.ndarray | Sequence[float] | None,
    spot_pitches: Sequence[float] | None,
) -> tuple[np.ndarray, str]:
    if spot_areas is not None and spot_pitches is not None:
        raise ValueError("Provide spot_areas or spot_pitches, not both")
    if spot_areas is not None:
        areas = np.asarray(spot_areas, dtype=np.float64)
        source = "caller-supplied per-spot area"
    elif spot_pitches is not None:
        pitches = np.asarray(spot_pitches, dtype=np.float64)
        if pitches.shape != (len(section_ids),):
            raise ValueError("spot_pitches must contain one value per section")
        if not np.all(np.isfinite(pitches)) or np.any(pitches <= 0.0):
            raise ValueError("spot_pitches must be finite and positive")
        lookup = dict(zip(section_ids, np.square(pitches)))
        areas = joint_spots["dataset"].astype(str).map(lookup).to_numpy(dtype=np.float64)
        source = "squared caller-supplied section pitch"
    else:
        areas = np.ones(joint_spots.shape[0], dtype=np.float64)
        source = "unit area"
    if areas.shape != (joint_spots.shape[0],):
        raise ValueError("spot_areas must have one value per joint spot")
    if not np.all(np.isfinite(areas)) or np.any(areas <= 0.0):
        raise ValueError("spot areas must be finite and positive")
    return areas, source


def _section_rows(
    joint_spots: pd.DataFrame, section_ids: Sequence[str]
) -> list[np.ndarray]:
    observed = joint_spots["dataset"].astype(str).to_numpy()
    rows = [np.flatnonzero(observed == name) for name in section_ids]
    if any(value.size == 0 for value in rows):
        missing = [name for name, value in zip(section_ids, rows) if value.size == 0]
        raise ValueError(f"No joint spots found for sections: {missing}")
    return rows


def _standardized_gaussian_inverse_transforms(store, section_ids, peak_ids):
    """Validate and load per-section training-scale inverse transforms."""

    children = getattr(store, "children", None)
    if children is None or len(children) != len(section_ids):
        raise ValueError(
            "store.children must provide one fitted-map source per section for "
            "standardized-Gaussian inverse transformation"
        )
    centers = np.empty((len(section_ids), peak_ids.size), dtype=np.float64)
    scales = np.empty_like(centers)
    sources: dict[str, object] = {}
    required = {
        "standardization_mean",
        "standardization_std",
        "posterior_map_scale",
    }
    for d, (section, child) in enumerate(zip(section_ids, children)):
        if getattr(child, "map_standardization", None) != "none":
            raise ValueError(
                f"{section}: relative abundance requires an uncached store with "
                "map_standardization='none'; LBM z-score caches cannot be inverted "
                "with the training standardization metadata"
            )
        metadata = getattr(child, "metadata", None)
        if not isinstance(metadata, pd.DataFrame):
            raise ValueError(f"{section}: fitted-map metadata is unavailable")
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise ValueError(
                f"{section}: fitted-map metadata lacks inverse-transform columns: {missing}"
            )
        if metadata.shape[0] != peak_ids.size:
            raise ValueError(
                f"{section}: fitted-map metadata feature count does not match common panel"
            )
        if "feature_position" in metadata:
            observed_positions = metadata["feature_position"].to_numpy(dtype=np.int64)
            if not np.array_equal(observed_positions, np.arange(peak_ids.size)):
                raise ValueError(
                    f"{section}: fitted-map metadata is not in common feature order"
                )
        posterior_scales = metadata["posterior_map_scale"].astype(str).to_numpy()
        if not np.all(posterior_scales == "standardized_gaussian"):
            observed = sorted(np.unique(posterior_scales).tolist())
            raise ValueError(
                f"{section}: expected posterior_map_scale='standardized_gaussian', "
                f"observed {observed}"
            )
        if "observation_model" in metadata:
            observation_models = metadata["observation_model"].astype(str).to_numpy()
            if not np.all(observation_models == "gaussian"):
                observed = sorted(np.unique(observation_models).tolist())
                raise ValueError(
                    f"{section}: expected Gaussian observation metadata, observed {observed}"
                )
        centers[d] = metadata["standardization_mean"].to_numpy(dtype=np.float64)
        scales[d] = metadata["standardization_std"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(centers[d])):
            raise ValueError(f"{section}: standardization_mean contains non-finite values")
        if not np.all(np.isfinite(scales[d])) or np.any(scales[d] <= 0.0):
            raise ValueError(
                f"{section}: standardization_std must be finite and positive"
            )
        selected_models_file = getattr(child, "selected_models_file", None)
        sources[str(section)] = {
            "posterior_map_scale": "standardized_gaussian",
            "metadata_source": (
                None if selected_models_file is None else str(selected_models_file)
            ),
        }
    provenance = {
        "applied": True,
        "formula": (
            "input_scale_mean = posterior_mean * standardization_std + "
            "standardization_mean"
        ),
        "metadata_columns": [
            "posterior_map_scale",
            "standardization_mean",
            "standardization_std",
        ],
        "required_source_scale": "standardized_gaussian",
        "sections": sources,
    }
    return centers, scales, provenance


def _iter_validated_mean_blocks(
    store,
    section_rows,
    peak_ids,
    transform_center,
    transform_scale,
):
    """Yield clipped blocks while checking complete dataset-feature coverage."""

    coverage = np.zeros((len(section_rows), peak_ids.size), dtype=bool)
    for raw_rows, raw_mean, _, raw_positions, raw_genes in store.iter_blocks(
        include_variance=False
    ):
        rows = np.asarray(raw_rows, dtype=np.int64)
        positions = np.asarray(raw_positions, dtype=np.int64)
        genes = np.asarray(raw_genes).astype(str)
        if rows.ndim != 1 or rows.size == 0 or np.unique(rows).size != rows.size:
            raise ValueError("Fitted-map iterator rows must be nonempty and unique")
        if positions.ndim != 1 or positions.size == 0:
            raise ValueError("Fitted-map iterator feature positions must be nonempty")
        if np.unique(positions).size != positions.size:
            raise ValueError("Fitted-map iterator contains duplicate feature positions")
        if positions.min() < 0 or positions.max() >= peak_ids.size:
            raise ValueError("Fitted-map iterator feature positions are out of range")
        if genes.shape != positions.shape or not np.array_equal(
            genes, peak_ids[positions]
        ):
            raise ValueError("Fitted-map iterator feature identifiers are misaligned")
        matches = [
            d for d, expected_rows in enumerate(section_rows)
            if np.array_equal(rows, expected_rows)
        ]
        if len(matches) != 1:
            raise ValueError(
                "Each fitted-map block must preserve the complete row order of one section"
            )
        d = matches[0]
        if coverage[d, positions].any():
            raise ValueError(
                f"Fitted-map iterator emitted duplicate feature positions for section {d}"
            )
        mean = np.asarray(raw_mean, dtype=np.float64)
        if mean.shape != (rows.size, positions.size):
            raise ValueError("Fitted-map iterator mean block has incompatible dimensions")
        if not np.all(np.isfinite(mean)):
            raise ValueError("Fitted posterior means contain non-finite values")
        input_scale_mean = (
            mean * transform_scale[d, positions][None, :]
            + transform_center[d, positions][None, :]
        )
        if not np.all(np.isfinite(input_scale_mean)):
            raise ValueError("Inverse-transformed fitted means contain non-finite values")
        coverage[d, positions] = True
        yield d, rows, positions, np.maximum(input_scale_mean, 0.0)
    if not coverage.all():
        missing = np.argwhere(~coverage)[:10].tolist()
        raise ValueError(f"Fitted-map iterator omitted section/feature positions {missing}")
