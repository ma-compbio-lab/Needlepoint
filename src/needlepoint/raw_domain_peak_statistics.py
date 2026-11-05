"""Sparse frozen-domain summaries for matched raw TEMI peak panels.

The joint domain responsibilities are treated as immutable. Within each
section, nonnegative raw intensities are normalized over that section's
complete raw called-peak catalog at every spot. The matched shared peaks are
then selected for export and averaged with represented spot area times the
saved soft-domain responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .domain_peak_statistics import (
    DomainPeakStatisticsResult,
    _resolve_spot_areas,
)
from .multidataset_lbm import reciprocal_nearest_peak_pairs


AVERAGE_RELATIVE_RAW_INTENSITY_FULL_CALLED_PANEL = (
    "average_relative_raw_intensity_full_called_panel"
)


@dataclass(frozen=True)
class RawPanelDiagnostics:
    """Auditable input diagnostics for a raw shared-peak tensor."""

    negative_entry_counts: np.ndarray
    clipped_negative_mass: np.ndarray
    row_sum_minimums: np.ndarray
    row_sum_maximums: np.ndarray


def build_anchored_raw_shared_peak_crosswalk(
    left_peak_metadata: pd.DataFrame,
    right_peak_metadata: pd.DataFrame,
    defining_crosswalk: pd.DataFrame,
    *,
    tolerance_ppm: float = 5.0,
    mz_min: float = 400.0,
    mz_max: float = 1200.0,
) -> pd.DataFrame:
    """Expand a defining crosswalk without changing any defining pair.

    Domain-defining pairs are immutable anchors.  Their left and right raw
    peak endpoints are removed before reciprocal-nearest matching of the
    remaining catalogs.  This makes the result a strict superset of the
    defining panel while retaining a conservative one-to-one match rule for
    every additional raw peak.
    """

    left = _prepare_raw_peak_catalog(left_peak_metadata, "left")
    right = _prepare_raw_peak_catalog(right_peak_metadata, "right")
    defining_required = {
        "canonical_peak_id",
        "left_peak_index",
        "right_peak_index",
    }
    missing = sorted(defining_required.difference(defining_crosswalk.columns))
    if missing:
        raise ValueError(f"Defining crosswalk lacks columns: {missing}")
    defining = defining_crosswalk.copy()
    defining["canonical_peak_id"] = defining["canonical_peak_id"].astype(str)
    if defining["canonical_peak_id"].duplicated().any():
        raise ValueError("Defining canonical_peak_id values must be unique")
    for side, catalog in (("left", left), ("right", right)):
        column = f"{side}_peak_index"
        indices = defining[column].to_numpy(dtype=np.int64)
        if np.unique(indices).size != indices.size:
            raise ValueError(f"Defining {side} peak endpoints must be unique")
        missing_indices = np.setdiff1d(indices, catalog["peak_index"].to_numpy())
        if missing_indices.size:
            raise ValueError(
                f"Defining {side} peaks are absent from raw metadata: "
                f"{missing_indices[:10].tolist()}"
            )

    left_lookup = left.set_index("peak_index", drop=False)
    right_lookup = right.set_index("peak_index", drop=False)
    anchored_records: list[dict[str, object]] = []
    for row in defining.itertuples(index=False):
        left_index = int(row.left_peak_index)
        right_index = int(row.right_peak_index)
        left_row = left_lookup.loc[left_index]
        right_row = right_lookup.loc[right_index]
        left_mz = float(left_row.peak_mz)
        right_mz = float(right_row.peak_mz)
        average_mz = 0.5 * (left_mz + right_mz)
        delta_mz = right_mz - left_mz
        delta_ppm = 1.0e6 * delta_mz / average_mz
        if abs(delta_ppm) > tolerance_ppm:
            raise ValueError(
                f"Defining pair {row.canonical_peak_id} exceeds {tolerance_ppm} ppm"
            )
        anchored_records.append(
            {
                "canonical_peak_id": str(row.canonical_peak_id),
                "left_peak_index": left_index,
                "left_gene": str(left_row.gene),
                "left_source_feature_position": int(
                    left_row.source_feature_position
                ),
                "left_peak_mz": left_mz,
                "right_peak_index": right_index,
                "right_gene": str(right_row.gene),
                "right_source_feature_position": int(
                    right_row.source_feature_position
                ),
                "right_peak_mz": right_mz,
                "delta_mz": delta_mz,
                "delta_ppm": delta_ppm,
                "absolute_delta_ppm": abs(delta_ppm),
                "was_used_for_joint_domain_fit": True,
                "match_rule": "domain_defining_anchor",
            }
        )

    used_left = set(defining["left_peak_index"].astype(np.int64))
    used_right = set(defining["right_peak_index"].astype(np.int64))
    remaining_left = left.loc[~left["peak_index"].isin(used_left)].copy()
    remaining_right = right.loc[~right["peak_index"].isin(used_right)].copy()
    additional = reciprocal_nearest_peak_pairs(
        remaining_left,
        remaining_right,
        tolerance_ppm=tolerance_ppm,
        mz_min=mz_min,
        mz_max=mz_max,
    ).sort_values(["left_peak_mz", "right_peak_mz"], kind="stable")
    additional = additional.reset_index(drop=True)
    additional["canonical_peak_id"] = [
        f"RAW_SHARED_{index:05d}" for index in range(additional.shape[0])
    ]
    additional["was_used_for_joint_domain_fit"] = False
    additional["match_rule"] = "reciprocal_nearest_after_anchor_exclusion"

    anchored = pd.DataFrame(anchored_records)
    output = pd.concat([anchored, additional], ignore_index=True)
    output["representative_peak_mz"] = output[
        ["left_peak_mz", "right_peak_mz"]
    ].mean(axis=1)
    output = output.sort_values(
        ["representative_peak_mz", "canonical_peak_id"], kind="stable"
    ).reset_index(drop=True)
    output.insert(0, "shared_peak_position", np.arange(output.shape[0], dtype=np.int64))
    if output["canonical_peak_id"].duplicated().any():
        raise ValueError("Expanded raw shared peak IDs are not unique")
    if output["left_peak_index"].duplicated().any() or output[
        "right_peak_index"
    ].duplicated().any():
        raise ValueError("Expanded raw shared crosswalk is not one-to-one")
    if int(output["was_used_for_joint_domain_fit"].sum()) != defining.shape[0]:
        raise ValueError("Expanded crosswalk lost domain-defining anchors")
    return output


def compute_raw_shared_peak_relative_abundance_tensor(
    section_stores: Sequence[object],
    joint_spots: pd.DataFrame,
    domain_probabilities: np.ndarray,
    section_ids: Sequence[str],
    canonical_peak_ids: Sequence[str],
    *,
    spot_areas: np.ndarray | Sequence[float] | None = None,
    spot_pitches: Sequence[float] | None = None,
    zero_denominator_tolerance: float = 0.0,
) -> tuple[DomainPeakStatisticsResult, RawPanelDiagnostics]:
    """Compute an area- and soft-domain-weighted raw composition tensor.

    Each store must expose ``iter_sparse_spot_chunks()`` and contain exactly
    one section in the same row order as the corresponding subset of
    ``joint_spots``. The store's normalization panel is the complete raw
    called-peak catalog for its section, whereas its emitted columns are the
    matched shared peaks. No dense spot-by-peak matrix is constructed.
    """

    names = [str(value) for value in section_ids]
    stores = list(section_stores)
    peak_ids = np.asarray(canonical_peak_ids, dtype=str)
    probabilities = np.asarray(domain_probabilities)
    if not names or len(set(names)) != len(names):
        raise ValueError("section_ids must be nonempty and unique")
    if len(stores) != len(names):
        raise ValueError("Provide one raw peak store per section")
    if peak_ids.ndim != 1 or peak_ids.size == 0:
        raise ValueError("canonical_peak_ids must be a nonempty vector")
    if np.unique(peak_ids).size != peak_ids.size:
        raise ValueError("canonical_peak_ids must be unique")
    if "dataset" not in joint_spots:
        raise ValueError("joint_spots must include dataset")
    if probabilities.ndim != 2 or probabilities.shape[0] != joint_spots.shape[0]:
        raise ValueError(
            "domain_probabilities must have shape (joint spots, domains)"
        )
    if probabilities.shape[1] == 0:
        raise ValueError("At least one domain is required")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("domain_probabilities must be finite and nonnegative")
    probability_sums = probabilities.sum(axis=1)
    if not np.allclose(probability_sums, 1.0, atol=1.0e-5, rtol=1.0e-5):
        raise ValueError("domain_probabilities rows must sum to one")
    if (
        not np.isfinite(zero_denominator_tolerance)
        or zero_denominator_tolerance < 0.0
    ):
        raise ValueError("zero_denominator_tolerance must be finite and nonnegative")

    areas, area_source = _resolve_spot_areas(
        joint_spots,
        names,
        spot_areas=spot_areas,
        spot_pitches=spot_pitches,
    )
    observed_sections = joint_spots["dataset"].astype(str).to_numpy()
    unknown = sorted(set(observed_sections) - set(names))
    if unknown:
        raise ValueError(f"joint_spots contains unexpected sections: {unknown}")
    section_rows = [np.flatnonzero(observed_sections == name) for name in names]
    if any(rows.size == 0 for rows in section_rows):
        missing = [name for name, rows in zip(names, section_rows) if rows.size == 0]
        raise ValueError(f"No joint spots found for sections: {missing}")

    d_count = len(names)
    r_count = probabilities.shape[1]
    p_count = peak_ids.size
    numerators = np.zeros((d_count, r_count, p_count), dtype=np.float64)
    domain_weight_sums = np.zeros((d_count, r_count), dtype=np.float64)
    zero_counts = np.zeros(d_count, dtype=np.int64)
    negative_counts = np.zeros(d_count, dtype=np.int64)
    clipped_masses = np.zeros(d_count, dtype=np.float64)
    row_sum_minimums = np.full(d_count, np.inf, dtype=np.float64)
    row_sum_maximums = np.full(d_count, -np.inf, dtype=np.float64)
    denominator_peak_counts: dict[str, int] = {}

    for d, (name, store, global_rows) in enumerate(
        zip(names, stores, section_rows, strict=True)
    ):
        _validate_raw_section_store(store, joint_spots.iloc[global_rows], name, p_count)
        denominator_peak_counts[name] = int(
            np.asarray(getattr(store, "normalization_peak_indices")).size
        )
        target_sum = float(getattr(store, "target_sum"))
        base_weights = (
            areas[global_rows, None]
            * np.asarray(probabilities[global_rows], dtype=np.float64)
        )
        seen = np.zeros(global_rows.size, dtype=bool)
        iterator = getattr(store, "iter_sparse_spot_chunks", None)
        if iterator is None:
            raise ValueError(
                f"{name}: raw store does not expose iter_sparse_spot_chunks()"
            )
        for (
            raw_rows,
            raw_matrix,
            raw_normalization_totals,
            raw_negative_count,
            raw_clipped_negative_mass,
        ) in iterator():
            rows = np.asarray(raw_rows, dtype=np.int64)
            matrix = sparse.csr_matrix(raw_matrix, dtype=np.float64)
            if rows.ndim != 1 or rows.size == 0 or np.unique(rows).size != rows.size:
                raise ValueError(f"{name}: sparse chunk rows must be nonempty and unique")
            if rows.min() < 0 or rows.max() >= global_rows.size:
                raise ValueError(f"{name}: sparse chunk rows are outside the section")
            if seen[rows].any():
                raise ValueError(f"{name}: sparse chunk rows overlap")
            if matrix.shape != (rows.size, p_count):
                raise ValueError(
                    f"{name}: sparse chunk has shape {matrix.shape}, expected "
                    f"({rows.size}, {p_count})"
                )
            if matrix.data.size and not np.all(np.isfinite(matrix.data)):
                raise ValueError(f"{name}: raw intensities contain non-finite values")
            if matrix.data.size and np.any(matrix.data < 0.0):
                raise ValueError(
                    f"{name}: sparse raw iterator did not apply nonnegative clipping"
                )
            row_totals = np.asarray(
                raw_normalization_totals, dtype=np.float64
            )
            if row_totals.shape != (rows.size,):
                raise ValueError(
                    f"{name}: normalization totals do not match sparse chunk rows"
                )
            if not np.all(np.isfinite(row_totals)) or np.any(row_totals < 0.0):
                raise ValueError(
                    f"{name}: normalization totals must be finite and nonnegative"
                )
            negative_counts[d] += int(raw_negative_count)
            clipped_masses[d] += float(raw_clipped_negative_mass)
            row_sum_minimums[d] = min(row_sum_minimums[d], float(row_totals.min()))
            row_sum_maximums[d] = max(row_sum_maximums[d], float(row_totals.max()))
            valid = row_totals > float(zero_denominator_tolerance)
            zero_counts[d] += int(np.count_nonzero(~valid))
            relative = matrix.multiply(1.0 / target_sum).tocsr()

            weights = base_weights[rows].copy()
            weights[~valid] = 0.0
            domain_weight_sums[d] += weights.sum(axis=0)
            block_numerator = weights.T @ relative
            numerators[d] += np.asarray(block_numerator, dtype=np.float64)
            seen[rows] = True
        if not seen.all():
            missing = np.flatnonzero(~seen)[:10].tolist()
            raise ValueError(f"{name}: sparse chunks omitted local rows {missing}")

    means = np.divide(
        numerators,
        domain_weight_sums[:, :, None],
        out=np.full_like(numerators, np.nan),
        where=domain_weight_sums[:, :, None] > 0.0,
    )
    values = means[..., None]
    metadata: dict[str, object] = {
        "method": "soft-domain average of per-spot raw called-panel relative intensity",
        "axis_order": ["section", "domain", "peak", "statistic"],
        "shape": [int(value) for value in values.shape],
        "statistic_definition": (
            "over spots i with positive full-called-panel total: sum_i(area_i * B_i,r * "
            "max(raw_i,p, 0) / sum_q max(raw_i,q, 0)) / sum_i(area_i * B_i,r), "
            "where q spans the complete raw called-peak catalog of section d"
        ),
        "relative_abundance_denominator": (
            "complete section-specific raw called-peak catalog within each spot"
        ),
        "denominator_scope": "section-specific complete raw called-peak panels",
        "denominator_peak_counts_by_section": denominator_peak_counts,
        "selected_output_peak_count": int(p_count),
        "shared_output_values_sum_to_one": False,
        "normalization_note": (
            "The selected shared output peaks are a subset of each denominator; "
            "their values therefore need not sum to one. This is called-peak-panel "
            "relative intensity, not continuous-spectrum TIC abundance."
        ),
        "raw_transform": "none",
        "negative_intensity_policy": "clip to zero before panel normalization",
        "zero_denominator_policy": "exclude spot from all domain averages",
        "zero_denominator_tolerance": float(zero_denominator_tolerance),
        "zero_denominator_counts_by_section": {
            name: int(count) for name, count in zip(names, zero_counts, strict=True)
        },
        "spot_area_source": area_source,
        "n_shared_raw_peaks": int(p_count),
        "n_domains": int(r_count),
        "n_sections": int(d_count),
        "fixed_domain_contract": (
            "saved joint soft-domain probabilities are read-only and not refitted"
        ),
    }
    result = DomainPeakStatisticsResult(
        values=values,
        section_ids=np.asarray(names, dtype=str),
        domain_ids=np.arange(r_count, dtype=np.int64),
        peak_ids=peak_ids.copy(),
        statistic_ids=np.asarray(
            [AVERAGE_RELATIVE_RAW_INTENSITY_FULL_CALLED_PANEL], dtype=str
        ),
        zero_denominator_counts=zero_counts,
        domain_weight_sums=domain_weight_sums,
        metadata=metadata,
    )
    diagnostics = RawPanelDiagnostics(
        negative_entry_counts=negative_counts,
        clipped_negative_mass=clipped_masses,
        row_sum_minimums=row_sum_minimums,
        row_sum_maximums=row_sum_maximums,
    )
    return result, diagnostics


def _validate_raw_section_store(
    store: object,
    section_spots: pd.DataFrame,
    section_id: str,
    n_peaks: int,
) -> None:
    if int(getattr(store, "n_spots", -1)) != section_spots.shape[0]:
        raise ValueError(f"{section_id}: raw store and joint spot counts differ")
    if int(getattr(store, "n_features", -1)) != n_peaks:
        raise ValueError(f"{section_id}: raw store feature count differs from crosswalk")
    if getattr(store, "map_standardization", None) != "none":
        raise ValueError(f"{section_id}: raw tensor requires map_standardization='none'")
    if getattr(store, "transform", None) != "none":
        raise ValueError(f"{section_id}: raw tensor requires transform='none'")
    target_sum = float(getattr(store, "target_sum", np.nan))
    if not np.isfinite(target_sum) or target_sum <= 0.0:
        raise ValueError(f"{section_id}: raw store target_sum must be positive")
    normalization_peak_indices = np.asarray(
        getattr(store, "normalization_peak_indices", np.empty(0)), dtype=np.int64
    )
    if normalization_peak_indices.ndim != 1 or normalization_peak_indices.size == 0:
        raise ValueError(f"{section_id}: raw normalization panel is unavailable")

    raw_spots = getattr(store, "spots", None)
    if not isinstance(raw_spots, pd.DataFrame) or raw_spots.shape[0] != section_spots.shape[0]:
        raise ValueError(f"{section_id}: raw store spot table is unavailable")
    if "dataset_map_row" in section_spots and "map_row" in raw_spots:
        expected = section_spots["dataset_map_row"].to_numpy(dtype=np.int64)
        observed = raw_spots["map_row"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise ValueError(f"{section_id}: raw and joint spot row orders differ")
    elif "spot_index" in section_spots and "spot_index" in raw_spots:
        expected = section_spots["spot_index"].to_numpy(dtype=np.int64)
        observed = raw_spots["spot_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise ValueError(f"{section_id}: raw and joint spot indices differ")
    else:
        raise ValueError(f"{section_id}: no shared spot-row identifier is available")

    if {"native_x", "native_y"}.issubset(section_spots.columns) and {
        "x",
        "y",
    }.issubset(raw_spots.columns):
        expected_xy = section_spots[["native_x", "native_y"]].to_numpy(dtype=np.float64)
        observed_xy = raw_spots[["x", "y"]].to_numpy(dtype=np.float64)
        if not np.allclose(observed_xy, expected_xy, atol=1.0e-6, rtol=0.0):
            raise ValueError(f"{section_id}: raw and joint native coordinates differ")


def _prepare_raw_peak_catalog(table: pd.DataFrame, side: str) -> pd.DataFrame:
    required = {"peak_index", "peak_mz"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{side} raw peak metadata lacks columns: {missing}")
    catalog = table.copy()
    catalog["peak_index"] = pd.to_numeric(
        catalog["peak_index"], errors="raise"
    ).astype(np.int64)
    catalog["peak_mz"] = pd.to_numeric(
        catalog["peak_mz"], errors="raise"
    ).astype(np.float64)
    if catalog["peak_index"].duplicated().any():
        raise ValueError(f"{side} raw peak indices must be unique")
    if not np.all(np.isfinite(catalog["peak_mz"])):
        raise ValueError(f"{side} raw peak m/z values must be finite")
    expected = np.arange(catalog.shape[0], dtype=np.int64)
    observed = np.sort(catalog["peak_index"].to_numpy())
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"{side} raw peak indices must be contiguous raw matrix columns"
        )
    catalog["gene"] = "peak_" + catalog["peak_index"].astype(str)
    catalog["source_feature_position"] = catalog["peak_index"]
    return catalog
