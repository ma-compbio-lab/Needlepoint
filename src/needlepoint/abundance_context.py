"""Abundance-aware context for domain-conditioned differential contrasts.

The differential workflow measures changes in standardized spatial deployment.
Those changes can be large in a domain where a peak is low in both sections.
This module keeps every original contrast and adds a second, explicitly
descriptive context derived from the frozen-domain raw-intensity tensor.

Let ``A[d, r, p]`` be the area- and soft-domain-weighted average relative raw
intensity for section ``d``, domain ``r``, and peak ``p``.  With domain area
fraction ``pi[d, r]``, the section-wide reference and two useful summaries are

``Abar[d, p] = sum_r pi[d, r] * A[d, r, p]``

``raw_abundance_fold[d, r, p] = A[d, r, p] / Abar[d, p]``

``peak_mass_share[d, r, p] = pi[d, r] * A[d, r, p] / Abar[d, p]``.

The raw tensor is section-compositional and is not an absolute-abundance or
batch-corrected measurement.  It is used only to label the abundance context
of the existing standardized-GP contrasts; it does not alter their estimates,
bootstrap results, or multiple-testing correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


RAW_STATISTIC_ID = "average_relative_raw_intensity_full_called_panel"


@dataclass(frozen=True)
class AbundanceContextThresholds:
    """Thresholds defining the retained and abundance-aware result tiers."""

    effect_size: float = 0.5
    conditional_qvalue: float = 0.05
    sign_stability: float = 0.95
    abundance_fold: float = 1.0
    headline_top_n: int = 3


@dataclass(frozen=True)
class LabeledRawDomainTensor:
    """The one-statistic raw tensor and its labeled axes."""

    values: np.ndarray
    section_ids: np.ndarray
    domain_ids: np.ndarray
    peak_ids: np.ndarray
    domain_weight_sums: np.ndarray
    statistic_id: str


@dataclass(frozen=True)
class AbundanceContextResult:
    """Original tables augmented with per-contrast and per-peak context."""

    contrast_table: pd.DataFrame
    peak_table: pd.DataFrame
    metadata: dict[str, object]


def load_raw_domain_tensor_npz(
    path: str | Path,
    *,
    statistic_id: str = RAW_STATISTIC_ID,
) -> LabeledRawDomainTensor:
    """Load one named statistic from a labeled domain-by-peak NPZ artifact."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "values",
            "section_ids",
            "domain_ids",
            "peak_ids",
            "statistic_ids",
            "domain_weight_sums",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Raw tensor NPZ lacks arrays: {missing}")
        values = np.asarray(archive["values"], dtype=np.float64)
        section_ids = np.asarray(archive["section_ids"]).astype(str)
        domain_ids = np.asarray(archive["domain_ids"])
        peak_ids = np.asarray(archive["peak_ids"]).astype(str)
        statistic_ids = np.asarray(archive["statistic_ids"]).astype(str)
        domain_weight_sums = np.asarray(
            archive["domain_weight_sums"], dtype=np.float64
        )

    matches = np.flatnonzero(statistic_ids == str(statistic_id))
    if matches.size != 1:
        raise ValueError(
            f"Expected exactly one statistic {statistic_id!r}; observed "
            f"{statistic_ids.tolist()}"
        )
    if values.ndim != 4:
        raise ValueError(
            "Raw tensor values must have axis order section x domain x peak x "
            "statistic"
        )
    if values.shape != (
        section_ids.size,
        domain_ids.size,
        peak_ids.size,
        statistic_ids.size,
    ):
        raise ValueError("Raw tensor values and labeled axes have different shapes")
    return LabeledRawDomainTensor(
        values=values[..., int(matches[0])],
        section_ids=section_ids,
        domain_ids=domain_ids,
        peak_ids=peak_ids,
        domain_weight_sums=domain_weight_sums,
        statistic_id=str(statistic_id),
    )


def classify_abundance_context(
    peak_domain_contrasts: pd.DataFrame,
    peak_summary: pd.DataFrame,
    raw_tensor: LabeledRawDomainTensor,
    *,
    left_section: str | None = None,
    right_section: str | None = None,
    thresholds: AbundanceContextThresholds | None = None,
) -> AbundanceContextResult:
    """Annotate all existing differential rows with raw-abundance context.

    The existing robust rule is reproduced from the original columns.  A
    robust contrast is called *primary abundance-relevant* when its raw
    abundance fold is at least ``abundance_fold`` in either section.  A
    primary contrast is in the *headline high-domain* tier when the tested
    domain ranks within the highest ``headline_top_n`` raw-abundance domains
    in either section.  No rows are removed.
    """

    thresholds = thresholds or AbundanceContextThresholds()
    _validate_thresholds(thresholds)
    contrasts = peak_domain_contrasts.copy()
    peaks = peak_summary.copy()
    _validate_input_tables(contrasts, peaks)
    values, weights = _validate_raw_tensor(raw_tensor)

    if left_section is None and right_section is None:
        if raw_tensor.section_ids.size != 2:
            raise ValueError(
                "left_section and right_section are required unless the raw tensor "
                "contains exactly two sections"
            )
        left_section = str(raw_tensor.section_ids[0])
        right_section = str(raw_tensor.section_ids[1])
    elif left_section is None or right_section is None:
        raise ValueError("Provide both left_section and right_section, or neither")
    left_section = str(left_section)
    right_section = str(right_section)
    if left_section == right_section:
        raise ValueError("left_section and right_section must differ")

    section_lookup = _unique_lookup(raw_tensor.section_ids, "section")
    try:
        section_positions = np.asarray(
            [section_lookup[left_section], section_lookup[right_section]],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            f"Requested section is absent from raw tensor: {error.args[0]}"
        ) from error

    # Restrict to the two comparison sections while retaining the complete raw
    # peak and domain axes needed to calculate ranks and section-wide means.
    abundance = values[section_positions]
    selected_weights = weights[section_positions]
    domain_area_fraction = selected_weights / selected_weights.sum(axis=1)[:, None]
    global_abundance = np.einsum(
        "dr,drp->dp", domain_area_fraction, abundance, optimize=True
    )
    raw_fold = np.divide(
        abundance,
        global_abundance[:, None, :],
        out=np.full_like(abundance, np.nan),
        where=global_abundance[:, None, :] > 0.0,
    )
    peak_mass_share = domain_area_fraction[:, :, None] * raw_fold
    raw_rank = _descending_min_rank(abundance)

    peak_lookup = _unique_lookup(raw_tensor.peak_ids, "peak")
    domain_lookup = _unique_lookup(raw_tensor.domain_ids, "domain")
    contrast_peak_positions = _lookup_positions(
        contrasts["canonical_peak_id"], peak_lookup, "contrast peaks"
    )
    contrast_domain_positions = _lookup_positions(
        contrasts["domain"], domain_lookup, "contrast domains"
    )
    left_raw = abundance[0, contrast_domain_positions, contrast_peak_positions]
    right_raw = abundance[1, contrast_domain_positions, contrast_peak_positions]
    left_global = global_abundance[0, contrast_peak_positions]
    right_global = global_abundance[1, contrast_peak_positions]
    left_fold = raw_fold[0, contrast_domain_positions, contrast_peak_positions]
    right_fold = raw_fold[1, contrast_domain_positions, contrast_peak_positions]
    left_share = peak_mass_share[
        0, contrast_domain_positions, contrast_peak_positions
    ]
    right_share = peak_mass_share[
        1, contrast_domain_positions, contrast_peak_positions
    ]
    left_rank = raw_rank[0, contrast_domain_positions, contrast_peak_positions]
    right_rank = raw_rank[1, contrast_domain_positions, contrast_peak_positions]
    left_pi = domain_area_fraction[0, contrast_domain_positions]
    right_pi = domain_area_fraction[1, contrast_domain_positions]

    support = _coerce_bool(contrasts["support_flag"], "support_flag")
    delta = contrasts["delta_enrichment"].to_numpy(dtype=np.float64)
    left_enrichment = contrasts["left_enrichment"].to_numpy(dtype=np.float64)
    right_enrichment = contrasts["right_enrichment"].to_numpy(dtype=np.float64)
    qvalue = contrasts["conditional_qvalue"].to_numpy(dtype=np.float64)
    stability = contrasts["sign_stability"].to_numpy(dtype=np.float64)
    robust = (
        support
        & (np.abs(delta) >= thresholds.effect_size)
        & (qvalue <= thresholds.conditional_qvalue)
        & (stability >= thresholds.sign_stability)
    )
    raw_context_available = np.isfinite(left_fold) & np.isfinite(right_fold)
    raw_below_average_both = (
        raw_context_available
        & (left_fold < thresholds.abundance_fold)
        & (right_fold < thresholds.abundance_fold)
    )
    depletion_only = robust & (left_enrichment < 0.0) & (right_enrichment < 0.0)
    primary = (
        robust
        & raw_context_available
        & (np.maximum(left_fold, right_fold) >= thresholds.abundance_fold)
    )
    raw_top = (left_rank <= thresholds.headline_top_n) | (
        right_rank <= thresholds.headline_top_n
    )
    headline = primary & raw_top
    dominant_left = left_rank == 1
    dominant_right = right_rank == 1
    dominant_corroborated = dominant_left | dominant_right
    raw_delta = right_raw - left_raw
    direction_evaluable = np.isfinite(raw_delta) & (raw_delta != 0.0) & (delta != 0.0)
    direction_concordant = direction_evaluable & (
        np.sign(raw_delta) == np.sign(delta)
    )

    annotations: dict[str, object] = {
        "left_raw_relative_intensity": left_raw,
        "right_raw_relative_intensity": right_raw,
        "raw_relative_intensity_delta_right_minus_left": raw_delta,
        "left_section_area_weighted_global_raw_abundance": left_global,
        "right_section_area_weighted_global_raw_abundance": right_global,
        "left_raw_abundance_fold": left_fold,
        "right_raw_abundance_fold": right_fold,
        "left_peak_mass_share": left_share,
        "right_peak_mass_share": right_share,
        "left_domain_area_fraction": left_pi,
        "right_domain_area_fraction": right_pi,
        "left_raw_domain_rank": left_rank,
        "right_raw_domain_rank": right_rank,
        "raw_context_available": raw_context_available,
        "raw_below_average_both": raw_below_average_both,
        "depletion_only": depletion_only,
        "robust_contrast": robust,
        "primary_abundance_relevant": primary,
        "raw_top_n_either_section": raw_top,
        "headline_top3": headline,
        "headline_high_domain": headline,
        "raw_dominant_domain_left": dominant_left,
        "raw_dominant_domain_right": dominant_right,
        "dominant_domain_corroborated": dominant_corroborated,
        "raw_direction_concordance_evaluable": direction_evaluable,
        "raw_direction_concordant": direction_concordant,
    }
    for column, data in annotations.items():
        if column in contrasts.columns:
            raise ValueError(f"Input contrast table already contains {column!r}")
        contrasts[column] = data
    peak_has_headline = contrasts.groupby("canonical_peak_id", sort=False)[
        "headline_top3"
    ].transform("any")
    peak_has_primary = contrasts.groupby("canonical_peak_id", sort=False)[
        "primary_abundance_relevant"
    ].transform("any")
    contrasts["gallery_eligible"] = np.where(
        peak_has_headline,
        contrasts["headline_top3"],
        np.where(
            peak_has_primary,
            contrasts["primary_abundance_relevant"],
            False,
        ),
    ).astype(bool)
    contrasts["abundance_context_class"] = np.select(
        [
            ~contrasts["robust_contrast"],
            contrasts["headline_high_domain"],
            contrasts["primary_abundance_relevant"],
            contrasts["depletion_only"] & contrasts["raw_below_average_both"],
            contrasts["raw_below_average_both"],
        ],
        [
            "not_robust",
            "headline_high_domain",
            "primary_abundance_relevant",
            "secondary_depletion_only",
            "secondary_below_average_raw",
        ],
        default="raw_context_unavailable",
    )
    contrasts["_domain_axis_position"] = contrast_domain_positions
    peak_annotations = _summarize_peaks(contrasts, peaks["canonical_peak_id"])
    contrasts.drop(columns="_domain_axis_position", inplace=True)
    peaks = peaks.merge(
        peak_annotations,
        on="canonical_peak_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    _fill_peak_defaults(peaks)
    peaks["selected_gallery_domain"] = peaks["selected_headline_domain"].where(
        peaks["has_headline_contrast"], peaks["selected_primary_domain"]
    )
    peaks["selected_gallery_delta_enrichment"] = peaks[
        "selected_headline_delta_enrichment"
    ].where(
        peaks["has_headline_contrast"],
        peaks["selected_primary_delta_enrichment"],
    )
    peaks["selected_gallery_absolute_delta_enrichment"] = peaks[
        "selected_headline_absolute_delta_enrichment"
    ].where(
        peaks["has_headline_contrast"],
        peaks["selected_primary_absolute_delta_enrichment"],
    )
    peaks["selected_gallery_tier"] = np.select(
        [peaks["has_headline_contrast"], peaks["has_primary_contrast"]],
        ["headline_top3", "primary_above_average"],
        default="tail_only_or_not_robust",
    )

    metadata: dict[str, object] = {
        "schema_version": "1.0.0",
        "method": "additive raw-abundance context for existing domain contrasts",
        "sections": {"left": left_section, "right": right_section},
        "source_raw_statistic": raw_tensor.statistic_id,
        "raw_tensor_axis_order": ["section", "domain", "peak"],
        "thresholds": asdict(thresholds),
        "formulas": {
            "domain_area_fraction": (
                "pi[d,r] = domain_weight_sums[d,r] / "
                "sum_s domain_weight_sums[d,s]"
            ),
            "section_area_weighted_global_abundance": (
                "Abar[d,p] = sum_r pi[d,r] * A[d,r,p]"
            ),
            "raw_abundance_fold": "raw_fold[d,r,p] = A[d,r,p] / Abar[d,p]",
            "peak_mass_share": (
                "mass_share[d,r,p] = pi[d,r] * A[d,r,p] / "
                "sum_s pi[d,s] * A[d,s,p] = pi[d,r] * raw_fold[d,r,p]"
            ),
            "raw_delta": "A[right,r,p] - A[left,r,p]",
        },
        "definitions": {
            "raw_domain_rank": (
                "descending within each section and peak across domains; rank 1 "
                "is highest and ties receive the minimum rank"
            ),
            "robust_contrast": (
                "support_flag and abs(delta_enrichment) >= effect_size and "
                "conditional_qvalue <= conditional_qvalue threshold and "
                "sign_stability >= sign_stability threshold"
            ),
            "raw_below_average_both": (
                "both section-specific raw_abundance_fold values are strictly "
                "below the abundance_fold threshold"
            ),
            "depletion_only": (
                "robust_contrast and left_enrichment < 0 and right_enrichment < 0"
            ),
            "primary_abundance_relevant": (
                "robust_contrast and max(left_raw_abundance_fold, "
                "right_raw_abundance_fold) >= abundance_fold threshold"
            ),
            "headline_high_domain": (
                "primary_abundance_relevant and raw domain rank <= headline_top_n "
                "in either section"
            ),
            "selected_gallery_domain": (
                "strongest headline_top3 domain when one exists; otherwise the "
                "strongest primary_above_average domain; tail-only and non-robust "
                "peaks have no gallery domain"
            ),
            "gallery_eligible": (
                "if a peak has any headline_top3 contrast, true for all of that "
                "peak's headline_top3 rows only; otherwise, if it has any primary "
                "abundance-relevant contrast, true for those primary rows only; "
                "false for tail-only and non-robust peaks"
            ),
            "dominant_domain_corroborated": (
                "the tested domain has raw domain rank 1 in either section"
            ),
            "raw_direction_concordant": (
                "both deltas are nonzero and sign(delta_enrichment) equals "
                "sign(A[right,r,p] - A[left,r,p])"
            ),
            "tail_only_peak": (
                "the peak has at least one robust contrast and no primary "
                "abundance-relevant contrast"
            ),
            "selected_tier_domain": (
                "eligible domain with largest abs(delta_enrichment); ties are "
                "resolved by raw tensor domain-axis order"
            ),
        },
        "interpretation": {
            "differential_estimand": (
                "Original enrichment contrasts describe standardized spatial "
                "deployment and are retained unchanged."
            ),
            "raw_context_estimand": (
                "A is relative raw intensity normalized over each section's full "
                "called-peak panel, then area- and soft-domain-averaged."
            ),
            "caution": (
                "Raw values are section-compositional and descriptive; they are "
                "not absolute abundance, batch-corrected abundance, or evidence "
                "of biological replication."
            ),
        },
        "counts": {
            "contrast_rows": int(contrasts.shape[0]),
            "peak_rows": int(peaks.shape[0]),
            "robust_contrasts": int(contrasts["robust_contrast"].sum()),
            "primary_abundance_relevant_contrasts": int(
                contrasts["primary_abundance_relevant"].sum()
            ),
            "headline_high_domain_contrasts": int(
                contrasts["headline_high_domain"].sum()
            ),
            "gallery_eligible_contrasts": int(contrasts["gallery_eligible"].sum()),
            "robust_raw_below_average_both_contrasts": int(
                (contrasts["robust_contrast"] & contrasts["raw_below_average_both"]).sum()
            ),
            "depletion_only_contrasts": int(contrasts["depletion_only"].sum()),
            "peaks_with_robust_contrast": int(peaks["has_robust_contrast"].sum()),
            "peaks_with_primary_contrast": int(peaks["has_primary_contrast"].sum()),
            "peaks_with_headline_contrast": int(peaks["has_headline_contrast"].sum()),
            "tail_only_peaks": int(peaks["tail_only_peak"].sum()),
        },
    }
    return AbundanceContextResult(
        contrast_table=contrasts,
        peak_table=peaks,
        metadata=metadata,
    )


def _validate_thresholds(thresholds: AbundanceContextThresholds) -> None:
    numeric = {
        "effect_size": thresholds.effect_size,
        "conditional_qvalue": thresholds.conditional_qvalue,
        "sign_stability": thresholds.sign_stability,
        "abundance_fold": thresholds.abundance_fold,
    }
    if any(not np.isfinite(value) for value in numeric.values()):
        raise ValueError("All abundance-context thresholds must be finite")
    if thresholds.effect_size < 0.0:
        raise ValueError("effect_size must be nonnegative")
    if not 0.0 <= thresholds.conditional_qvalue <= 1.0:
        raise ValueError("conditional_qvalue must lie in [0, 1]")
    if not 0.0 <= thresholds.sign_stability <= 1.0:
        raise ValueError("sign_stability must lie in [0, 1]")
    if thresholds.abundance_fold <= 0.0:
        raise ValueError("abundance_fold must be positive")
    if thresholds.headline_top_n <= 0:
        raise ValueError("headline_top_n must be positive")


def _validate_input_tables(contrasts: pd.DataFrame, peaks: pd.DataFrame) -> None:
    contrast_columns = {
        "canonical_peak_id",
        "domain",
        "left_enrichment",
        "right_enrichment",
        "delta_enrichment",
        "sign_stability",
        "conditional_qvalue",
        "support_flag",
    }
    missing = sorted(contrast_columns.difference(contrasts.columns))
    if missing:
        raise ValueError(f"Contrast table lacks columns: {missing}")
    if "canonical_peak_id" not in peaks:
        raise ValueError("Peak summary lacks canonical_peak_id")
    if contrasts.empty or peaks.empty:
        raise ValueError("Contrast and peak-summary tables must be nonempty")
    contrasts["canonical_peak_id"] = contrasts["canonical_peak_id"].astype(str)
    peaks["canonical_peak_id"] = peaks["canonical_peak_id"].astype(str)
    if contrasts.duplicated(["canonical_peak_id", "domain"]).any():
        raise ValueError("Contrast peak-domain rows must be unique")
    if peaks["canonical_peak_id"].duplicated().any():
        raise ValueError("Peak summary canonical_peak_id values must be unique")
    unknown = sorted(
        set(contrasts["canonical_peak_id"]) - set(peaks["canonical_peak_id"])
    )
    if unknown:
        raise ValueError(f"Contrast peaks are absent from peak summary: {unknown[:10]}")
    numeric = [
        "left_enrichment",
        "right_enrichment",
        "delta_enrichment",
        "sign_stability",
        "conditional_qvalue",
    ]
    for column in numeric:
        values = pd.to_numeric(contrasts[column], errors="coerce").to_numpy()
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Contrast column {column!r} must be finite numeric data")


def _validate_raw_tensor(
    tensor: LabeledRawDomainTensor,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tensor.values, dtype=np.float64)
    sections = np.asarray(tensor.section_ids)
    domains = np.asarray(tensor.domain_ids)
    peaks = np.asarray(tensor.peak_ids).astype(str)
    weights = np.asarray(tensor.domain_weight_sums, dtype=np.float64)
    if values.ndim != 3 or values.shape != (
        sections.size,
        domains.size,
        peaks.size,
    ):
        raise ValueError(
            "Raw values must match section x domain x peak labeled axes"
        )
    if weights.shape != (sections.size, domains.size):
        raise ValueError("domain_weight_sums must match section x domain axes")
    _unique_lookup(sections, "section")
    _unique_lookup(domains, "domain")
    _unique_lookup(peaks, "peak")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Raw abundance values must be finite and nonnegative")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("domain_weight_sums must be finite and positive")
    if np.any(weights.sum(axis=1) <= 0.0):
        raise ValueError("Every section must have positive total domain weight")
    return values, weights


def _unique_lookup(values: Sequence[object], label: str) -> dict[object, int]:
    array = np.asarray(values)
    lookup: dict[object, int] = {}
    for position, raw_value in enumerate(array.tolist()):
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        if value in lookup:
            raise ValueError(f"Raw tensor {label} IDs must be unique")
        lookup[value] = position
    return lookup


def _lookup_positions(
    values: pd.Series, lookup: dict[object, int], label: str
) -> np.ndarray:
    positions: list[int] = []
    missing: list[object] = []
    for raw_value in values.tolist():
        value = raw_value.item() if isinstance(raw_value, np.generic) else raw_value
        if value not in lookup:
            missing.append(value)
        else:
            positions.append(int(lookup[value]))
    if missing:
        raise ValueError(f"{label} are absent from raw tensor: {missing[:10]}")
    return np.asarray(positions, dtype=np.int64)


def _descending_min_rank(values: np.ndarray) -> np.ndarray:
    """Rank domains within every section/peak, highest first, ties=min."""

    ranks = np.empty(values.shape, dtype=np.int64)
    for section in range(values.shape[0]):
        ranked = pd.DataFrame(values[section]).rank(
            axis=0, method="min", ascending=False, na_option="bottom"
        )
        ranks[section] = ranked.to_numpy(dtype=np.int64)
    return ranks


def _coerce_bool(values: pd.Series, label: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    lowered = values.astype(str).str.strip().str.lower()
    valid = lowered.isin(["true", "false"])
    if not valid.all():
        raise ValueError(f"{label} must contain only boolean values")
    return lowered.eq("true").to_numpy(dtype=bool)


def _summarize_peaks(
    contrasts: pd.DataFrame, peak_ids: pd.Series
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    grouped = {key: group for key, group in contrasts.groupby("canonical_peak_id")}
    for peak_id in peak_ids.astype(str):
        group = grouped.get(peak_id)
        if group is None:
            records.append({"canonical_peak_id": peak_id})
            continue
        robust_count = int(group["robust_contrast"].sum())
        primary_count = int(group["primary_abundance_relevant"].sum())
        headline_count = int(group["headline_high_domain"].sum())
        record: dict[str, object] = {
            "canonical_peak_id": peak_id,
            "robust_contrast_count": robust_count,
            "primary_abundance_relevant_contrast_count": primary_count,
            "headline_high_domain_contrast_count": headline_count,
            "robust_raw_below_average_both_contrast_count": int(
                (group["robust_contrast"] & group["raw_below_average_both"]).sum()
            ),
            "depletion_only_contrast_count": int(group["depletion_only"].sum()),
            "robust_raw_direction_concordant_contrast_count": int(
                (group["robust_contrast"] & group["raw_direction_concordant"]).sum()
            ),
            "robust_dominant_domain_corroborated_contrast_count": int(
                (
                    group["robust_contrast"]
                    & group["dominant_domain_corroborated"]
                ).sum()
            ),
            "has_robust_contrast": robust_count > 0,
            "has_primary_contrast": primary_count > 0,
            "has_headline_contrast": headline_count > 0,
            "tail_only_peak": robust_count > 0 and primary_count == 0,
            "depletion_only_peak": robust_count > 0
            and int(group["depletion_only"].sum()) == robust_count,
        }
        for prefix, flag in (
            ("selected_robust", "robust_contrast"),
            ("selected_primary", "primary_abundance_relevant"),
            ("selected_headline", "headline_high_domain"),
        ):
            selected = _strongest_eligible(group, flag)
            record[f"{prefix}_domain"] = (
                pd.NA if selected is None else selected["domain"]
            )
            record[f"{prefix}_delta_enrichment"] = (
                np.nan if selected is None else float(selected["delta_enrichment"])
            )
            record[f"{prefix}_absolute_delta_enrichment"] = (
                np.nan if selected is None else abs(float(selected["delta_enrichment"]))
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _strongest_eligible(group: pd.DataFrame, flag: str) -> pd.Series | None:
    eligible = group.loc[group[flag]].copy()
    if eligible.empty:
        return None
    eligible["_absolute_delta_enrichment"] = eligible["delta_enrichment"].abs()
    eligible.sort_values(
        ["_absolute_delta_enrichment", "_domain_axis_position"],
        ascending=[False, True],
        kind="stable",
        inplace=True,
    )
    return eligible.iloc[0]


def _fill_peak_defaults(peaks: pd.DataFrame) -> None:
    count_columns = [
        column
        for column in peaks.columns
        if column.endswith("_contrast_count")
    ]
    for column in count_columns:
        peaks[column] = peaks[column].fillna(0).astype(np.int64)
    bool_columns = [
        "has_robust_contrast",
        "has_primary_contrast",
        "has_headline_contrast",
        "tail_only_peak",
        "depletion_only_peak",
    ]
    for column in bool_columns:
        peaks[column] = peaks[column].fillna(False).astype(bool)
