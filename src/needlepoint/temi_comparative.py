"""Cross-dataset alignment for TEMI peak, prototype, and domain outputs."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class ComparativeDataset:
    name: str
    score_csv: Path
    model_metadata_csv: Path
    prototype_dir: Path
    lbm_dir: Path
    spot_pitch: float


@dataclass(frozen=True)
class PeakMatchSettings:
    tolerance_ppm: float = 10.0
    anchor_tolerance_ppm: float = 5.0
    minimum_anchor_count: int = 20
    recalibration: str = "median-ppm"

    def validate(self) -> None:
        if self.tolerance_ppm <= 0.0 or self.anchor_tolerance_ppm <= 0.0:
            raise ValueError("Peak matching tolerances must be positive")
        if self.minimum_anchor_count < 3:
            raise ValueError("minimum_anchor_count must be at least 3")
        if self.recalibration not in {"none", "median-ppm"}:
            raise ValueError("recalibration must be none or median-ppm")


def load_selected_peak_table(dataset: ComparativeDataset) -> pd.DataFrame:
    """Join fitted feature order to peak m/z and multiscale metadata."""

    scores = pd.read_csv(dataset.score_csv)
    metadata = pd.read_csv(dataset.model_metadata_csv)
    if "gene" not in metadata or "peak_index" not in scores or "peak_mz" not in scores:
        raise ValueError(f"Missing gene/peak_index/peak_mz columns for {dataset.name}")
    peak_index = metadata["gene"].astype(str).str.extract(r"(\d+)$")[0]
    if peak_index.isna().any():
        bad = metadata.loc[peak_index.isna(), "gene"].head().tolist()
        raise ValueError(f"Cannot parse peak indices from {dataset.name}: {bad}")
    selected = metadata[["gene", "feature_position"]].copy()
    selected["peak_index"] = peak_index.astype(np.int64)
    selected = selected.merge(scores, on="peak_index", how="left", validate="one_to_one")
    if selected["peak_mz"].isna().any():
        raise ValueError(f"Missing score metadata for selected peaks in {dataset.name}")
    selected["dataset"] = dataset.name
    return selected


def build_peak_crosswalk(
    left: pd.DataFrame,
    right: pd.DataFrame,
    settings: PeakMatchSettings | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Create an auditable one-to-one m/z crosswalk with optional shift correction."""

    settings = settings or PeakMatchSettings()
    settings.validate()
    left_mz = left["peak_mz"].to_numpy(dtype=np.float64)
    right_mz = right["peak_mz"].to_numpy(dtype=np.float64)
    shift_ppm, anchors = _estimate_median_ppm_shift(
        left_mz,
        right_mz,
        settings.anchor_tolerance_ppm,
    )
    if settings.recalibration == "none" or anchors < settings.minimum_anchor_count:
        applied_shift_ppm = 0.0
    else:
        applied_shift_ppm = shift_ppm
    calibrated_right_mz = right_mz / (1.0 + applied_shift_ppm * 1.0e-6)
    delta = calibrated_right_mz[None, :] - left_mz[:, None]
    delta_ppm = 1.0e6 * delta / np.maximum(left_mz[:, None], 1.0e-12)
    normalized_cost = np.abs(delta_ppm) / settings.tolerance_ppm
    assignment_cost = np.where(normalized_cost <= 1.0, normalized_cost, 2.0)
    left_rows, right_rows = linear_sum_assignment(assignment_cost)
    valid = assignment_cost[left_rows, right_rows] <= 1.0
    left_rows = left_rows[valid]
    right_rows = right_rows[valid]

    rows = []
    matched_left = set(left_rows.tolist())
    matched_right = set(right_rows.tolist())
    canonical = 0
    for left_row, right_row in sorted(zip(left_rows, right_rows), key=lambda pair: left_mz[pair[0]]):
        rows.append(
            _crosswalk_row(
                canonical,
                left.iloc[left_row],
                right.iloc[right_row],
                calibrated_right_mz[right_row],
                applied_shift_ppm,
                "shared",
            )
        )
        canonical += 1
    for left_row in sorted(set(range(left.shape[0])) - matched_left, key=lambda row: left_mz[row]):
        rows.append(
            _crosswalk_row(
                canonical,
                left.iloc[left_row],
                None,
                np.nan,
                applied_shift_ppm,
                "left_only",
            )
        )
        canonical += 1
    for right_row in sorted(
        set(range(right.shape[0])) - matched_right,
        key=lambda row: calibrated_right_mz[row],
    ):
        rows.append(
            _crosswalk_row(
                canonical,
                None,
                right.iloc[right_row],
                calibrated_right_mz[right_row],
                applied_shift_ppm,
                "right_only",
            )
        )
        canonical += 1
    crosswalk = pd.DataFrame(rows)
    summary = {
        "settings": asdict(settings),
        "estimated_right_minus_left_shift_ppm": float(shift_ppm),
        "applied_right_minus_left_shift_ppm": float(applied_shift_ppm),
        "n_mutual_nearest_anchors": int(anchors),
        "n_left_selected": int(left.shape[0]),
        "n_right_selected": int(right.shape[0]),
        "n_shared_one_to_one": int(valid.sum()),
        "n_left_only": int(left.shape[0] - valid.sum()),
        "n_right_only": int(right.shape[0] - valid.sum()),
    }
    return crosswalk, summary


def build_shared_peak_scores(
    crosswalk: pd.DataFrame,
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    shared = crosswalk.loc[crosswalk["match_status"] == "shared"].copy()
    left_lookup = left.set_index("peak_index")
    right_lookup = right.set_index("peak_index")
    common_columns = [
        column
        for column in left.columns.intersection(right.columns)
        if column
        not in {
            "dataset",
            "gene",
            "feature_position",
            "peak_index",
            "peak_bin",
        }
        and pd.api.types.is_numeric_dtype(left[column])
        and pd.api.types.is_numeric_dtype(right[column])
    ]
    output = shared[
        [
            "canonical_peak_id",
            "left_peak_index",
            "right_peak_index",
            "left_peak_mz",
            "right_peak_mz",
            "delta_ppm_after_calibration",
        ]
    ].copy()
    for column in common_columns:
        output[f"{left_name}_{column}"] = left_lookup.loc[
            shared["left_peak_index"].astype(int), column
        ].to_numpy()
        output[f"{right_name}_{column}"] = right_lookup.loc[
            shared["right_peak_index"].astype(int), column
        ].to_numpy()
    return output


def build_shared_peak_rank_comparison(
    crosswalk: pd.DataFrame,
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    """Compare within-dataset score percentiles without equating raw intensity scales."""

    shared = crosswalk.loc[crosswalk["match_status"] == "shared"].copy()
    output = shared[
        [
            "canonical_peak_id",
            "left_peak_index",
            "right_peak_index",
            "left_peak_mz",
            "right_peak_mz",
        ]
    ].copy()
    rank_columns = {}
    for column in _comparable_numeric_columns(left, right):
        left_rank = pd.Series(
            left[column].rank(method="average", pct=True).to_numpy(),
            index=left["peak_index"],
        )
        right_rank = pd.Series(
            right[column].rank(method="average", pct=True).to_numpy(),
            index=right["peak_index"],
        )
        left_values = left_rank.loc[shared["left_peak_index"].astype(int)].to_numpy()
        right_values = right_rank.loc[shared["right_peak_index"].astype(int)].to_numpy()
        rank_columns[f"{left_name}_{column}_percentile"] = left_values
        rank_columns[f"{right_name}_{column}_percentile"] = right_values
        rank_columns[f"{column}_percentile_difference"] = right_values - left_values
    return pd.concat([output, pd.DataFrame(rank_columns, index=output.index)], axis=1)


def summarize_paired_peak_scores(
    shared_scores: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    """Report Pearson and rank correlation for every paired numeric peak score."""

    left_prefix = f"{left_name}_"
    right_prefix = f"{right_name}_"
    rows = []
    for left_column in shared_scores.columns:
        if not left_column.startswith(left_prefix):
            continue
        score = left_column[len(left_prefix) :]
        right_column = f"{right_prefix}{score}"
        if right_column not in shared_scores:
            continue
        paired = shared_scores[[left_column, right_column]].dropna()
        if paired.shape[0] < 3:
            continue
        left_values = paired[left_column]
        right_values = paired[right_column]
        left_std = float(left_values.std())
        right_std = float(right_values.std())
        has_variation = left_std > 0.0 and right_std > 0.0
        rows.append(
            {
                "score": score,
                "n_shared_peaks": int(paired.shape[0]),
                "pearson_correlation": float(left_values.corr(right_values, method="pearson"))
                if has_variation
                else float("nan"),
                "spearman_correlation": float(left_values.corr(right_values, method="spearman"))
                if has_variation
                else float("nan"),
                "left_standard_deviation": left_std,
                "right_standard_deviation": right_std,
            }
        )
    return pd.DataFrame(rows)


def match_prototypes(
    crosswalk: pd.DataFrame,
    left_dir: Path,
    right_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Match prototype families by soft membership over shared canonical peaks."""

    left_profiles, right_profiles = _aligned_prototype_profiles(crosswalk, left_dir, right_dir)
    similarity = _cosine_cross_similarity(left_profiles, right_profiles)
    left_rows, right_rows = linear_sum_assignment(-similarity)
    rows = [
        {
            "left_prototype": int(left_row),
            "right_prototype": int(right_row),
            "shared_peak_membership_cosine": float(similarity[left_row, right_row]),
        }
        for left_row, right_row in zip(left_rows, right_rows)
    ]
    matched_left = set(left_rows.tolist())
    matched_right = set(right_rows.tolist())
    rows.extend(
        {
            "left_prototype": int(index),
            "right_prototype": np.nan,
            "shared_peak_membership_cosine": np.nan,
        }
        for index in sorted(set(range(similarity.shape[0])) - matched_left)
    )
    rows.extend(
        {
            "left_prototype": np.nan,
            "right_prototype": int(index),
            "shared_peak_membership_cosine": np.nan,
        }
        for index in sorted(set(range(similarity.shape[1])) - matched_right)
    )
    return pd.DataFrame(rows), similarity


def match_domains(
    crosswalk: pd.DataFrame,
    left_lbm_dir: Path,
    right_lbm_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Match domains by expected per-peak block signatures over shared peaks."""

    left_signature, left_peak_index = _domain_peak_signatures(left_lbm_dir)
    right_signature, right_peak_index = _domain_peak_signatures(right_lbm_dir)
    shared = crosswalk.loc[crosswalk["match_status"] == "shared"]
    left_rows = pd.Series(np.arange(left_peak_index.size), index=left_peak_index).loc[
        shared["left_peak_index"].astype(int)
    ]
    right_rows = pd.Series(np.arange(right_peak_index.size), index=right_peak_index).loc[
        shared["right_peak_index"].astype(int)
    ]
    left_aligned = left_signature[left_rows.to_numpy(dtype=np.int64)]
    right_aligned = right_signature[right_rows.to_numpy(dtype=np.int64)]
    similarity = _correlation_cross_similarity(left_aligned, right_aligned)
    left_domains, right_domains = linear_sum_assignment(-similarity)
    rows = [
        {
            "left_domain": int(left_domain),
            "right_domain": int(right_domain),
            "shared_peak_signature_correlation": float(
                similarity[left_domain, right_domain]
            ),
        }
        for left_domain, right_domain in zip(left_domains, right_domains)
    ]
    return pd.DataFrame(rows), similarity


def common_scale_policy(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_pitch: float,
    right_pitch: float,
) -> dict:
    pattern = re.compile(r"scale_score_(\d+(?:\.\d+)?)um$")
    left_scales = {float(match.group(1)) for column in left for match in [pattern.match(column)] if match}
    right_scales = {float(match.group(1)) for column in right for match in [pattern.match(column)] if match}
    common = sorted(left_scales & right_scales)
    minimum_shared = 2.0 * max(float(left_pitch), float(right_pitch))
    return {
        "left_spot_pitch_coordinate_units": float(left_pitch),
        "right_spot_pitch_coordinate_units": float(right_pitch),
        "minimum_shared_comparison_scale": minimum_shared,
        "all_common_scale_labels": common,
        "recommended_shared_scales": [scale for scale in common if scale >= minimum_shared],
        "fine_scale_policy": (
            "Scales below twice the coarser spot pitch remain dataset-specific and are not "
            "used for direct cross-dataset claims. Source columns retain their historical 'um' "
            "names, but this report treats them as calibrated coordinate-scale labels unless "
            "physical units are independently verified."
        ),
    }


def write_comparative_manifest(
    output_dir: Path,
    left: ComparativeDataset,
    right: ComparativeDataset,
    peak_summary: dict,
    scale_policy: dict,
) -> None:
    payload = {
        "left_dataset": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(left).items()},
        "right_dataset": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(right).items()},
        "peak_matching": peak_summary,
        "resolution_policy": scale_policy,
        "interpretation": {
            "peak_crosswalk": "chemical-axis correspondence only; not metabolite identification",
            "prototype_crosswalk": "soft peak-family correspondence over one-to-one shared peaks",
            "domain_crosswalk": "molecular-signature correspondence; labels are not spatial registration",
        },
    }
    (Path(output_dir) / "comparative_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _comparable_numeric_columns(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    excluded = {
        "feature_position",
        "peak_index",
        "peak_bin",
        "left_bin_inclusive",
        "right_bin_exclusive",
        "left_mz",
        "right_mz_inclusive",
        "peak_mz",
    }
    return [
        column
        for column in left.columns.intersection(right.columns)
        if column not in excluded
        and pd.api.types.is_numeric_dtype(left[column])
        and pd.api.types.is_numeric_dtype(right[column])
    ]


def _estimate_median_ppm_shift(
    left_mz: np.ndarray,
    right_mz: np.ndarray,
    tolerance_ppm: float,
) -> tuple[float, int]:
    distance = np.abs(left_mz[:, None] - right_mz[None, :])
    nearest_right = distance.argmin(axis=1)
    nearest_left = distance.argmin(axis=0)
    anchors = []
    for left_row, right_row in enumerate(nearest_right):
        if nearest_left[right_row] != left_row:
            continue
        ppm = 1.0e6 * (right_mz[right_row] - left_mz[left_row]) / left_mz[left_row]
        if abs(ppm) <= tolerance_ppm:
            anchors.append(ppm)
    return (float(np.median(anchors)) if anchors else 0.0), len(anchors)


def _crosswalk_row(
    canonical: int,
    left: pd.Series | None,
    right: pd.Series | None,
    calibrated_right_mz: float,
    shift_ppm: float,
    status: str,
) -> dict:
    left_mz = np.nan if left is None else float(left["peak_mz"])
    right_mz = np.nan if right is None else float(right["peak_mz"])
    delta_ppm = (
        np.nan
        if left is None or right is None
        else 1.0e6 * (calibrated_right_mz - left_mz) / left_mz
    )
    return {
        "canonical_peak_id": f"CP{canonical:05d}",
        "match_status": status,
        "left_peak_index": np.nan if left is None else int(left["peak_index"]),
        "left_gene": None if left is None else str(left["gene"]),
        "left_feature_position": np.nan if left is None else int(left["feature_position"]),
        "left_peak_mz": left_mz,
        "right_peak_index": np.nan if right is None else int(right["peak_index"]),
        "right_gene": None if right is None else str(right["gene"]),
        "right_feature_position": np.nan if right is None else int(right["feature_position"]),
        "right_peak_mz": right_mz,
        "right_peak_mz_calibrated": calibrated_right_mz,
        "delta_mz_after_calibration": calibrated_right_mz - left_mz
        if left is not None and right is not None
        else np.nan,
        "delta_ppm_after_calibration": delta_ppm,
        "applied_right_minus_left_shift_ppm": shift_ppm,
    }


def _aligned_prototype_profiles(crosswalk, left_dir, right_dir):
    shared = crosswalk.loc[crosswalk["match_status"] == "shared"]
    left_assignments = pd.read_csv(Path(left_dir) / "prototype_assignments.csv")
    right_assignments = pd.read_csv(Path(right_dir) / "prototype_assignments.csv")
    left_index = left_assignments["gene"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    right_index = right_assignments["gene"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    left_gamma = np.load(Path(left_dir) / "prototype_responsibilities.npy")
    right_gamma = np.load(Path(right_dir) / "prototype_responsibilities.npy")
    left_rows = pd.Series(np.arange(left_index.size), index=left_index).loc[
        shared["left_peak_index"].astype(int)
    ]
    right_rows = pd.Series(np.arange(right_index.size), index=right_index).loc[
        shared["right_peak_index"].astype(int)
    ]
    return left_gamma[left_rows].T, right_gamma[right_rows].T


def _domain_peak_signatures(lbm_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    lbm_dir = Path(lbm_dir)
    peak_table = pd.read_csv(lbm_dir / "vski_lbm_peak_modules.csv")
    peak_index = peak_table["gene"].astype(str).str.extract(r"(\d+)$")[0].astype(int).to_numpy()
    responsibilities = np.load(lbm_dir / "vski_lbm_peak_module_probabilities.npy")
    blocks = pd.read_csv(lbm_dir / "vski_lbm_block_parameters.csv")
    n_modules = responsibilities.shape[1]
    n_domains = int(blocks["domain"].max()) + 1
    means = np.zeros((n_modules, n_domains), dtype=np.float64)
    means[blocks["peak_module"].astype(int), blocks["domain"].astype(int)] = blocks[
        "block_mean"
    ]
    return responsibilities @ means, peak_index


def _cosine_cross_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1.0e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1.0e-12)
    return np.nan_to_num(left_norm @ right_norm.T, nan=0.0)


def _correlation_cross_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_z = (left - left.mean(axis=0, keepdims=True)) / np.maximum(
        left.std(axis=0, keepdims=True), 1.0e-12
    )
    right_z = (right - right.mean(axis=0, keepdims=True)) / np.maximum(
        right.std(axis=0, keepdims=True), 1.0e-12
    )
    return np.nan_to_num(left_z.T @ right_z / max(left.shape[0], 1), nan=0.0)
