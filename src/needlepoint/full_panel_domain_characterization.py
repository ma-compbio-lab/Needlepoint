"""Frozen-domain characterization of complete fitted TEMI peak panels.

The accepted joint-domain responsibilities are immutable inputs. Shared peaks
receive a two-section interaction analysis, whereas section-specific peaks are
summarized only within their observed section and projected onto association
programs learned from shared peaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from .domain_peak_differential import (
    DifferentialSettings,
    DomainPeakResult,
    SingleSectionDomainPeakResult,
    analyze_domain_conditioned_peaks,
    analyze_single_section_domain_peaks,
)
from .multidataset_lbm import reciprocal_nearest_peak_pairs


EPS = 1.0e-12
ALLOWED_MATCH_STATUS = {"shared", "left_only", "right_only"}
ALLOWED_TIERS = {
    "domain_defining_shared",
    "domain_characterizing_shared",
    "left_only",
    "right_only",
}


@dataclass(frozen=True)
class FullPanelSettings:
    """Settings that extend the fixed-domain differential configuration."""

    differential: DifferentialSettings = DifferentialSettings()
    map_standardization: str = "zscore"
    module_assignment_minimum_cosine: float = 0.6
    novel_pattern_threshold: float = 0.4
    n_domain_programs: int | None = None
    mass_series_deltas: tuple[float, ...] = (1.003355, 14.015650, 28.031300)
    mass_difference_tolerance_ppm: float = 10.0
    composition_dominance_fraction: float = 2.0 / 3.0

    def validate(self) -> None:
        self.differential.validate()
        if self.map_standardization not in {"zscore", "none"}:
            raise ValueError("map_standardization must be zscore or none")
        if not -1.0 <= self.module_assignment_minimum_cosine <= 1.0:
            raise ValueError("module_assignment_minimum_cosine must be in [-1, 1]")
        if not -1.0 <= self.novel_pattern_threshold <= 1.0:
            raise ValueError("novel_pattern_threshold must be in [-1, 1]")
        if self.n_domain_programs is not None and self.n_domain_programs < 2:
            raise ValueError("n_domain_programs must be at least 2")
        if self.mass_difference_tolerance_ppm <= 0:
            raise ValueError("mass_difference_tolerance_ppm must be positive")
        if any(value <= 0 for value in self.mass_series_deltas):
            raise ValueError("mass_series_deltas must be positive")
        if not 0.5 <= self.composition_dominance_fraction <= 1.0:
            raise ValueError("composition_dominance_fraction must be in [0.5, 1]")


@dataclass
class FullPanelResult:
    """Tables and arrays written by the full-panel runner."""

    inventory: pd.DataFrame
    shared_result: DomainPeakResult
    section_specific_results: dict[str, SingleSectionDomainPeakResult]
    shared_validation_summary: pd.DataFrame
    reference_behavior_module_transfer: pd.DataFrame
    domain_program_assignments: pd.DataFrame
    domain_program_summary: pd.DataFrame
    domain_program_centroids: pd.DataFrame
    section_specific_characterization: pd.DataFrame
    section_specific_program_assignments: pd.DataFrame
    full_panel_behavior_modules: pd.DataFrame
    composition_decomposition: pd.DataFrame
    candidate_mass_relationships: pd.DataFrame
    candidate_mass_series_groups: pd.DataFrame
    prioritized_tables: dict[str, pd.DataFrame]
    analysis_metadata: dict


def validate_expanded_crosswalk(
    expanded: pd.DataFrame,
    defining: pd.DataFrame,
    *,
    left_model_genes: Sequence[str] | None = None,
    right_model_genes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Validate evidence tiers and exact provenance of domain-defining peaks."""

    expanded = _normalize_crosswalk_columns(expanded).copy()
    defining = _normalize_crosswalk_columns(defining).copy()
    required = {
        "expanded_feature_id",
        "match_status",
        "feature_evidence_tier",
        "left_gene",
        "left_peak_index",
        "left_feature_position",
        "left_peak_mz",
        "right_gene",
        "right_peak_index",
        "right_feature_position",
        "right_peak_mz",
        "canonical_peak_id",
        "was_used_for_joint_domain_fit",
    }
    missing = sorted(required.difference(expanded.columns))
    if missing:
        raise ValueError(f"Expanded crosswalk lacks columns: {missing}")
    if expanded["expanded_feature_id"].isna().any() or expanded[
        "expanded_feature_id"
    ].astype(str).duplicated().any():
        raise ValueError("expanded_feature_id must be complete and unique")
    unknown_status = set(expanded["match_status"].astype(str)) - ALLOWED_MATCH_STATUS
    unknown_tiers = set(expanded["feature_evidence_tier"].astype(str)) - ALLOWED_TIERS
    if unknown_status:
        raise ValueError(f"Unknown match_status values: {sorted(unknown_status)}")
    if unknown_tiers:
        raise ValueError(f"Unknown feature_evidence_tier values: {sorted(unknown_tiers)}")
    expected_tier_status = {
        "domain_defining_shared": "shared",
        "domain_characterizing_shared": "shared",
        "left_only": "left_only",
        "right_only": "right_only",
    }
    invalid = expanded.apply(
        lambda row: expected_tier_status[str(row["feature_evidence_tier"])]
        != str(row["match_status"]),
        axis=1,
    )
    if invalid.any():
        raise ValueError("Evidence tier and match_status disagree")
    used = _as_bool(expanded["was_used_for_joint_domain_fit"])
    if not np.array_equal(
        used,
        expanded["feature_evidence_tier"].astype(str).eq("domain_defining_shared").to_numpy(),
    ):
        raise ValueError("was_used_for_joint_domain_fit disagrees with the defining tier")

    defining_required = {
        "canonical_peak_id",
        "left_gene",
        "right_gene",
        "left_peak_index",
        "right_peak_index",
    }
    missing_defining = sorted(defining_required.difference(defining.columns))
    if missing_defining:
        raise ValueError(f"Domain-defining crosswalk lacks columns: {missing_defining}")
    defining_rows = expanded.loc[used].copy()
    if defining_rows.shape[0] != defining.shape[0]:
        raise ValueError("Expanded crosswalk does not contain every defining peak exactly once")
    merged = defining.merge(
        defining_rows,
        on="canonical_peak_id",
        how="left",
        suffixes=("_reference", "_expanded"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise ValueError("A domain-defining canonical peak is absent from expanded crosswalk")
    for column in ("left_gene", "right_gene", "left_peak_index", "right_peak_index"):
        left_values = merged[f"{column}_reference"]
        right_values = merged[f"{column}_expanded"]
        if column.endswith("peak_index"):
            matches = np.allclose(
                pd.to_numeric(left_values).to_numpy(dtype=np.float64),
                pd.to_numeric(right_values).to_numpy(dtype=np.float64),
                equal_nan=True,
            )
        else:
            matches = left_values.astype(str).equals(right_values.astype(str))
        if not matches:
            raise ValueError(f"Defining feature identity mismatch in {column}")

    if left_model_genes is not None:
        available = set(map(str, left_model_genes))
        expected = expanded["left_gene"].notna()
        observed = expanded["left_gene"].astype(str).isin(available) & expected
        missing_genes = expanded.loc[expected & ~observed, "left_gene"].astype(str).tolist()
        if missing_genes:
            raise ValueError(f"Left fitted models are missing features: {missing_genes[:10]}")
        expanded["left_model_available"] = observed
    else:
        expanded["left_model_available"] = expanded["left_gene"].notna()
    if right_model_genes is not None:
        available = set(map(str, right_model_genes))
        expected = expanded["right_gene"].notna()
        observed = expanded["right_gene"].astype(str).isin(available) & expected
        missing_genes = expanded.loc[expected & ~observed, "right_gene"].astype(str).tolist()
        if missing_genes:
            raise ValueError(f"Right fitted models are missing features: {missing_genes[:10]}")
        expanded["right_model_available"] = observed
    else:
        expanded["right_model_available"] = expanded["right_gene"].notna()
    expanded["analysis_status"] = "included"
    expanded["exclusion_reason"] = ""
    return expanded


def build_expanded_peak_crosswalk(
    left_fitted: pd.DataFrame,
    right_fitted: pd.DataFrame,
    defining_crosswalk: pd.DataFrame,
    *,
    tolerance_ppm: float = 10.0,
    mz_min: float = 400.0,
    mz_max: float = 1200.0,
) -> pd.DataFrame:
    """Build a complete fitted-panel inventory from reciprocal m/z matches.

    Existing defining pairs retain their canonical IDs. Additional reciprocal
    pairs are explicitly held out from domain construction; every remaining
    fitted model becomes a section-only feature.
    """

    defining = _normalize_crosswalk_columns(defining_crosswalk)
    pairs = reciprocal_nearest_peak_pairs(
        left_fitted,
        right_fitted,
        tolerance_ppm=tolerance_ppm,
        mz_min=mz_min,
        mz_max=mz_max,
    )
    defining_lookup = {
        (str(row.left_gene), str(row.right_gene)): str(row.canonical_peak_id)
        for row in defining.itertuples(index=False)
    }
    records: list[dict] = []
    used_left: set[str] = set()
    used_right: set[str] = set()
    characterizing_index = 0
    for row in pairs.itertuples(index=False):
        key = (str(row.left_gene), str(row.right_gene))
        canonical = defining_lookup.get(key)
        if canonical is None:
            expanded_id = f"HC{characterizing_index:05d}"
            characterizing_index += 1
            tier = "domain_characterizing_shared"
            used_for_fit = False
            canonical = ""
        else:
            expanded_id = canonical
            tier = "domain_defining_shared"
            used_for_fit = True
        records.append(
            {
                "expanded_feature_id": expanded_id,
                "match_status": "shared",
                "feature_evidence_tier": tier,
                "left_gene": str(row.left_gene),
                "left_peak_index": int(row.left_peak_index),
                "left_feature_position": int(row.left_source_feature_position),
                "left_peak_mz": float(row.left_peak_mz),
                "right_gene": str(row.right_gene),
                "right_peak_index": int(row.right_peak_index),
                "right_feature_position": int(row.right_source_feature_position),
                "right_peak_mz": float(row.right_peak_mz),
                "canonical_peak_id": canonical,
                "was_used_for_joint_domain_fit": used_for_fit,
                "delta_mz": float(row.delta_mz),
                "delta_ppm": float(row.delta_ppm),
                "absolute_delta_ppm": float(row.absolute_delta_ppm),
            }
        )
        used_left.add(str(row.left_gene))
        used_right.add(str(row.right_gene))
    missing_defining = set(defining_lookup.values()) - {
        record["canonical_peak_id"] for record in records if record["canonical_peak_id"]
    }
    if missing_defining:
        raise ValueError(
            "Expanded matching lost domain-defining pairs: "
            + ", ".join(sorted(missing_defining)[:10])
        )
    for side, fitted, used in (
        ("left", left_fitted, used_left),
        ("right", right_fitted, used_right),
    ):
        remaining = fitted.loc[~fitted["gene"].astype(str).isin(used)].sort_values(
            "feature_position", kind="stable"
        )
        for index, row in enumerate(remaining.itertuples(index=False)):
            record = {
                "expanded_feature_id": f"{side[0].upper()}O{index:05d}",
                "match_status": f"{side}_only",
                "feature_evidence_tier": f"{side}_only",
                "left_gene": np.nan,
                "left_peak_index": np.nan,
                "left_feature_position": np.nan,
                "left_peak_mz": np.nan,
                "right_gene": np.nan,
                "right_peak_index": np.nan,
                "right_feature_position": np.nan,
                "right_peak_mz": np.nan,
                "canonical_peak_id": "",
                "was_used_for_joint_domain_fit": False,
                "delta_mz": np.nan,
                "delta_ppm": np.nan,
                "absolute_delta_ppm": np.nan,
            }
            record[f"{side}_gene"] = str(row.gene)
            record[f"{side}_peak_index"] = int(row.peak_index)
            record[f"{side}_feature_position"] = int(row.source_feature_position)
            record[f"{side}_peak_mz"] = float(row.peak_mz)
            records.append(record)
    output = pd.DataFrame(records)
    return validate_expanded_crosswalk(output, defining)


def selected_model_table(
    model_metadata: pd.DataFrame,
    inventory: pd.DataFrame,
    side: str,
    vski_dir: Path,
) -> pd.DataFrame:
    """Create a VSKIMapStore-compatible, inventory-ordered model table."""

    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    gene_column = f"{side}_gene"
    requested = inventory.loc[inventory[gene_column].notna(), gene_column].astype(str)
    metadata = model_metadata.copy()
    if "gene" not in metadata or metadata["gene"].astype(str).duplicated().any():
        raise ValueError("Model metadata must contain unique gene identifiers")
    lookup = metadata.assign(gene=metadata["gene"].astype(str)).set_index("gene", drop=False)
    missing = requested.loc[~requested.isin(lookup.index)]
    if not missing.empty:
        raise ValueError(f"{side} model metadata lacks genes: {missing.head().tolist()}")
    selected = lookup.loc[requested].reset_index(drop=True).copy()
    selected["source_feature_position"] = selected["feature_position"].astype(np.int64)
    selected["feature_position"] = np.arange(selected.shape[0], dtype=np.int64)
    if "model_chunk_path" not in selected:
        if "map_chunk_file" not in selected:
            raise ValueError("Model metadata lacks model_chunk_path and map_chunk_file")
        selected["model_chunk_path"] = selected["map_chunk_file"].map(
            lambda value: str(Path(vski_dir) / str(value))
        )
    selected["selected"] = True
    return selected


def analyze_full_panel(
    *,
    shared_store,
    left_only_store,
    right_only_store,
    joint_spots: pd.DataFrame,
    domain_probabilities: np.ndarray,
    dataset_names: Sequence[str],
    spot_pitches: Sequence[float],
    inventory: pd.DataFrame,
    reference_differential_dir: Path,
    settings: FullPanelSettings,
) -> FullPanelResult:
    """Run all frozen-domain analyses from preconstructed streamed stores."""

    settings.validate()
    if len(dataset_names) != 2 or len(spot_pitches) != 2:
        raise ValueError("Full-panel analysis requires exactly two sections")
    shared_inventory = inventory.loc[inventory["match_status"].eq("shared")].reset_index(drop=True)
    shared_crosswalk = shared_inventory.copy()
    shared_crosswalk["canonical_peak_id"] = shared_inventory["expanded_feature_id"].astype(str)
    if not np.array_equal(shared_store.genes.astype(str), shared_crosswalk["canonical_peak_id"]):
        raise ValueError("Shared store feature order does not match expanded crosswalk")
    shared_result = analyze_domain_conditioned_peaks(
        shared_store,
        joint_spots,
        domain_probabilities,
        dataset_names,
        spot_pitches,
        shared_crosswalk,
        settings.differential,
    )
    _verify_reference_defining_subset(
        shared_result, shared_inventory, Path(reference_differential_dir)
    )

    section_results: dict[str, SingleSectionDomainPeakResult] = {}
    for side, name, pitch, store in (
        ("left", dataset_names[0], spot_pitches[0], left_only_store),
        ("right", dataset_names[1], spot_pitches[1], right_only_store),
    ):
        subset = inventory.loc[inventory["match_status"].eq(f"{side}_only")].reset_index(drop=True)
        if subset.empty:
            continue
        if store is None or store.n_features != subset.shape[0]:
            raise ValueError(f"{side}_only fitted-map store is missing or misaligned")
        mask = joint_spots["dataset"].astype(str).eq(str(name)).to_numpy()
        section_spots = joint_spots.loc[mask].reset_index(drop=True)
        section_results[side] = analyze_single_section_domain_peaks(
            store,
            section_spots,
            np.asarray(domain_probabilities[mask]),
            str(name),
            float(pitch),
            subset,
            settings.differential,
        )

    reference_transfer = transfer_reference_behavior_modules(
        shared_result,
        shared_inventory,
        Path(reference_differential_dir),
        settings.module_assignment_minimum_cosine,
    )
    program_assignments, program_centroids, program_meta = build_shared_domain_programs(
        shared_result,
        shared_inventory,
        settings.n_domain_programs,
        settings.differential.random_state,
    )
    program_summary = summarize_shared_domain_programs(
        shared_result, shared_inventory, program_assignments, program_centroids
    )
    section_characterization, section_assignments = characterize_section_specific_peaks(
        section_results,
        program_centroids,
        settings,
    )
    validation = summarize_reference_validation(
        shared_result,
        shared_inventory,
        settings.differential.effect_size_threshold,
        settings.differential.fdr_threshold,
    )
    if settings.map_standardization == "none":
        composition = classify_composition_decomposition(
            shared_result.composition_decomposition,
            settings.composition_dominance_fraction,
        )
    else:
        composition = shared_result.composition_decomposition.copy()
        composition["composition_absolute_fraction"] = np.nan
        composition["within_domain_absolute_fraction"] = np.nan
        composition["composition_class"] = "not_computed_for_standardized_maps"
        composition["classification_rule"] = "requires --map-standardization none"
    full_behavior = shared_result.peak_behavior_modules.merge(
        reference_transfer,
        left_on="canonical_peak_id",
        right_on="expanded_feature_id",
        how="left",
    ).merge(
        program_assignments[["expanded_feature_id", "domain_program"]],
        left_on="canonical_peak_id",
        right_on="expanded_feature_id",
        how="left",
        suffixes=("", "_program"),
    )
    relationships, mass_groups, mass_group_lookup = candidate_mass_relationships(
        inventory,
        full_behavior,
        section_assignments,
        settings.mass_series_deltas,
        settings.mass_difference_tolerance_ppm,
        differential_profiles={
            str(feature_id): shared_result.domain_delta[:, index]
            for index, feature_id in enumerate(shared_inventory["expanded_feature_id"])
        },
        domain_profiles={
            **{
                str(feature_id): 0.5
                * (
                    shared_result.domain_enrichment[0, :, index]
                    + shared_result.domain_enrichment[1, :, index]
                )
                for index, feature_id in enumerate(shared_inventory["expanded_feature_id"])
            },
            **{
                str(feature_id): result.domain_enrichment[:, index]
                for result in section_results.values()
                for index, feature_id in enumerate(result.feature_ids)
            },
        },
    )
    enriched_inventory = inventory.copy()
    enriched_inventory["candidate_mass_series_group"] = enriched_inventory[
        "expanded_feature_id"
    ].map(mass_group_lookup)
    if "annotation" not in enriched_inventory:
        enriched_inventory["annotation"] = ""
    rankings = build_candidate_rankings(
        shared_result,
        shared_inventory,
        reference_transfer,
        program_assignments,
        section_characterization,
        section_assignments,
        composition,
        mass_group_lookup,
        settings,
    )
    analysis_metadata = {
        "domain_program_selection": program_meta,
        "n_features_by_tier": enriched_inventory["feature_evidence_tier"]
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict(),
        "n_candidate_mass_relationships": int(relationships.shape[0]),
        "n_candidate_mass_series_groups": int(mass_groups.shape[0]),
    }
    return FullPanelResult(
        inventory=enriched_inventory,
        shared_result=shared_result,
        section_specific_results=section_results,
        shared_validation_summary=validation,
        reference_behavior_module_transfer=reference_transfer,
        domain_program_assignments=program_assignments,
        domain_program_summary=program_summary,
        domain_program_centroids=program_centroids,
        section_specific_characterization=section_characterization,
        section_specific_program_assignments=section_assignments,
        full_panel_behavior_modules=full_behavior,
        composition_decomposition=composition,
        candidate_mass_relationships=relationships,
        candidate_mass_series_groups=mass_groups,
        prioritized_tables=rankings,
        analysis_metadata=analysis_metadata,
    )


def transfer_profiles(
    profiles: np.ndarray,
    centroid_matrix: np.ndarray,
    centroid_ids: np.ndarray,
    minimum_cosine: float,
) -> pd.DataFrame:
    """Assign signed L2-normalized profiles to fixed centroids by cosine."""

    profiles = np.asarray(profiles, dtype=np.float64)
    centroids = np.asarray(centroid_matrix, dtype=np.float64)
    if profiles.ndim != 2 or centroids.ndim != 2 or profiles.shape[1] != centroids.shape[1]:
        raise ValueError("Profiles and centroids must have the same domain axis")
    p_norm = np.linalg.norm(profiles, axis=1)
    c_norm = np.linalg.norm(centroids, axis=1)
    normalized_p = profiles / np.maximum(p_norm[:, None], EPS)
    normalized_c = centroids / np.maximum(c_norm[:, None], EPS)
    cosine = normalized_p @ normalized_c.T
    nearest = np.argmax(cosine, axis=1)
    similarity = cosine[np.arange(profiles.shape[0]), nearest]
    assigned = np.asarray(centroid_ids)[nearest].astype(object)
    status = np.where(
        (p_norm > EPS) & (similarity >= minimum_cosine),
        "assigned",
        "unassigned_differential_pattern",
    )
    assigned[status != "assigned"] = -1
    return pd.DataFrame(
        {
            "assigned_id": assigned,
            "assignment_cosine": similarity,
            "assignment_status": status,
        }
    )


def transfer_reference_behavior_modules(
    shared_result: DomainPeakResult,
    shared_inventory: pd.DataFrame,
    reference_dir: Path,
    minimum_cosine: float,
) -> pd.DataFrame:
    """Project held-out shared differential profiles onto reference modules."""

    centroids = pd.read_csv(reference_dir / "behavior_module_centroids.csv")
    required = {"behavior_module", "domain", "centroid_delta_enrichment"}
    if not required.issubset(centroids.columns):
        raise ValueError("Reference behavior-module centroid table has an invalid schema")
    modules = np.sort(centroids["behavior_module"].unique())
    domains = shared_result.domain_ids
    matrix = np.full((modules.size, domains.size), np.nan, dtype=np.float64)
    for module_index, module in enumerate(modules):
        rows = centroids.loc[centroids["behavior_module"].eq(module)].set_index("domain")
        matrix[module_index] = rows.loc[domains, "centroid_delta_enrichment"].to_numpy()
    transferred = transfer_profiles(
        shared_result.domain_delta.T, matrix, modules, minimum_cosine
    ).rename(
        columns={
            "assigned_id": "reference_behavior_module",
            "assignment_cosine": "module_assignment_cosine",
            "assignment_status": "module_assignment_status",
        }
    )
    transferred.insert(0, "expanded_feature_id", shared_inventory["expanded_feature_id"].to_numpy())
    transferred.insert(1, "feature_evidence_tier", shared_inventory["feature_evidence_tier"].to_numpy())
    transferred.loc[
        transferred["feature_evidence_tier"].eq("domain_defining_shared"),
        "module_assignment_status",
    ] = "reference_domain_defining_peak"
    return transferred


def build_shared_domain_programs(
    shared_result: DomainPeakResult,
    shared_inventory: pd.DataFrame,
    requested_programs: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Cluster conserved domain-enrichment profiles of all shared peaks."""

    del random_state  # Average-linkage clustering is deterministic.
    consensus = 0.5 * (
        shared_result.domain_enrichment[0] + shared_result.domain_enrichment[1]
    )
    profiles = consensus.T
    norms = np.linalg.norm(profiles, axis=1)
    normalized = profiles / np.maximum(norms[:, None], EPS)
    informative = np.flatnonzero(norms > EPS)
    labels = np.full(profiles.shape[0], -1, dtype=np.int64)
    scores: dict[int, float] = {}
    if informative.size >= 3:
        if requested_programs is None:
            maximum = min(12, informative.size - 1)
            for k in range(2, maximum + 1):
                candidate = AgglomerativeClustering(
                    n_clusters=k, metric="cosine", linkage="average"
                ).fit_predict(normalized[informative])
                scores[k] = float(
                    silhouette_score(normalized[informative], candidate, metric="cosine")
                )
            selected_k = max(scores, key=scores.get)
        else:
            selected_k = min(int(requested_programs), informative.size)
        labels[informative] = AgglomerativeClustering(
            n_clusters=selected_k, metric="cosine", linkage="average"
        ).fit_predict(normalized[informative])
    elif informative.size:
        selected_k = 1
        labels[informative] = 0
    else:
        selected_k = 0

    assignment = shared_inventory[
        ["expanded_feature_id", "feature_evidence_tier", "left_peak_mz", "right_peak_mz"]
    ].copy()
    assignment["domain_program"] = labels
    assignment["consensus_profile_l2"] = norms
    centroid_records: list[dict] = []
    for program in sorted(set(labels) - {-1}):
        members = labels == program
        centroid = profiles[members].mean(axis=0)
        member_inventory = shared_inventory.loc[members]
        leading = np.argsort(np.abs(centroid))[::-1][:3]
        for domain, value in zip(shared_result.domain_ids, centroid):
            centroid_records.append(
                {
                    "program_id": int(program),
                    "domain": int(domain),
                    "centroid_enrichment": float(value),
                    "n_peaks": int(members.sum()),
                    "n_domain_defining_peaks": int(
                        member_inventory["feature_evidence_tier"]
                        .eq("domain_defining_shared")
                        .sum()
                    ),
                    "n_domain_characterizing_peaks": int(
                        member_inventory["feature_evidence_tier"]
                        .eq("domain_characterizing_shared")
                        .sum()
                    ),
                    "leading_domains": ";".join(f"D{int(d)}" for d in leading),
                }
            )
    metadata = {
        "selected_k": int(selected_k),
        "candidate_silhouette_scores": {str(key): value for key, value in scores.items()},
        "representation": "L2-normalized consensus domain-enrichment profile",
        "distance": "cosine",
        "linkage": "average",
    }
    return assignment, pd.DataFrame(centroid_records), metadata


def summarize_shared_domain_programs(
    shared_result: DomainPeakResult,
    shared_inventory: pd.DataFrame,
    assignments: pd.DataFrame,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Create one interpretable record per shared domain-association program."""

    if centroids.empty:
        return pd.DataFrame()
    consensus = 0.5 * (
        shared_result.domain_enrichment[0] + shared_result.domain_enrichment[1]
    ).T
    records = []
    for program in sorted(assignments["domain_program"].unique()):
        if int(program) < 0:
            continue
        member_mask = assignments["domain_program"].eq(program).to_numpy()
        member_ids = shared_inventory.loc[member_mask, "expanded_feature_id"].astype(str).to_numpy()
        centroid = (
            centroids.loc[centroids["program_id"].eq(program)]
            .sort_values("domain")["centroid_enrichment"]
            .to_numpy(dtype=np.float64)
        )
        projection = consensus[member_mask] @ centroid
        positive_order = np.argsort(projection)[::-1][:5]
        negative_order = np.argsort(projection)[:5]
        leading = np.argsort(np.abs(centroid))[::-1][:3]
        member_tiers = shared_inventory.loc[member_mask, "feature_evidence_tier"]
        records.append(
            {
                "program_id": int(program),
                "n_peaks": int(member_mask.sum()),
                "n_domain_defining_peaks": int(
                    member_tiers.eq("domain_defining_shared").sum()
                ),
                "n_domain_characterizing_peaks": int(
                    member_tiers.eq("domain_characterizing_shared").sum()
                ),
                "leading_domains": ";".join(f"D{int(value)}" for value in leading),
                "top_positive_peaks": ";".join(member_ids[positive_order]),
                "top_negative_peaks": ";".join(member_ids[negative_order]),
            }
        )
    return pd.DataFrame(records)


def characterize_section_specific_peaks(
    section_results: Mapping[str, SingleSectionDomainPeakResult],
    program_centroids: pd.DataFrame,
    settings: FullPanelSettings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project unmatched peaks onto conserved shared domain programs."""

    characterizations: list[pd.DataFrame] = []
    assignments: list[pd.DataFrame] = []
    if program_centroids.empty:
        centroid_ids = np.empty(0, dtype=np.int64)
        centroid_matrix = np.empty((0, 0), dtype=np.float64)
    else:
        centroid_ids = np.sort(program_centroids["program_id"].unique())
        domains = np.sort(program_centroids["domain"].unique())
        centroid_matrix = np.vstack(
            [
                program_centroids.loc[program_centroids["program_id"].eq(program)]
                .set_index("domain")
                .loc[domains, "centroid_enrichment"]
                .to_numpy(dtype=np.float64)
                for program in centroid_ids
            ]
        )
    for side, result in section_results.items():
        characterization = result.peak_characterization.copy()
        characterization["section_side"] = side
        characterizations.append(characterization)
        if centroid_ids.size:
            transfer = transfer_profiles(
                result.domain_enrichment.T,
                centroid_matrix,
                centroid_ids,
                settings.module_assignment_minimum_cosine,
            )
            transfer = transfer.rename(
                columns={
                    "assigned_id": "shared_domain_program",
                    "assignment_cosine": "program_assignment_cosine",
                    "assignment_status": "program_assignment_status",
                }
            )
        else:
            transfer = pd.DataFrame(
                {
                    "shared_domain_program": np.full(result.feature_ids.size, -1),
                    "program_assignment_cosine": np.full(result.feature_ids.size, np.nan),
                    "program_assignment_status": "novel_section_specific_domain_pattern",
                }
            )
        transfer.insert(0, "expanded_feature_id", result.feature_ids)
        transfer.insert(1, "section_side", side)
        low_support = characterization["low_support"].to_numpy(dtype=bool)
        similarity = transfer["program_assignment_cosine"].to_numpy(dtype=np.float64)
        labels = np.full(result.feature_ids.size, "section_specific_augmentation_of_shared_program", dtype=object)
        labels[
            transfer["program_assignment_status"].ne("assigned").to_numpy()
            | (similarity < settings.novel_pattern_threshold)
        ] = "novel_section_specific_domain_pattern"
        labels[low_support] = "low_support_section_specific_peak"
        transfer["interpretation_label"] = labels
        assignments.append(transfer)
    characterization_table = (
        pd.concat(characterizations, ignore_index=True) if characterizations else pd.DataFrame()
    )
    assignment_table = pd.concat(assignments, ignore_index=True) if assignments else pd.DataFrame()
    return characterization_table, assignment_table


def summarize_reference_validation(
    shared_result: DomainPeakResult,
    shared_inventory: pd.DataFrame,
    effect_threshold: float,
    fdr_threshold: float,
) -> pd.DataFrame:
    """Compare robust domain-effect prevalence between defining and held-out tiers."""

    contrasts = shared_result.peak_domain_contrasts.copy()
    if "expanded_feature_id" not in contrasts:
        contrasts["expanded_feature_id"] = contrasts["canonical_peak_id"].astype(str)
    if "feature_evidence_tier" not in contrasts:
        contrasts = contrasts.merge(
            shared_inventory[["expanded_feature_id", "feature_evidence_tier"]],
            on="expanded_feature_id",
            how="left",
            validate="many_to_one",
        )
    robust = (
        contrasts["support_flag"].astype(bool)
        & contrasts["conditional_qvalue"].le(fdr_threshold)
        & contrasts["sign_stability"].ge(0.95)
        & contrasts["delta_enrichment"].abs().ge(effect_threshold)
    )
    contrasts["robust_contrast"] = robust
    records: list[dict] = []
    for domain in shared_result.domain_ids:
        row: dict = {"summary_type": "domain", "domain": int(domain)}
        for tier, prefix in (
            ("domain_defining_shared", "defining"),
            ("domain_characterizing_shared", "characterizing"),
        ):
            tier_ids = shared_inventory.loc[
                shared_inventory["feature_evidence_tier"].eq(tier), "expanded_feature_id"
            ]
            block = contrasts.loc[
                contrasts["domain"].eq(domain)
                & contrasts["expanded_feature_id"].isin(tier_ids)
            ]
            robust_block = block.loc[block["robust_contrast"]]
            row[f"n_{prefix}_peaks"] = int(tier_ids.size)
            row[f"n_{prefix}_robust_contrasts"] = int(robust_block.shape[0])
            row[f"fraction_{prefix}_robust"] = float(
                robust_block.shape[0] / max(tier_ids.size, 1)
            )
            row[f"median_{prefix}_absolute_effect"] = float(
                block["delta_enrichment"].abs().median()
            ) if not block.empty else np.nan
            row[f"{prefix}_net_direction"] = float(
                np.sign(robust_block["delta_enrichment"]).sum()
            )
        reference_fraction = row["fraction_defining_robust"]
        validation_fraction = row["fraction_characterizing_robust"]
        if row["n_characterizing_peaks"] == 0:
            status = "insufficient_additional_peaks"
        elif reference_fraction <= EPS:
            status = "supported" if validation_fraction <= 0.01 else "partially_supported"
        else:
            same_direction = (
                row["defining_net_direction"] == 0
                or row["characterizing_net_direction"] == 0
                or np.sign(row["defining_net_direction"])
                == np.sign(row["characterizing_net_direction"])
            )
            ratio = validation_fraction / reference_fraction
            if ratio >= 0.75 and same_direction:
                status = "supported"
            elif ratio >= 0.25:
                status = "partially_supported"
            else:
                status = "not_supported"
        row["validation_status"] = status
        row["fraction_with_hard_assignment_direction_concordance"] = np.nan
        row["status_rule"] = (
            "supported: held-out robust-effect fraction >=75% of defining fraction and "
            "net direction agrees; partial: >=25%; otherwise not supported"
        )
        records.append(row)
    feature_summary = shared_result.peak_differential_summary.copy()
    if "expanded_feature_id" not in feature_summary:
        feature_summary["expanded_feature_id"] = feature_summary["canonical_peak_id"].astype(str)
    for tier, prefix in (
        ("domain_defining_shared", "defining"),
        ("domain_characterizing_shared", "characterizing"),
    ):
        tier_ids = shared_inventory.loc[
            shared_inventory["feature_evidence_tier"].eq(tier), "expanded_feature_id"
        ]
        tier_contrasts = contrasts.loc[contrasts["expanded_feature_id"].isin(tier_ids)]
        robust_tier = tier_contrasts.loc[tier_contrasts["robust_contrast"]]
        tier_features = feature_summary.loc[
            feature_summary["expanded_feature_id"].isin(tier_ids)
        ]
        records.append(
            {
                "summary_type": "evidence_tier",
                "domain": np.nan,
                "feature_evidence_tier": tier,
                "n_peaks": int(tier_ids.size),
                "n_robust_peak_domain_contrasts": int(robust_tier.shape[0]),
                "n_peaks_with_robust_contrast": int(
                    robust_tier["expanded_feature_id"].nunique()
                ),
                "n_preferred_domain_switches": int(
                    tier_features["preferred_domain_switch"].astype(bool).sum()
                ),
                "fraction_with_hard_assignment_direction_concordance": np.nan,
                "validation_status": "reported_by_domain_above",
                "status_rule": "tier total; hard-assignment concordance requires a paired sensitivity run",
            }
        )
    return pd.DataFrame(records)


def classify_composition_decomposition(
    decomposition: pd.DataFrame,
    dominance_fraction: float,
) -> pd.DataFrame:
    """Attach a transparent composition-versus-within-domain classification."""

    output = decomposition.copy()
    composition = output["composition_effect"].abs().to_numpy(dtype=np.float64)
    within = output["within_domain_effect"].abs().to_numpy(dtype=np.float64)
    total = composition + within
    composition_share = composition / np.maximum(total, EPS)
    within_share = within / np.maximum(total, EPS)
    labels = np.full(output.shape[0], "mixed", dtype=object)
    labels[total <= 1.0e-10] = "minimal_global_difference"
    labels[(total > 1.0e-10) & (composition_share >= dominance_fraction)] = "composition_dominant"
    labels[(total > 1.0e-10) & (within_share >= dominance_fraction)] = "within_domain_dominant"
    output["composition_absolute_fraction"] = composition_share
    output["within_domain_absolute_fraction"] = within_share
    output["composition_class"] = labels
    output["classification_rule"] = (
        f"dominant when absolute contribution fraction >= {dominance_fraction:.6g}"
    )
    return output


def candidate_mass_relationships(
    inventory: pd.DataFrame,
    shared_behavior: pd.DataFrame,
    section_assignments: pd.DataFrame,
    target_deltas: Sequence[float],
    tolerance_ppm: float,
    *,
    differential_profiles: Mapping[str, np.ndarray] | None = None,
    domain_profiles: Mapping[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Find candidate mass-difference edges without assigning identities."""

    features = inventory.copy()
    features["representative_mz"] = features[["left_peak_mz", "right_peak_mz"]].mean(
        axis=1, skipna=True
    )
    features = features.loc[np.isfinite(features["representative_mz"])].sort_values(
        "representative_mz", kind="stable"
    ).reset_index(drop=True)
    behavior_lookup = (
        shared_behavior.drop_duplicates("canonical_peak_id").set_index("canonical_peak_id")
        if not shared_behavior.empty
        else pd.DataFrame()
    )
    section_lookup = (
        section_assignments.drop_duplicates("expanded_feature_id").set_index("expanded_feature_id")
        if not section_assignments.empty
        else pd.DataFrame()
    )
    rows: list[dict] = []
    mz = features["representative_mz"].to_numpy(dtype=np.float64)
    for a in range(features.shape[0]):
        for target in target_deltas:
            desired = mz[a] + float(target)
            insertion = int(np.searchsorted(mz, desired))
            for b in range(max(a + 1, insertion - 2), min(features.shape[0], insertion + 3)):
                observed = mz[b] - mz[a]
                error_ppm = 1.0e6 * (observed - target) / target
                if abs(error_ppm) > tolerance_ppm:
                    continue
                feature_a = features.iloc[a]
                feature_b = features.iloc[b]
                id_a, id_b = str(feature_a["expanded_feature_id"]), str(feature_b["expanded_feature_id"])
                relation_type = _mass_relationship_label(float(target))
                differential_cosine = _profile_cosine(
                    None if differential_profiles is None else differential_profiles.get(id_a),
                    None if differential_profiles is None else differential_profiles.get(id_b),
                )
                domain_cosine = _profile_cosine(
                    None if domain_profiles is None else domain_profiles.get(id_a),
                    None if domain_profiles is None else domain_profiles.get(id_b),
                )
                rows.append(
                    {
                        "peak_a": id_a,
                        "peak_b": id_b,
                        "mz_a": mz[a],
                        "mz_b": mz[b],
                        "observed_delta_mz": observed,
                        "target_delta_mz": float(target),
                        "delta_error_ppm": error_ppm,
                        "feature_tier_a": feature_a["feature_evidence_tier"],
                        "feature_tier_b": feature_b["feature_evidence_tier"],
                        "behavior_module_a": _lookup_value(behavior_lookup, id_a, "behavior_module"),
                        "behavior_module_b": _lookup_value(behavior_lookup, id_b, "behavior_module"),
                        "domain_program_a": _program_lookup(behavior_lookup, section_lookup, id_a),
                        "domain_program_b": _program_lookup(behavior_lookup, section_lookup, id_b),
                        "differential_profile_cosine": differential_cosine,
                        "domain_enrichment_profile_cosine": domain_cosine,
                        "candidate_relationship_type": relation_type,
                        "interpretation_scope": "candidate mass relationship; not molecular identification",
                    }
                )
    relationships = pd.DataFrame(rows)
    if relationships.empty:
        return relationships, pd.DataFrame(), {}
    relationships = relationships.drop_duplicates(
        ["peak_a", "peak_b", "target_delta_mz"], keep="first"
    ).reset_index(drop=True)
    parent = {feature_id: feature_id for feature_id in features["expanded_feature_id"].astype(str)}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for row in relationships.itertuples(index=False):
        union(str(row.peak_a), str(row.peak_b))
    connected = sorted(
        ({find(str(a)) for a in relationships["peak_a"]}
         | {find(str(b)) for b in relationships["peak_b"]})
    )
    root_to_group = {root: index for index, root in enumerate(connected)}
    group_lookup = {
        feature_id: root_to_group[find(feature_id)]
        for feature_id in parent
        if find(feature_id) in root_to_group
    }
    relationships["candidate_mass_series_group"] = relationships["peak_a"].map(group_lookup)
    group_rows: list[dict] = []
    for group, members in pd.Series(group_lookup).groupby(lambda key: group_lookup[key]):
        member_ids = list(members.index)
        member_table = features.loc[features["expanded_feature_id"].astype(str).isin(member_ids)]
        edge_table = relationships.loc[relationships["candidate_mass_series_group"].eq(group)]
        group_rows.append(
            {
                "candidate_mass_series_group": int(group),
                "n_peaks": int(len(member_ids)),
                "mz_min": float(member_table["representative_mz"].min()),
                "mz_max": float(member_table["representative_mz"].max()),
                "feature_evidence_tiers": ";".join(
                    sorted(member_table["feature_evidence_tier"].astype(str).unique())
                ),
                "dominant_behavior_module": _mode_or_missing(
                    [_lookup_value(behavior_lookup, item, "behavior_module") for item in member_ids]
                ),
                "dominant_domain_association_program": _mode_or_missing(
                    [_program_lookup(behavior_lookup, section_lookup, item) for item in member_ids]
                ),
                "mean_profile_coherence": float(
                    edge_table["domain_enrichment_profile_cosine"].mean()
                ),
                "candidate_relationship_labels": ";".join(
                    sorted(edge_table["candidate_relationship_type"].unique())
                ),
                "available_annotations": "",
                "interpretation_scope": "candidate mass series; not molecular identification",
            }
        )
    return relationships, pd.DataFrame(group_rows), group_lookup


def build_candidate_rankings(
    shared_result: DomainPeakResult,
    shared_inventory: pd.DataFrame,
    reference_transfer: pd.DataFrame,
    program_assignments: pd.DataFrame,
    section_characterization: pd.DataFrame,
    section_assignments: pd.DataFrame,
    composition: pd.DataFrame,
    mass_group_lookup: Mapping[str, int],
    settings: FullPanelSettings,
) -> dict[str, pd.DataFrame]:
    """Build reproducible rankings while retaining every score component."""

    summary = shared_result.peak_differential_summary.copy()
    if "expanded_feature_id" in summary:
        summary = summary.drop(columns=["canonical_peak_id"], errors="ignore")
    else:
        summary = summary.rename(columns={"canonical_peak_id": "expanded_feature_id"})
    if "expanded_feature_id" not in summary:
        summary["expanded_feature_id"] = shared_inventory["expanded_feature_id"].to_numpy()
    summary = summary.merge(
        shared_inventory[["expanded_feature_id", "feature_evidence_tier", "left_peak_mz", "right_peak_mz"]],
        on="expanded_feature_id",
        how="left",
        suffixes=("", "_inventory"),
    ).merge(reference_transfer, on=["expanded_feature_id", "feature_evidence_tier"], how="left")
    summary = summary.merge(
        program_assignments[["expanded_feature_id", "domain_program"]],
        on="expanded_feature_id",
        how="left",
    )
    composition_columns = [
        "expanded_feature_id",
        "global_shift",
        "composition_effect",
        "within_domain_effect",
        "composition_class",
        "within_domain_absolute_fraction",
    ]
    comp = composition.copy()
    if "expanded_feature_id" in comp:
        comp = comp.drop(columns=["canonical_peak_id"], errors="ignore")
    else:
        comp = comp.rename(columns={"canonical_peak_id": "expanded_feature_id"})
    if "expanded_feature_id" in comp:
        summary = summary.merge(
            comp[[column for column in composition_columns if column in comp]],
            on="expanded_feature_id",
            how="left",
        )
    summary["candidate_mass_series_group"] = summary["expanded_feature_id"].map(mass_group_lookup)
    contrast = shared_result.peak_domain_contrasts.copy()
    if "expanded_feature_id" not in contrast:
        contrast["expanded_feature_id"] = contrast["canonical_peak_id"].astype(str)
    robust_contrast = (
        contrast["conditional_qvalue"].le(settings.differential.fdr_threshold)
        & contrast["sign_stability"].ge(0.95)
        & contrast["delta_enrichment"].abs().ge(
            settings.differential.effect_size_threshold
        )
        & contrast["support_flag"].astype(bool)
    )
    minimum_conditional_q = contrast.groupby("expanded_feature_id")[
        "conditional_qvalue"
    ].min()
    maximum_sign_stability = contrast.groupby("expanded_feature_id")[
        "sign_stability"
    ].max()
    n_robust = contrast.loc[robust_contrast].groupby("expanded_feature_id").size()
    summary["minimum_conditional_domain_qvalue"] = summary["expanded_feature_id"].map(
        minimum_conditional_q
    )
    summary["maximum_domain_sign_stability"] = summary["expanded_feature_id"].map(
        maximum_sign_stability
    )
    summary["n_robust_domain_contrasts"] = (
        summary["expanded_feature_id"].map(n_robust).fillna(0).astype(int)
    )
    summary["component_effect_amplitude"] = _percentile_score(
        summary["weighted_l2_deployment_shift"]
    )
    summary["component_conditional_qvalue"] = _percentile_score(
        -summary["minimum_conditional_domain_qvalue"].fillna(1.0)
    )
    summary["component_sign_stability"] = summary[
        "maximum_domain_sign_stability"
    ].fillna(0.0)
    summary["component_soft_hard_direction_concordance"] = np.nan
    summary["component_switch_stability"] = summary[
        "preferred_domain_switch_frequency"
    ].fillna(0.0)
    summary["component_within_domain_dominant"] = summary.get(
        "composition_class", pd.Series(index=summary.index, dtype=object)
    ).eq("within_domain_dominant").astype(float)
    summary["component_reference_module_support"] = summary[
        "module_assignment_status"
    ].eq("assigned").astype(float)
    summary["component_candidate_mass_series_support"] = summary[
        "candidate_mass_series_group"
    ].notna().astype(float)
    component_columns = [column for column in summary if column.startswith("component_")]
    score_matrix = summary[component_columns].fillna(0.5)
    summary["integrated_priority_score"] = score_matrix.mean(axis=1)
    summary["priority_score_rule"] = "unweighted mean of retained component columns"

    validation = summary.loc[
        summary["feature_evidence_tier"].eq("domain_characterizing_shared")
    ].sort_values(
        ["n_robust_domain_contrasts", "integrated_priority_score", "expanded_feature_id"],
        ascending=[False, False, True],
    )
    within = summary.loc[
        summary.get("composition_class", "").eq("within_domain_dominant")
    ].sort_values(["integrated_priority_score", "expanded_feature_id"], ascending=[False, True])
    switch = summary.loc[
        summary["preferred_domain_switch"].astype(bool)
        & summary["left_top_domain_margin"].gt(0)
        & summary["right_top_domain_margin"].gt(0)
        & summary["preferred_domain_switch_frequency"].ge(0.75)
    ].copy()
    if not switch.empty:
        signed = contrast.assign(
            robust=(
                contrast["conditional_qvalue"].le(settings.differential.fdr_threshold)
                & contrast["sign_stability"].ge(0.95)
                & contrast["delta_enrichment"].abs().ge(
                    settings.differential.effect_size_threshold
                )
            )
        )
        positive = signed.loc[signed["robust"] & signed["delta_enrichment"].gt(0)].groupby(
            "expanded_feature_id"
        ).size()
        negative = signed.loc[signed["robust"] & signed["delta_enrichment"].lt(0)].groupby(
            "expanded_feature_id"
        ).size()
        switch = switch.loc[
            switch["expanded_feature_id"].map(positive).fillna(0).gt(0)
            & switch["expanded_feature_id"].map(negative).fillna(0).gt(0)
        ]
    switch = switch.sort_values(
        ["integrated_priority_score", "expanded_feature_id"], ascending=[False, True]
    )

    if section_characterization.empty:
        section_ranked = pd.DataFrame()
    else:
        section_ranked = section_characterization.merge(
            section_assignments,
            on=["expanded_feature_id", "section_side"],
            how="left",
        )
        section_ranked["candidate_mass_series_group"] = section_ranked[
            "expanded_feature_id"
        ].map(mass_group_lookup)
        section_ranked["component_domain_specificity"] = _percentile_score(
            section_ranked["profile_specificity"]
        )
        section_ranked["component_spatial_block_stability"] = section_ranked[
            "preferred_domain_bootstrap_frequency"
        ].fillna(0.0)
        section_ranked["component_program_similarity"] = (
            section_ranked["program_assignment_cosine"].fillna(-1.0) + 1.0
        ) / 2.0
        section_ranked["component_candidate_mass_series_support"] = section_ranked[
            "candidate_mass_series_group"
        ].notna().astype(float)
        section_ranked["integrated_priority_score"] = section_ranked[
            [
                "component_domain_specificity",
                "component_spatial_block_stability",
                "component_program_similarity",
                "component_candidate_mass_series_support",
            ]
        ].mean(axis=1)
        section_ranked["priority_score_rule"] = "unweighted mean of retained component columns"
    augmentations = (
        section_ranked.loc[
            section_ranked["interpretation_label"].eq(
                "section_specific_augmentation_of_shared_program"
            )
        ].sort_values(["integrated_priority_score", "expanded_feature_id"], ascending=[False, True])
        if not section_ranked.empty
        else section_ranked
    )
    novel = (
        section_ranked.loc[
            section_ranked["interpretation_label"].eq("novel_section_specific_domain_pattern")
        ].copy()
        if not section_ranked.empty
        else section_ranked
    )
    if not novel.empty:
        novel["component_program_novelty"] = 1.0 - novel["component_program_similarity"]
        novel["integrated_priority_score"] = novel[
            [
                "component_domain_specificity",
                "component_spatial_block_stability",
                "component_program_novelty",
                "component_candidate_mass_series_support",
            ]
        ].mean(axis=1)
        novel = novel.sort_values(
            ["integrated_priority_score", "expanded_feature_id"], ascending=[False, True]
        )
    return {
        "shared_validation": validation.reset_index(drop=True),
        "within_domain": within.reset_index(drop=True),
        "domain_switch": switch.reset_index(drop=True),
        "section_specific_augmentation": augmentations.reset_index(drop=True),
        "novel_section_specific": novel.reset_index(drop=True),
    }


def _verify_reference_defining_subset(
    shared_result: DomainPeakResult,
    shared_inventory: pd.DataFrame,
    reference_dir: Path,
    tolerance: float = 1.0e-6,
) -> None:
    reference = pd.read_csv(reference_dir / "peak_domain_contrasts.csv.gz")
    observed = shared_result.peak_domain_contrasts.copy()
    mapping = shared_inventory.loc[
        shared_inventory["feature_evidence_tier"].eq("domain_defining_shared"),
        ["expanded_feature_id", "canonical_peak_id"],
    ]
    observed = observed.merge(
        mapping,
        left_on="canonical_peak_id",
        right_on="expanded_feature_id",
        how="inner",
        suffixes=("_expanded", "_reference"),
    )
    merged = reference.merge(
        observed,
        left_on=["canonical_peak_id", "domain"],
        right_on=["canonical_peak_id_reference", "domain"],
        how="left",
        suffixes=("_reference", "_expanded"),
        validate="one_to_one",
    )
    if merged["delta_enrichment_expanded"].isna().any():
        raise ValueError("Full-panel analysis did not reproduce every defining contrast")
    for column in ("left_enrichment", "right_enrichment", "delta_enrichment"):
        error = np.nanmax(
            np.abs(
                merged[f"{column}_reference"].to_numpy(dtype=np.float64)
                - merged[f"{column}_expanded"].to_numpy(dtype=np.float64)
            )
        )
        if error > tolerance:
            difference = np.abs(
                merged[f"{column}_reference"].to_numpy(dtype=np.float64)
                - merged[f"{column}_expanded"].to_numpy(dtype=np.float64)
            )
            worst = merged.iloc[int(np.nanargmax(difference))]
            raise ValueError(
                f"Defining-panel regression failed for {column}: max error {error:.3g}; "
                f"peak={worst['canonical_peak_id']} domain={int(worst['domain'])} "
                f"reference={worst[f'{column}_reference']:.6g} "
                f"expanded={worst[f'{column}_expanded']:.6g}"
            )


def _normalize_crosswalk_columns(table: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "left_source_feature_position": "left_feature_position",
        "right_source_feature_position": "right_feature_position",
    }
    output = table.copy()
    for source, target in aliases.items():
        if target not in output and source in output:
            output[target] = output[source]
    return output


def _as_bool(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.to_numpy(dtype=bool)
    return values.astype(str).str.lower().isin({"1", "true", "yes"}).to_numpy()


def _mass_relationship_label(target: float) -> str:
    if abs(target - 1.003355) < 1.0e-4:
        return "candidate isotope relationship"
    if abs(target - 14.015650) < 1.0e-4 or abs(target - 28.031300) < 1.0e-4:
        return "candidate homologous series"
    return "candidate configured mass relationship"


def _lookup_value(table: pd.DataFrame, feature_id: str, column: str):
    if table.empty or feature_id not in table.index or column not in table:
        return np.nan
    return table.at[feature_id, column]


def _program_lookup(shared: pd.DataFrame, section: pd.DataFrame, feature_id: str):
    value = _lookup_value(shared, feature_id, "domain_program")
    if pd.notna(value):
        return value
    return _lookup_value(section, feature_id, "shared_domain_program")


def _mode_or_missing(values: Sequence) -> object:
    clean = [value for value in values if pd.notna(value)]
    if not clean:
        return np.nan
    return stats.mode(np.asarray(clean), keepdims=False).mode.item()


def _percentile_score(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return values.rank(method="average", pct=True).fillna(0.0)


def _profile_cosine(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return np.nan
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        return np.nan
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > EPS else np.nan
