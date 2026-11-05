"""Domain-conditioned differential peak analysis for fixed joint TEMI domains.

The estimand is a section-by-domain interaction computed from fitted GP maps
and fixed joint-domain responsibilities. Spots are population elements, not
biological replicates; optional block-bootstrap results are therefore spatial
stability summaries conditional on the two observed sections.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


EPS = 1.0e-12


@dataclass(frozen=True)
class DifferentialSettings:
    """Configuration for domain-conditioned peak summaries."""

    assignment_mode: str = "soft"
    domain_confidence_threshold: float = 0.8
    minimum_domain_area: float = 0.0
    minimum_domain_fraction: float = 0.005
    distribution_sample_per_domain: int = 2000
    hotspot_z_threshold: float = 1.0
    n_bootstrap: int = 0
    spatial_block_size: float = 1000.0
    minimum_block_domain_spots: float = 5.0
    random_state: int = 0
    effect_size_threshold: float = 0.5
    fdr_threshold: float = 0.05
    n_behavior_modules: int | None = None

    def validate(self) -> None:
        if self.assignment_mode not in {"soft", "hard", "high-confidence"}:
            raise ValueError("assignment_mode must be soft, hard, or high-confidence")
        if not 0.0 <= self.domain_confidence_threshold <= 1.0:
            raise ValueError("domain_confidence_threshold must be in [0, 1]")
        if self.minimum_domain_area < 0 or self.minimum_domain_fraction < 0:
            raise ValueError("domain support thresholds must be nonnegative")
        if self.distribution_sample_per_domain < 0 or self.n_bootstrap < 0:
            raise ValueError("sample and bootstrap counts must be nonnegative")
        if self.spatial_block_size <= 0:
            raise ValueError("spatial_block_size must be positive")
        if self.minimum_block_domain_spots < 0:
            raise ValueError("minimum_block_domain_spots must be nonnegative")
        if self.n_behavior_modules is not None and self.n_behavior_modules < 2:
            raise ValueError("n_behavior_modules must be at least 2")


@dataclass
class DomainPeakResult:
    """Arrays and output tables produced by one analysis run."""

    dataset_ids: np.ndarray
    domain_ids: np.ndarray
    feature_ids: np.ndarray
    domain_mean: np.ndarray
    domain_enrichment: np.ndarray
    domain_delta: np.ndarray
    domain_variance: np.ndarray
    domain_posterior_variance: np.ndarray
    section_domain_peak_summary: pd.DataFrame
    peak_domain_contrasts: pd.DataFrame
    peak_differential_summary: pd.DataFrame
    peak_behavior_modules: pd.DataFrame
    behavior_module_centroids: pd.DataFrame
    domain_support_summary: pd.DataFrame
    spatial_block_bootstrap_summary: pd.DataFrame
    composition_decomposition: pd.DataFrame
    behavior_selection: dict


@dataclass
class SingleSectionDomainPeakResult:
    """Fixed-domain summaries for peaks fitted in only one section."""

    dataset_id: str
    domain_ids: np.ndarray
    feature_ids: np.ndarray
    domain_mean: np.ndarray
    domain_enrichment: np.ndarray
    domain_variance: np.ndarray
    domain_posterior_variance: np.ndarray
    section_domain_peak_summary: pd.DataFrame
    peak_characterization: pd.DataFrame
    spatial_block_stability: pd.DataFrame
    domain_support_summary: pd.DataFrame


@dataclass
class _StreamedStatistics:
    sum_weight: np.ndarray
    sum_weight_squared: np.ndarray
    soft_mass: np.ndarray
    spot_count: np.ndarray
    section_area: np.ndarray
    analyzed_area: np.ndarray
    sum_mean: np.ndarray
    sum_squared_mean: np.ndarray
    sum_posterior_variance: np.ndarray
    sum_weight_squared_posterior_variance: np.ndarray
    section_sum: np.ndarray
    distribution: dict[tuple[int, int], np.ndarray]
    distribution_sample_size: np.ndarray
    block_sum: list[np.ndarray]
    block_mass: list[np.ndarray]
    block_section_sum: list[np.ndarray]
    block_section_mass: list[np.ndarray]


def validate_joint_axes(
    store,
    joint_spots: pd.DataFrame,
    domain_probabilities: np.ndarray,
    dataset_names: Sequence[str],
) -> None:
    """Fail if the fixed joint result and fitted-map store are not identical."""

    required = {"dataset", "dataset_map_row", "spot_index"}
    missing = sorted(required.difference(joint_spots.columns))
    if missing:
        raise ValueError(f"Joint spot table lacks alignment columns: {missing}")
    if joint_spots.shape[0] != store.n_spots:
        raise ValueError(
            f"Joint spot rows ({joint_spots.shape[0]}) != map rows ({store.n_spots})"
        )
    if domain_probabilities.shape[0] != store.n_spots:
        raise ValueError("Domain-probability rows do not match fitted-map rows")
    if domain_probabilities.ndim != 2 or domain_probabilities.shape[1] < 2:
        raise ValueError("Domain probabilities must be an N x R array with R >= 2")
    if not np.all(np.isfinite(domain_probabilities)):
        raise ValueError("Domain probabilities contain non-finite values")
    if not np.allclose(domain_probabilities.sum(axis=1), 1.0, atol=1.0e-5):
        raise ValueError("Domain-probability rows do not sum to one")
    expected = store.spots.reset_index(drop=True)
    observed = joint_spots.reset_index(drop=True)
    for column in ("dataset", "dataset_map_row", "spot_index"):
        if not np.array_equal(expected[column].to_numpy(), observed[column].to_numpy()):
            raise ValueError(f"Joint spot alignment mismatch in column {column}")
    observed_names = observed["dataset"].drop_duplicates().astype(str).tolist()
    if observed_names != list(dataset_names):
        raise ValueError(
            f"Dataset order mismatch: joint table has {observed_names}, expected {list(dataset_names)}"
        )
    if store.n_features != len(store.genes) or len(np.unique(store.genes)) != store.n_features:
        raise ValueError("Fitted-map feature identifiers are incomplete or duplicated")


def assignment_probabilities(
    probabilities: np.ndarray,
    mode: str,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return analysis responsibilities and a retained-spot mask."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if mode == "soft":
        return probabilities, np.ones(probabilities.shape[0], dtype=bool)
    labels = probabilities.argmax(axis=1)
    retained = np.ones(probabilities.shape[0], dtype=bool)
    if mode == "high-confidence":
        retained = probabilities.max(axis=1) >= confidence_threshold
    elif mode != "hard":
        raise ValueError(f"Unknown assignment mode: {mode}")
    assigned = np.zeros_like(probabilities)
    assigned[np.arange(probabilities.shape[0])[retained], labels[retained]] = 1.0
    return assigned, retained


def analyze_domain_conditioned_peaks(
    store,
    joint_spots: pd.DataFrame,
    domain_probabilities: np.ndarray,
    dataset_names: Sequence[str],
    spot_pitches: Sequence[float],
    crosswalk: pd.DataFrame,
    settings: DifferentialSettings,
) -> DomainPeakResult:
    """Run the complete streamed analysis against fixed joint domains."""

    settings.validate()
    if len(dataset_names) != 2 or len(spot_pitches) != 2:
        raise ValueError("The current differential workflow requires exactly two sections")
    if any(float(value) <= 0 for value in spot_pitches):
        raise ValueError("Spot pitches must be positive")
    validate_joint_axes(store, joint_spots, domain_probabilities, dataset_names)
    if crosswalk.shape[0] != store.n_features:
        raise ValueError("Crosswalk row count does not match fitted-map feature count")
    if "canonical_peak_id" not in crosswalk:
        raise ValueError("Crosswalk lacks canonical_peak_id")
    if not np.array_equal(
        crosswalk["canonical_peak_id"].astype(str).to_numpy(), store.genes.astype(str)
    ):
        raise ValueError("Crosswalk order does not match fitted-map feature order")

    analysis_B, retained = assignment_probabilities(
        domain_probabilities,
        settings.assignment_mode,
        settings.domain_confidence_threshold,
    )
    stats_out = _stream_statistics(
        store,
        joint_spots,
        domain_probabilities,
        analysis_B,
        retained,
        dataset_names,
        spot_pitches,
        settings,
    )
    arrays = _derive_arrays(stats_out, settings)
    support = (
        (arrays["domain_area"] >= settings.minimum_domain_area)
        & (arrays["domain_fraction"] >= settings.minimum_domain_fraction)
    )
    shared_support = support[0] & support[1]
    omega = np.minimum(arrays["domain_fraction"][0], arrays["domain_fraction"][1])
    omega[~shared_support] = 0.0
    if omega.sum() > 0:
        omega /= omega.sum()

    bootstrap = _block_bootstrap(
        stats_out,
        arrays,
        support,
        omega,
        settings,
        spot_pitches,
    )
    feature_metrics = _feature_profile_metrics(
        arrays["enrichment"], arrays["delta"], support, omega, bootstrap, settings
    )
    decomposition = _composition_decomposition(arrays)
    behavior, centroids, selection = _cluster_behavior_profiles(
        arrays["delta"], shared_support, feature_metrics, store.genes, settings
    )
    section_table = _section_domain_peak_table(
        stats_out, arrays, support, crosswalk, dataset_names, store.genes, settings
    )
    contrast_table = _peak_domain_contrast_table(
        arrays, support, bootstrap, crosswalk, store.genes
    )
    feature_table = _feature_summary_table(
        feature_metrics, decomposition, behavior, crosswalk, store.genes
    )
    support_table = _domain_support_table(
        stats_out, arrays, support, dataset_names, retained, joint_spots
    )
    bootstrap_table = _bootstrap_table(bootstrap, store.genes, arrays["domain_ids"])
    decomposition_table = _decomposition_table(decomposition, crosswalk, store.genes)

    return DomainPeakResult(
        dataset_ids=np.asarray(dataset_names, dtype=str),
        domain_ids=arrays["domain_ids"],
        feature_ids=np.asarray(store.genes, dtype=str),
        domain_mean=arrays["mean"],
        domain_enrichment=arrays["enrichment"],
        domain_delta=arrays["delta"],
        domain_variance=arrays["variance"],
        domain_posterior_variance=arrays["posterior_variance"],
        section_domain_peak_summary=section_table,
        peak_domain_contrasts=contrast_table,
        peak_differential_summary=feature_table,
        peak_behavior_modules=behavior,
        behavior_module_centroids=centroids,
        domain_support_summary=support_table,
        spatial_block_bootstrap_summary=bootstrap_table,
        composition_decomposition=decomposition_table,
        behavior_selection=selection,
    )


def analyze_single_section_domain_peaks(
    store,
    section_spots: pd.DataFrame,
    domain_probabilities: np.ndarray,
    dataset_name: str,
    spot_pitch: float,
    feature_metadata: pd.DataFrame,
    settings: DifferentialSettings,
) -> SingleSectionDomainPeakResult:
    """Characterize one-section peaks against fixed joint-domain probabilities.

    This is intentionally not a differential test. It uses the same streamed
    moments and spatial blocks as :func:`analyze_domain_conditioned_peaks`, but
    reports only within-section enrichment and its conditional stability.
    """

    settings.validate()
    if float(spot_pitch) <= 0:
        raise ValueError("spot_pitch must be positive")
    if section_spots.shape[0] != store.n_spots:
        raise ValueError("Section spot rows do not match fitted-map rows")
    if domain_probabilities.shape[0] != store.n_spots:
        raise ValueError("Section domain-probability rows do not match fitted-map rows")
    if domain_probabilities.ndim != 2 or domain_probabilities.shape[1] < 2:
        raise ValueError("Domain probabilities must be an N x R array with R >= 2")
    if not np.all(np.isfinite(domain_probabilities)) or not np.allclose(
        domain_probabilities.sum(axis=1), 1.0, atol=1.0e-5
    ):
        raise ValueError("Domain probabilities must be finite and row-normalized")
    if feature_metadata.shape[0] != store.n_features:
        raise ValueError("Feature metadata row count does not match fitted-map features")
    if "expanded_feature_id" not in feature_metadata:
        raise ValueError("Feature metadata lacks expanded_feature_id")
    feature_ids = feature_metadata["expanded_feature_id"].astype(str).to_numpy()
    if len(np.unique(feature_ids)) != feature_ids.size:
        raise ValueError("expanded_feature_id values must be unique")

    spots = section_spots.reset_index(drop=True).copy()
    spots["dataset"] = str(dataset_name)
    if "dataset_map_row" in spots and not np.array_equal(
        spots["dataset_map_row"].to_numpy(dtype=np.int64),
        np.arange(store.n_spots, dtype=np.int64),
    ):
        raise ValueError("Section spot rows are not in fitted-map order")
    for column in ("spot_index", "x", "y"):
        if column in spots and column in store.spots:
            observed_column = (
                f"native_{column}"
                if column in {"x", "y"} and f"native_{column}" in spots
                else column
            )
            if not np.allclose(
                spots[observed_column].to_numpy(dtype=np.float64),
                store.spots[column].to_numpy(dtype=np.float64),
                equal_nan=True,
            ):
                raise ValueError(
                    f"Section spot alignment mismatch in {observed_column}/{column}"
                )

    analysis_B, retained = assignment_probabilities(
        domain_probabilities,
        settings.assignment_mode,
        settings.domain_confidence_threshold,
    )
    streamed = _stream_statistics(
        store,
        spots,
        domain_probabilities,
        analysis_B,
        retained,
        [dataset_name],
        [spot_pitch],
        settings,
    )
    arrays = _derive_arrays(streamed, settings)
    support = (
        (arrays["domain_area"] >= settings.minimum_domain_area)
        & (arrays["domain_fraction"] >= settings.minimum_domain_fraction)
    )
    stability = _single_section_block_stability(
        streamed, arrays, support[0], settings, float(spot_pitch)
    )
    table_metadata = feature_metadata.drop(columns=["canonical_peak_id"], errors="ignore").rename(
        columns={"expanded_feature_id": "canonical_peak_id"}
    )
    section_table = _section_domain_peak_table(
        streamed,
        arrays,
        support,
        table_metadata,
        [dataset_name],
        feature_ids,
        settings,
    )
    section_table["expanded_feature_id"] = section_table.pop("canonical_peak_id")
    characterization = _single_section_feature_table(
        arrays,
        support[0],
        stability,
        feature_metadata,
        feature_ids,
    )
    stability_table = _single_section_stability_table(
        stability, feature_ids, arrays["domain_ids"]
    )
    support_table = _domain_support_table(
        streamed, arrays, support, [dataset_name], retained, spots
    )
    return SingleSectionDomainPeakResult(
        dataset_id=str(dataset_name),
        domain_ids=arrays["domain_ids"],
        feature_ids=feature_ids,
        domain_mean=arrays["mean"][0],
        domain_enrichment=arrays["enrichment"][0],
        domain_variance=arrays["variance"][0],
        domain_posterior_variance=arrays["posterior_variance"][0],
        section_domain_peak_summary=section_table,
        peak_characterization=characterization,
        spatial_block_stability=stability_table,
        domain_support_summary=support_table,
    )


def _stream_statistics(
    store,
    spots: pd.DataFrame,
    raw_B: np.ndarray,
    analysis_B: np.ndarray,
    retained: np.ndarray,
    dataset_names: Sequence[str],
    spot_pitches: Sequence[float],
    settings: DifferentialSettings,
) -> _StreamedStatistics:
    d_count, r_count, p_count = len(dataset_names), raw_B.shape[1], store.n_features
    shape_dr = (d_count, r_count)
    shape_drp = (d_count, r_count, p_count)
    sum_weight = np.zeros(shape_dr, dtype=np.float64)
    sum_weight_squared = np.zeros(shape_dr, dtype=np.float64)
    soft_mass = np.zeros(shape_dr, dtype=np.float64)
    spot_count = np.zeros(shape_dr, dtype=np.int64)
    section_area = np.zeros(d_count, dtype=np.float64)
    analyzed_area = np.zeros(d_count, dtype=np.float64)
    sum_mean = np.zeros(shape_drp, dtype=np.float64)
    sum_squared_mean = np.zeros(shape_drp, dtype=np.float64)
    sum_u = np.zeros(shape_drp, dtype=np.float64)
    sum_w2_u = np.zeros(shape_drp, dtype=np.float64)
    section_sum = np.zeros((d_count, p_count), dtype=np.float64)

    dataset_rows: list[np.ndarray] = []
    dataset_q: list[np.ndarray] = []
    dataset_area: list[np.ndarray] = []
    block_designs = []
    samples: dict[tuple[int, int], np.ndarray] = {}
    sample_size = np.zeros(shape_dr, dtype=np.int64)
    rng = np.random.default_rng(settings.random_state)

    for d, (name, pitch) in enumerate(zip(dataset_names, spot_pitches)):
        rows = np.flatnonzero(spots["dataset"].astype(str).to_numpy() == name)
        if rows.size == 0:
            raise ValueError(f"No joint spots found for dataset {name}")
        q = analysis_B[rows]
        area = np.full(rows.size, float(pitch) ** 2, dtype=np.float64)
        dataset_rows.append(rows)
        dataset_q.append(q)
        dataset_area.append(area)
        weighted = q * area[:, None]
        sum_weight[d] = weighted.sum(axis=0)
        sum_weight_squared[d] = np.square(weighted).sum(axis=0)
        soft_mass[d] = raw_B[rows].sum(axis=0)
        labels = raw_B[rows].argmax(axis=1)
        kept_local = retained[rows]
        spot_count[d] = np.bincount(labels[kept_local], minlength=r_count)
        section_area[d] = area.sum()
        analyzed_area[d] = area[kept_local].sum()
        block_designs.append(
            _make_block_design(spots.iloc[rows], q, area, float(pitch), settings)
            if settings.n_bootstrap > 0
            else None
        )
        if settings.distribution_sample_per_domain > 0:
            for r in range(r_count):
                probability = q[:, r]
                if probability.sum() <= EPS:
                    samples[(d, r)] = np.empty(0, dtype=np.int64)
                    continue
                count = settings.distribution_sample_per_domain
                local = rng.choice(
                    rows.size,
                    size=count,
                    replace=True,
                    p=probability / probability.sum(),
                )
                samples[(d, r)] = rows[local]
                sample_size[d, r] = count

    distribution: dict[tuple[int, int], np.ndarray] = {
        key: np.full((indices.size, p_count), np.nan, dtype=np.float32)
        for key, indices in samples.items()
    }
    block_sum = [
        np.zeros((0, r_count, p_count), dtype=np.float64)
        if design is None
        else np.zeros((design["n_blocks"], r_count, p_count), dtype=np.float64)
        for design in block_designs
    ]
    block_section_sum = [
        np.zeros((0, p_count), dtype=np.float64)
        if design is None
        else np.zeros((design["n_blocks"], p_count), dtype=np.float64)
        for design in block_designs
    ]

    covered = np.zeros((d_count, p_count), dtype=bool)
    for block_rows, mean, variance, positions, genes in store.iter_blocks(
        include_variance=True
    ):
        positions = np.asarray(positions, dtype=np.int64)
        if not np.array_equal(np.asarray(genes).astype(str), store.genes[positions].astype(str)):
            raise ValueError("Fitted-map iterator feature identifiers are misaligned")
        for d, rows in enumerate(dataset_rows):
            mask = np.isin(block_rows, rows, assume_unique=True)
            if not mask.any():
                continue
            if covered[d, positions].any():
                raise ValueError(
                    f"Fitted-map iterator emitted duplicate feature positions for dataset {d}"
                )
            covered[d, positions] = True
            observed_rows = np.asarray(block_rows)[mask]
            expected_rows = rows
            if not np.array_equal(observed_rows, expected_rows):
                raise ValueError("Map blocks do not preserve complete dataset row order")
            values = np.asarray(mean[mask], dtype=np.float64)
            q = dataset_q[d]
            area = dataset_area[d]
            weighted = q * area[:, None]
            sum_mean[d][:, positions] = weighted.T @ values
            sum_squared_mean[d][:, positions] = weighted.T @ np.square(values)
            section_sum[d, positions] = (area[:, None] * values).sum(axis=0)
            if variance is not None:
                u = np.asarray(variance[mask], dtype=np.float64)
                sum_u[d][:, positions] = weighted.T @ u
                sum_w2_u[d][:, positions] = np.square(weighted).T @ u
            design = block_designs[d]
            if design is not None:
                aggregated = np.asarray(design["domain_matrix"] @ values)
                block_sum[d][:, :, positions] = aggregated.reshape(
                    design["n_blocks"], r_count, positions.size
                )
                block_section_sum[d][:, positions] = np.asarray(
                    design["section_matrix"] @ values
                )
            for r in range(r_count):
                selected = samples.get((d, r))
                if selected is None or selected.size == 0:
                    continue
                local = selected - rows[0]
                if local.min(initial=0) < 0 or local.max(initial=-1) >= rows.size:
                    raise ValueError("Distribution sample rows are outside their dataset block")
                distribution[(d, r)][:, positions] = values[local]
    if not covered.all():
        missing = np.argwhere(~covered)[:10].tolist()
        raise ValueError(f"Map iterator omitted dataset/feature positions {missing}")

    return _StreamedStatistics(
        sum_weight=sum_weight,
        sum_weight_squared=sum_weight_squared,
        soft_mass=soft_mass,
        spot_count=spot_count,
        section_area=section_area,
        analyzed_area=analyzed_area,
        sum_mean=sum_mean,
        sum_squared_mean=sum_squared_mean,
        sum_posterior_variance=sum_u,
        sum_weight_squared_posterior_variance=sum_w2_u,
        section_sum=section_sum,
        distribution=distribution,
        distribution_sample_size=sample_size,
        block_sum=block_sum,
        block_mass=[
            np.zeros((0, r_count), dtype=np.float64)
            if design is None
            else design["domain_mass"]
            for design in block_designs
        ],
        block_section_sum=block_section_sum,
        block_section_mass=[
            np.zeros(0, dtype=np.float64)
            if design is None
            else design["section_mass"]
            for design in block_designs
        ],
    )


def _make_block_design(
    spots: pd.DataFrame,
    q: np.ndarray,
    area: np.ndarray,
    pitch: float,
    settings: DifferentialSettings,
) -> dict:
    x_column = "native_x" if "native_x" in spots else "x"
    y_column = "native_y" if "native_y" in spots else "y"
    x = spots[x_column].to_numpy(dtype=np.float64) * pitch
    y = spots[y_column].to_numpy(dtype=np.float64) * pitch
    bx = np.floor((x - x.min()) / settings.spatial_block_size).astype(np.int64)
    by = np.floor((y - y.min()) / settings.spatial_block_size).astype(np.int64)
    pairs = np.column_stack([bx, by])
    _, block = np.unique(pairs, axis=0, return_inverse=True)
    n_blocks = int(block.max()) + 1
    n, r_count = q.shape
    domain_rows = np.repeat(block, r_count) * r_count + np.tile(np.arange(r_count), n)
    domain_columns = np.repeat(np.arange(n), r_count)
    domain_data = (q * area[:, None]).reshape(-1).astype(np.float64)
    domain_matrix = sparse.csr_matrix(
        (domain_data, (domain_rows, domain_columns)), shape=(n_blocks * r_count, n)
    )
    section_matrix = sparse.csr_matrix(
        (area, (block, np.arange(n))), shape=(n_blocks, n)
    )
    return {
        "n_blocks": n_blocks,
        "domain_matrix": domain_matrix,
        "section_matrix": section_matrix,
        "domain_mass": np.asarray(domain_matrix.sum(axis=1)).reshape(n_blocks, r_count),
        "section_mass": np.asarray(section_matrix.sum(axis=1)).ravel(),
    }


def _derive_arrays(streamed: _StreamedStatistics, settings: DifferentialSettings) -> dict:
    denominator = np.maximum(streamed.sum_weight[:, :, None], EPS)
    mean = streamed.sum_mean / denominator
    variance = np.maximum(streamed.sum_squared_mean / denominator - np.square(mean), 0.0)
    posterior = streamed.sum_posterior_variance / denominator
    posterior_mean_variance = (
        streamed.sum_weight_squared_posterior_variance / np.square(denominator)
    )
    section_mean = streamed.section_sum / np.maximum(streamed.section_area[:, None], EPS)
    enrichment = mean - section_mean[:, None, :]
    domain_area = streamed.sum_weight
    domain_fraction = domain_area / np.maximum(streamed.section_area[:, None], EPS)
    effective_support = np.square(streamed.sum_weight) / np.maximum(
        streamed.sum_weight_squared, EPS
    )
    delta = enrichment[1] - enrichment[0] if enrichment.shape[0] == 2 else None
    return {
        "mean": mean,
        "variance": variance,
        "posterior_variance": posterior,
        "posterior_mean_variance": posterior_mean_variance,
        "section_mean": section_mean,
        "enrichment": enrichment,
        "delta": delta,
        "domain_area": domain_area,
        "domain_fraction": domain_fraction,
        "effective_support": effective_support,
        "domain_ids": np.arange(mean.shape[1], dtype=np.int64),
    }


def _single_section_block_stability(
    streamed: _StreamedStatistics,
    arrays: dict,
    support: np.ndarray,
    settings: DifferentialSettings,
    spot_pitch: float,
) -> dict[str, np.ndarray]:
    """Bootstrap one-section enrichment over the existing spatial blocks."""

    r_count, p_count = arrays["enrichment"].shape[1:]
    empty = {
        "median": np.full((r_count, p_count), np.nan),
        "ci_lower": np.full((r_count, p_count), np.nan),
        "ci_upper": np.full((r_count, p_count), np.nan),
        "sign_stability": np.full((r_count, p_count), np.nan),
        "preferred_domain_frequency": np.full(p_count, np.nan),
    }
    if settings.n_bootstrap <= 0:
        return empty
    sums = streamed.block_sum[0]
    mass = streamed.block_mass[0]
    section_sums = streamed.block_section_sum[0]
    section_mass = streamed.block_section_mass[0]
    if sums.shape[0] == 0:
        return empty
    rng = np.random.default_rng(settings.random_state + 130363)
    draws = np.full((settings.n_bootstrap, r_count, p_count), np.nan)
    preferred = np.full((settings.n_bootstrap, p_count), -1, dtype=np.int64)
    valid_domains = np.flatnonzero(support)
    valid_block_domain = mass >= settings.minimum_block_domain_spots * spot_pitch**2
    for bootstrap_index in range(settings.n_bootstrap):
        counts = rng.multinomial(
            sums.shape[0], np.full(sums.shape[0], 1.0 / sums.shape[0])
        )
        selected_mass = np.tensordot(
            counts, mass * valid_block_domain, axes=(0, 0)
        )
        selected_sum = np.tensordot(
            counts, sums * valid_block_domain[:, :, None], axes=(0, 0)
        )
        theta = selected_sum / np.maximum(selected_mass[:, None], EPS)
        selected_section_sum = np.tensordot(counts, section_sums, axes=(0, 0))
        selected_section_mass = np.tensordot(counts, section_mass, axes=(0, 0))
        enrichment = theta - selected_section_sum[None, :] / max(
            selected_section_mass, EPS
        )
        draws[bootstrap_index] = enrichment
        if valid_domains.size:
            preferred[bootstrap_index] = valid_domains[
                np.argmax(enrichment[valid_domains], axis=0)
            ]
    observed = arrays["enrichment"][0]
    observed_preferred = np.full(p_count, -1, dtype=np.int64)
    if valid_domains.size:
        observed_preferred = valid_domains[np.argmax(observed[valid_domains], axis=0)]
    return {
        "median": np.nanmedian(draws, axis=0),
        "ci_lower": np.nanquantile(draws, 0.025, axis=0),
        "ci_upper": np.nanquantile(draws, 0.975, axis=0),
        "sign_stability": np.nanmean(np.sign(draws) == np.sign(observed)[None], axis=0),
        "preferred_domain_frequency": np.mean(
            preferred == observed_preferred[None, :], axis=0
        ),
    }


def _single_section_feature_table(
    arrays: dict,
    support: np.ndarray,
    stability: dict[str, np.ndarray],
    metadata: pd.DataFrame,
    feature_ids: np.ndarray,
) -> pd.DataFrame:
    records = []
    valid = np.flatnonzero(support)
    for p, feature_id in enumerate(feature_ids):
        profile = arrays["enrichment"][0, :, p]
        if valid.size:
            preferred = int(valid[np.argmax(profile[valid])])
            ordered = np.sort(profile[valid])
            margin = float(ordered[-1] - ordered[-2]) if valid.size > 1 else np.nan
            sign_stability = float(stability["sign_stability"][preferred, p])
            posterior = float(
                np.average(
                    arrays["posterior_variance"][0, valid, p],
                    weights=arrays["domain_fraction"][0, valid],
                )
            )
            effective_support = float(np.min(arrays["effective_support"][0, valid]))
        else:
            preferred, margin = -1, np.nan
            sign_stability = posterior = effective_support = np.nan
        record = metadata.iloc[p].to_dict()
        record.update(
            {
                "expanded_feature_id": str(feature_id),
                "preferred_domain": preferred,
                "top_domain_margin": margin,
                "profile_specificity": _profile_specificity(profile, support),
                "domain_profile_variance": float(np.var(profile[valid])) if valid.size else np.nan,
                "mean_posterior_variance": posterior,
                "minimum_effective_weighted_support": effective_support,
                "preferred_domain_bootstrap_frequency": stability[
                    "preferred_domain_frequency"
                ][p],
                "preferred_domain_sign_stability": sign_stability,
                "n_supported_domains": int(valid.size),
                "low_support": bool(valid.size < 2),
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def _single_section_stability_table(
    stability: dict[str, np.ndarray],
    feature_ids: np.ndarray,
    domain_ids: np.ndarray,
) -> pd.DataFrame:
    records = []
    for domain in domain_ids:
        for p, feature_id in enumerate(feature_ids):
            records.append(
                {
                    "expanded_feature_id": str(feature_id),
                    "domain": int(domain),
                    "bootstrap_median_enrichment": stability["median"][domain, p],
                    "bootstrap_ci_lower": stability["ci_lower"][domain, p],
                    "bootstrap_ci_upper": stability["ci_upper"][domain, p],
                    "sign_stability_probability": stability["sign_stability"][domain, p],
                    "preferred_domain_bootstrap_frequency": stability[
                        "preferred_domain_frequency"
                    ][p],
                    "stability_scope": "conditional on one observed section",
                }
            )
    return pd.DataFrame(records)


def _weighted_correlation(left: np.ndarray, right: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(left) & np.isfinite(right) & (weights > 0)
    if keep.sum() < 2:
        return np.nan
    w = weights[keep] / weights[keep].sum()
    x, y = left[keep], right[keep]
    xc = x - np.sum(w * x)
    yc = y - np.sum(w * y)
    denominator = np.sqrt(np.sum(w * xc * xc) * np.sum(w * yc * yc))
    return float(np.sum(w * xc * yc) / denominator) if denominator > EPS else np.nan


def _profile_specificity(values: np.ndarray, support: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)[support]
    if values.size < 2:
        return np.nan
    shifted = values - values.max()
    probability = np.exp(np.clip(shifted, -50.0, 0.0))
    probability /= probability.sum()
    entropy = -np.sum(probability * np.log(np.maximum(probability, EPS)))
    return float(1.0 - entropy / np.log(values.size))


def _feature_profile_metrics(
    enrichment: np.ndarray,
    delta: np.ndarray,
    support: np.ndarray,
    omega: np.ndarray,
    bootstrap: dict,
    settings: DifferentialSettings,
) -> pd.DataFrame:
    p_count = delta.shape[1]
    records = []
    for p in range(p_count):
        common = support[0] & support[1]
        valid = np.flatnonzero(common)
        left = enrichment[0, :, p]
        right = enrichment[1, :, p]
        if valid.size:
            abs_delta = np.abs(delta[valid, p])
            max_local = int(np.argmax(abs_delta))
            max_domain = int(valid[max_local])
            max_delta = float(abs_delta[max_local])
            preferred_left = int(valid[np.argmax(left[valid])])
            preferred_right = int(valid[np.argmax(right[valid])])
            left_sorted = np.sort(left[valid])
            right_sorted = np.sort(right[valid])
            left_margin = float(left_sorted[-1] - left_sorted[-2]) if valid.size > 1 else np.nan
            right_margin = float(right_sorted[-1] - right_sorted[-2]) if valid.size > 1 else np.nan
            spear = stats.spearmanr(left[valid], right[valid]).statistic if valid.size > 1 else np.nan
        else:
            max_domain, max_delta = -1, np.nan
            preferred_left = preferred_right = -1
            left_margin = right_margin = spear = np.nan
        left_specificity = _profile_specificity(left, common)
        right_specificity = _profile_specificity(right, common)
        records.append(
            {
                "weighted_profile_correlation": _weighted_correlation(left, right, omega),
                "spearman_profile_correlation": float(spear) if np.isfinite(spear) else np.nan,
                "weighted_l2_deployment_shift": float(
                    np.sqrt(np.sum(omega * np.square(delta[:, p])))
                ),
                "maximum_absolute_domain_delta": max_delta,
                "domain_of_maximum_absolute_delta": max_domain,
                "preferred_domain_left": preferred_left,
                "preferred_domain_right": preferred_right,
                "preferred_domain_switch": preferred_left != preferred_right,
                "left_top_domain_margin": left_margin,
                "right_top_domain_margin": right_margin,
                "left_profile_specificity": left_specificity,
                "right_profile_specificity": right_specificity,
                "specificity_change": right_specificity - left_specificity,
                "number_of_domains_above_effect_threshold": int(
                    np.sum(common & (np.abs(delta[:, p]) >= settings.effect_size_threshold))
                ),
                "omnibus_deployment_pvalue": bootstrap["omnibus_pvalue"][p],
                "omnibus_deployment_qvalue": bootstrap["omnibus_qvalue"][p],
                "preferred_domain_switch_frequency": bootstrap["switch_frequency"][p],
                "n_shared_supported_domains": int(common.sum()),
            }
        )
    return pd.DataFrame(records)


def _block_bootstrap(
    streamed: _StreamedStatistics,
    arrays: dict,
    support: np.ndarray,
    omega: np.ndarray,
    settings: DifferentialSettings,
    spot_pitches: Sequence[float],
) -> dict:
    r_count, p_count = arrays["delta"].shape
    empty = {
        "ci_lower": np.full((r_count, p_count), np.nan),
        "ci_upper": np.full((r_count, p_count), np.nan),
        "median": np.full((r_count, p_count), np.nan),
        "sign_stability": np.full((r_count, p_count), np.nan),
        "conditional_pvalue": np.full((r_count, p_count), np.nan),
        "conditional_qvalue": np.full((r_count, p_count), np.nan),
        "omnibus_pvalue": np.full(p_count, np.nan),
        "omnibus_qvalue": np.full(p_count, np.nan),
        "switch_frequency": np.full(p_count, np.nan),
    }
    if settings.n_bootstrap <= 0:
        return empty

    rng = np.random.default_rng(settings.random_state + 104729)
    observed_draws = np.full((settings.n_bootstrap, r_count, p_count), np.nan)
    null_draws = np.full_like(observed_draws, np.nan)
    switch = np.zeros((settings.n_bootstrap, p_count), dtype=np.float64)
    pooled = 0.5 * (arrays["enrichment"][0] + arrays["enrichment"][1])
    null_theta = np.empty_like(arrays["mean"])
    for d in range(2):
        pi = arrays["domain_fraction"][d]
        pi = pi / max(pi.sum(), EPS)
        recenter = np.sum(pi[:, None] * pooled, axis=0)
        null_theta[d] = arrays["section_mean"][d][None, :] + pooled - recenter[None, :]

    observed_components = []
    null_components = []
    for d in range(2):
        sums = streamed.block_sum[d]
        mass = streamed.block_mass[d]
        area_per_spot = float(spot_pitches[d]) ** 2
        valid = mass >= settings.minimum_block_domain_spots * area_per_spot
        observed_components.append((sums, mass, valid))
        centered = sums - mass[:, :, None] * arrays["mean"][d][None, :, :]
        null_sums = centered + mass[:, :, None] * null_theta[d][None, :, :]
        null_components.append((null_sums, mass, valid))

    for b in range(settings.n_bootstrap):
        enrichment_observed = []
        enrichment_null = []
        for d in range(2):
            n_blocks = observed_components[d][0].shape[0]
            counts = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks))
            for components, target in (
                (observed_components, enrichment_observed),
                (null_components, enrichment_null),
            ):
                sums, mass, valid = components[d]
                selected_mass = np.tensordot(counts, mass * valid, axes=(0, 0))
                selected_sum = np.tensordot(
                    counts, sums * valid[:, :, None], axes=(0, 0)
                )
                theta = selected_sum / np.maximum(selected_mass[:, None], EPS)
                section_sum = np.tensordot(
                    counts, sums.sum(axis=1), axes=(0, 0)
                )
                section_mass = np.tensordot(counts, mass.sum(axis=1), axes=(0, 0))
                section_mean = section_sum / max(section_mass, EPS)
                target.append(theta - section_mean[None, :])
        observed_draws[b] = enrichment_observed[1] - enrichment_observed[0]
        null_draws[b] = enrichment_null[1] - enrichment_null[0]
        common = support[0] & support[1]
        for p in range(p_count):
            domains = np.flatnonzero(common)
            if domains.size:
                left_best = domains[np.argmax(enrichment_observed[0][domains, p])]
                right_best = domains[np.argmax(enrichment_observed[1][domains, p])]
                switch[b, p] = left_best != right_best

    ci_lower = np.nanquantile(observed_draws, 0.025, axis=0)
    ci_upper = np.nanquantile(observed_draws, 0.975, axis=0)
    median = np.nanmedian(observed_draws, axis=0)
    observed_sign = np.sign(arrays["delta"])[None, :, :]
    sign_stability = np.nanmean(np.sign(observed_draws) == observed_sign, axis=0)
    conditional_p = (
        1.0
        + np.sum(np.abs(null_draws) >= np.abs(arrays["delta"])[None, :, :], axis=0)
    ) / (settings.n_bootstrap + 1.0)
    supported_tests = support[0][:, None] & support[1][:, None]
    conditional_p[~np.broadcast_to(supported_tests, conditional_p.shape)] = np.nan
    conditional_q = _bh_fdr(conditional_p)
    observed_l2 = np.sqrt(np.sum(omega[:, None] * np.square(arrays["delta"]), axis=0))
    null_l2 = np.sqrt(np.sum(omega[None, :, None] * np.square(null_draws), axis=1))
    omnibus_p = (1.0 + np.sum(null_l2 >= observed_l2[None, :], axis=0)) / (
        settings.n_bootstrap + 1.0
    )
    omnibus_q = _bh_fdr(omnibus_p)
    return {
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median": median,
        "sign_stability": sign_stability,
        "conditional_pvalue": conditional_p,
        "conditional_qvalue": conditional_q,
        "omnibus_pvalue": omnibus_p,
        "omnibus_qvalue": omnibus_q,
        "switch_frequency": switch.mean(axis=0),
    }


def _bh_fdr(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    finite = np.flatnonzero(np.isfinite(values.ravel()))
    if finite.size == 0:
        return output
    p = values.ravel()[finite]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * ranked.size / np.arange(1, ranked.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    output.ravel()[finite] = restored
    return output


def _composition_decomposition(arrays: dict) -> dict:
    pi_left = arrays["domain_fraction"][0]
    pi_right = arrays["domain_fraction"][1]
    theta_left = arrays["mean"][0]
    theta_right = arrays["mean"][1]
    composition = np.sum(
        (pi_right - pi_left)[:, None] * 0.5 * (theta_left + theta_right), axis=0
    )
    within = np.sum(
        0.5 * (pi_left + pi_right)[:, None] * (theta_right - theta_left), axis=0
    )
    global_shift = arrays["section_mean"][1] - arrays["section_mean"][0]
    reconstruction = composition + within
    return {
        "global_shift": global_shift,
        "composition_effect": composition,
        "within_domain_effect": within,
        "reconstructed_global_shift": reconstruction,
        "reconstruction_error": global_shift - reconstruction,
    }


def _cluster_behavior_profiles(
    delta: np.ndarray,
    shared_support: np.ndarray,
    feature_metrics: pd.DataFrame,
    feature_ids: np.ndarray,
    settings: DifferentialSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    domains = np.flatnonzero(shared_support)
    p_count = delta.shape[1]
    if domains.size < 2:
        assignments = np.full(p_count, -1, dtype=np.int64)
        return _behavior_flags(assignments, delta, shared_support, feature_metrics, feature_ids, settings), pd.DataFrame(), {"selected_k": 0, "reason": "fewer than two shared supported domains"}
    profiles = delta[domains].T
    amplitude = np.linalg.norm(profiles, axis=1)
    nonzero = amplitude > 1.0e-10
    normalized = np.zeros_like(profiles)
    normalized[nonzero] = profiles[nonzero] / amplitude[nonzero, None]
    assignments = np.full(p_count, -1, dtype=np.int64)
    candidate_indices = np.flatnonzero(nonzero)
    scores: dict[int, float] = {}
    if candidate_indices.size >= 3:
        if settings.n_behavior_modules is not None:
            selected_k = min(settings.n_behavior_modules, candidate_indices.size)
        else:
            maximum = min(12, candidate_indices.size - 1)
            candidates = range(2, maximum + 1)
            for k in candidates:
                labels = AgglomerativeClustering(
                    n_clusters=k, metric="cosine", linkage="average"
                ).fit_predict(normalized[candidate_indices])
                scores[k] = float(
                    silhouette_score(normalized[candidate_indices], labels, metric="cosine")
                )
            selected_k = max(scores, key=scores.get) if scores else 1
        if selected_k >= 2:
            assignments[candidate_indices] = AgglomerativeClustering(
                n_clusters=selected_k, metric="cosine", linkage="average"
            ).fit_predict(normalized[candidate_indices])
        else:
            assignments[candidate_indices] = 0
    elif candidate_indices.size:
        selected_k = 1
        assignments[candidate_indices] = 0
    else:
        selected_k = 0
    behavior = _behavior_flags(
        assignments, delta, shared_support, feature_metrics, feature_ids, settings
    )
    centroid_records = []
    for module in sorted(set(assignments) - {-1}):
        members = assignments == module
        centroid = delta[:, members].mean(axis=1)
        for domain, value in enumerate(centroid):
            centroid_records.append(
                {
                    "behavior_module": int(module),
                    "domain": int(domain),
                    "centroid_delta_enrichment": float(value),
                    "domain_used_for_clustering": bool(shared_support[domain]),
                    "n_peaks": int(members.sum()),
                }
            )
    selection = {
        "selected_k": int(selected_k),
        "candidate_silhouette_scores": {str(k): value for k, value in scores.items()},
        "representation": "L2-normalized signed domain-delta profiles",
        "distance": "cosine",
        "linkage": "average",
    }
    return behavior, pd.DataFrame(centroid_records), selection


def _behavior_flags(
    assignments: np.ndarray,
    delta: np.ndarray,
    shared_support: np.ndarray,
    metrics: pd.DataFrame,
    feature_ids: np.ndarray,
    settings: DifferentialSettings,
) -> pd.DataFrame:
    records = []
    for p, feature in enumerate(feature_ids):
        values = delta[shared_support, p]
        absolute = np.abs(values)
        total = absolute.sum()
        dominant_share = float(absolute.max() / total) if total > EPS else 0.0
        n_effect = int(np.sum(absolute >= settings.effect_size_threshold))
        qvalue = metrics.iloc[p]["omnibus_deployment_qvalue"]
        significant = bool(np.isfinite(qvalue) and qvalue <= settings.fdr_threshold)
        max_effect = float(absolute.max()) if absolute.size else np.nan
        specificity_change = metrics.iloc[p]["specificity_change"]
        records.append(
            {
                "canonical_peak_id": str(feature),
                "behavior_module": int(assignments[p]),
                "differential_profile_l2_amplitude": float(np.linalg.norm(values)),
                "conserved_profile": bool(
                    np.isfinite(max_effect)
                    and max_effect < settings.effect_size_threshold
                    and not significant
                ),
                "dominant_domain_gain_or_loss": bool(
                    n_effect >= 1 and dominant_share >= 0.6
                ),
                "preferred_domain_switch": bool(
                    metrics.iloc[p]["preferred_domain_switch"]
                ),
                "broadened_profile": bool(
                    np.isfinite(specificity_change)
                    and specificity_change <= -0.15
                    and n_effect >= 1
                ),
                "narrowed_profile": bool(
                    np.isfinite(specificity_change)
                    and specificity_change >= 0.15
                    and n_effect >= 1
                ),
                "complex_redistribution": bool(n_effect >= 2 and dominant_share < 0.6),
                "low_support": bool(shared_support.sum() < 2),
                "dominant_absolute_effect_share": dominant_share,
            }
        )
    return pd.DataFrame(records)


def _metadata_records(crosswalk: pd.DataFrame, p: int) -> dict:
    return {
        key: crosswalk.iloc[p][key]
        for key in crosswalk.columns
        if key != "canonical_peak_id"
    }


def _distribution_summary(values: np.ndarray, threshold: float) -> dict:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: np.nan for key in ("weighted_median", "q10", "q25", "q75", "q90", "fraction_above_z1", "fraction_above_threshold")}
    q10, q25, median, q75, q90 = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "weighted_median": median,
        "q10": q10,
        "q25": q25,
        "q75": q75,
        "q90": q90,
        "fraction_above_z1": float(np.mean(values > 1.0)),
        "fraction_above_threshold": float(np.mean(values > threshold)),
    }


def _section_domain_peak_table(
    streamed, arrays, support, crosswalk, dataset_names, feature_ids, settings
) -> pd.DataFrame:
    records = []
    for d, section in enumerate(dataset_names):
        for r in arrays["domain_ids"]:
            sample = streamed.distribution.get((d, int(r)))
            for p, feature in enumerate(feature_ids):
                record = {
                    "section": section,
                    "domain": int(r),
                    "canonical_peak_id": feature,
                    **_metadata_records(crosswalk, p),
                    "soft_domain_mass": streamed.soft_mass[d, r],
                    "analysis_domain_mass": streamed.sum_weight[d, r],
                    "domain_area": arrays["domain_area"][d, r],
                    "domain_fraction": arrays["domain_fraction"][d, r],
                    "weighted_mean": arrays["mean"][d, r, p],
                    "weighted_variance": arrays["variance"][d, r, p],
                    "weighted_sd": np.sqrt(arrays["variance"][d, r, p]),
                    "mean_posterior_variance": arrays["posterior_variance"][d, r, p],
                    "posterior_variance_of_weighted_mean": arrays["posterior_mean_variance"][d, r, p],
                    "effective_weighted_support": arrays["effective_support"][d, r],
                    "hard_spot_count": streamed.spot_count[d, r],
                    "distribution_sample_size": streamed.distribution_sample_size[d, r],
                    "support_flag": bool(support[d, r]),
                }
                record.update(
                    _distribution_summary(
                        np.empty(0) if sample is None else sample[:, p],
                        settings.hotspot_z_threshold,
                    )
                )
                records.append(record)
    return pd.DataFrame(records)


def _peak_domain_contrast_table(arrays, support, bootstrap, crosswalk, feature_ids):
    records = []
    for r in arrays["domain_ids"]:
        for p, feature in enumerate(feature_ids):
            records.append(
                {
                    "canonical_peak_id": feature,
                    **_metadata_records(crosswalk, p),
                    "domain": int(r),
                    "left_enrichment": arrays["enrichment"][0, r, p],
                    "right_enrichment": arrays["enrichment"][1, r, p],
                    "delta_enrichment": arrays["delta"][r, p],
                    "bootstrap_ci_lower": bootstrap["ci_lower"][r, p],
                    "bootstrap_ci_upper": bootstrap["ci_upper"][r, p],
                    "sign_stability": bootstrap["sign_stability"][r, p],
                    "conditional_pvalue": bootstrap["conditional_pvalue"][r, p],
                    "conditional_qvalue": bootstrap["conditional_qvalue"][r, p],
                    "left_support": bool(support[0, r]),
                    "right_support": bool(support[1, r]),
                    "support_flag": bool(support[0, r] and support[1, r]),
                }
            )
    return pd.DataFrame(records)


def _feature_summary_table(metrics, decomposition, behavior, crosswalk, feature_ids):
    output = pd.concat(
        [crosswalk.reset_index(drop=True), metrics.reset_index(drop=True)], axis=1
    )
    for key, values in decomposition.items():
        output[key] = values
    behavior_extra = behavior.drop(columns=["canonical_peak_id"])
    behavior_extra = behavior_extra.drop(
        columns=[column for column in behavior_extra if column in output],
        errors="ignore",
    )
    return pd.concat([output, behavior_extra], axis=1)


def _domain_support_table(streamed, arrays, support, dataset_names, retained, spots):
    records = []
    for d, section in enumerate(dataset_names):
        section_rows = spots["dataset"].astype(str).to_numpy() == section
        for r in arrays["domain_ids"]:
            records.append(
                {
                    "section": section,
                    "domain": int(r),
                    "soft_domain_mass": streamed.soft_mass[d, r],
                    "analysis_domain_mass": streamed.sum_weight[d, r],
                    "domain_area": arrays["domain_area"][d, r],
                    "domain_fraction": arrays["domain_fraction"][d, r],
                    "effective_weighted_support": arrays["effective_support"][d, r],
                    "hard_spot_count": streamed.spot_count[d, r],
                    "support_flag": bool(support[d, r]),
                    "section_spots": int(section_rows.sum()),
                    "retained_section_spots": int(np.sum(retained & section_rows)),
                    "section_area": streamed.section_area[d],
                    "analyzed_area": streamed.analyzed_area[d],
                }
            )
    return pd.DataFrame(records)


def _bootstrap_table(bootstrap, feature_ids, domain_ids):
    records = []
    for r in domain_ids:
        for p, feature in enumerate(feature_ids):
            records.append(
                {
                    "canonical_peak_id": feature,
                    "domain": int(r),
                    "bootstrap_median_delta": bootstrap["median"][r, p],
                    "bootstrap_ci_lower": bootstrap["ci_lower"][r, p],
                    "bootstrap_ci_upper": bootstrap["ci_upper"][r, p],
                    "sign_stability_probability": bootstrap["sign_stability"][r, p],
                    "domain_contrast_pvalue": bootstrap["conditional_pvalue"][r, p],
                    "domain_contrast_qvalue": bootstrap["conditional_qvalue"][r, p],
                    "stability_scope": "conditional on the two observed sections",
                }
            )
    return pd.DataFrame(records)


def _decomposition_table(decomposition, crosswalk, feature_ids):
    output = crosswalk.copy().reset_index(drop=True)
    for key, values in decomposition.items():
        output[key] = values
    return output


def wasserstein_by_domain(
    left_samples: np.ndarray, right_samples: np.ndarray
) -> np.ndarray:
    """Return deterministic sampled Wasserstein distances feature by feature."""

    if left_samples.shape[1] != right_samples.shape[1]:
        raise ValueError("Sampled distributions have different feature counts")
    return np.asarray(
        [
            stats.wasserstein_distance(left_samples[:, p], right_samples[:, p])
            for p in range(left_samples.shape[1])
        ],
        dtype=np.float64,
    )


def dense_reference_statistics(
    values: np.ndarray,
    posterior_variance: np.ndarray,
    probabilities: np.ndarray,
    areas: np.ndarray,
) -> dict[str, np.ndarray]:
    """Small dense reference used by deterministic unit tests."""

    values = np.asarray(values, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    weights = probabilities * areas[:, None]
    denominator = weights.sum(axis=0)
    mean = weights.T @ values / np.maximum(denominator[:, None], EPS)
    second = weights.T @ np.square(values) / np.maximum(denominator[:, None], EPS)
    mean_u = weights.T @ posterior_variance / np.maximum(denominator[:, None], EPS)
    return {
        "mean": mean,
        "variance": np.maximum(second - np.square(mean), 0.0),
        "mean_posterior_variance": mean_u,
        "effective_support": np.square(denominator)
        / np.maximum(np.square(weights).sum(axis=0), EPS),
    }
