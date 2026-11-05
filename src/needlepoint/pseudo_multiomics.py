"""Registration-free prototype coupling utilities for pseudo-multi-omics benchmarks.

The functions in this module deliberately operate on spatial summaries and
prototype-to-domain relations. Feature identifiers, m/z values, and pointwise
cross-section correspondences are neither accepted nor used by the coupling
code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class CouplingResult:
    """Outputs of alternating prototype and domain coupling."""

    prototype_similarity: np.ndarray
    domain_similarity: np.ndarray
    domain_pairs: np.ndarray
    history: pd.DataFrame


def stratified_disjoint_split(
    canonical_ids: np.ndarray,
    reference_families: np.ndarray,
    *,
    random_state: int = 0,
) -> pd.DataFrame:
    """Split features equally while retaining each reference family on both sides.

    Reference families are used only as an experimental-design stratum. They
    are not included in model inputs or in spatial-signature descriptors.
    """
    canonical_ids = np.asarray(canonical_ids, dtype=str)
    reference_families = np.asarray(reference_families)
    if canonical_ids.ndim != 1 or reference_families.shape != canonical_ids.shape:
        raise ValueError("canonical_ids and reference_families must be aligned vectors")
    if np.unique(canonical_ids).size != canonical_ids.size:
        raise ValueError("canonical_ids must be unique")
    if canonical_ids.size % 2:
        raise ValueError("An equal split requires an even number of features")

    rng = np.random.default_rng(random_state)
    assignments = np.empty(canonical_ids.size, dtype=object)
    counts = {"A": 0, "B": 0}
    for family in np.unique(reference_families):
        indices = np.flatnonzero(reference_families == family)
        indices = rng.permutation(indices)
        half = indices.size // 2
        assignments[indices[:half]] = "A"
        assignments[indices[half : 2 * half]] = "B"
        counts["A"] += half
        counts["B"] += half
        if indices.size % 2:
            side = "A" if counts["A"] <= counts["B"] else "B"
            assignments[indices[-1]] = side
            counts[side] += 1

    if counts["A"] != counts["B"]:
        raise RuntimeError(f"Stratified split was not balanced: {counts}")
    return pd.DataFrame(
        {
            "canonical_peak_id": canonical_ids,
            "reference_peak_family": reference_families,
            "pseudo_modality_panel": assignments,
        }
    )


def invariant_prototype_descriptors(
    prototype_maps: np.ndarray,
    coordinates: np.ndarray,
    responsibilities: np.ndarray,
    feature_metadata: pd.DataFrame,
    *,
    max_spots: int = 50_000,
    random_state: int = 0,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build translation/rotation/reflection/scale-tolerant prototype descriptors.

    Returns the descriptor table and the deterministic spot sample used for
    graph and morphology summaries. Coordinates are used only within a
    modality; no coordinates are compared across modalities.
    """
    maps = prototype_maps
    coords = np.asarray(coordinates, dtype=np.float64)
    gamma = np.asarray(responsibilities, dtype=np.float64)
    if maps.ndim != 2 or coords.shape != (maps.shape[0], 2):
        raise ValueError("prototype_maps and coordinates have incompatible shapes")
    if gamma.ndim != 2 or gamma.shape[1] != maps.shape[1]:
        raise ValueError("responsibilities must have one column per prototype")
    if gamma.shape[0] != feature_metadata.shape[0]:
        raise ValueError("feature metadata and responsibilities are not aligned")

    # Sampling by row identity, rather than coordinate bins, makes the selected
    # observations invariant to rigid transforms of the coordinate system.
    # At the intended 50k observations, uniform sampling is still dense enough
    # for the morphology and neighborhood summaries used below.
    sample_size = min(max_spots, maps.shape[0])
    if sample_size == maps.shape[0]:
        rows = np.arange(maps.shape[0], dtype=np.int64)
    else:
        rng = np.random.default_rng(random_state)
        rows = np.sort(rng.choice(maps.shape[0], size=sample_size, replace=False))
    values = np.asarray(maps[rows], dtype=np.float64)
    sampled_coords = coords[rows]
    coordinate_center = coords.mean(axis=0)
    physical_scale = float(
        np.sqrt(np.mean(np.square(coords - coordinate_center).sum(axis=1)))
    )
    physical_scale = max(physical_scale, 1.0e-12)
    normalized_coords = (sampled_coords - coordinate_center) / physical_scale
    tissue_center = normalized_coords.mean(axis=0)

    neighbor_ranks = (1, 4, 16, 64)
    query_k = min(max(neighbor_ranks) + 1, rows.size)
    neighbor_distances, neighbors = cKDTree(normalized_coords).query(
        normalized_coords, k=query_k, workers=-1
    )
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
        neighbor_distances = neighbor_distances[:, None]
    # KD-tree ordering is unspecified when several neighbors are equidistant.
    # Resolve those ties by row identity so rotations/reflections of regular
    # lattices produce identical descriptors.
    index_order = np.argsort(neighbors, axis=1, kind="mergesort")
    neighbors = np.take_along_axis(neighbors, index_order, axis=1)
    neighbor_distances = np.take_along_axis(neighbor_distances, index_order, axis=1)
    distance_order = np.argsort(
        np.round(neighbor_distances, decimals=12), axis=1, kind="mergesort"
    )
    neighbors = np.take_along_axis(neighbors, distance_order, axis=1)
    graph_k = min(5, neighbors.shape[1])
    source = np.repeat(np.arange(rows.size), max(graph_k - 1, 0))
    target = neighbors[:, 1:graph_k].reshape(-1)
    graph = sparse.csr_matrix(
        (np.ones(source.size, dtype=np.uint8), (source, target)),
        shape=(rows.size, rows.size),
    )
    graph = graph.maximum(graph.T)

    global_scale = _metadata_vector(
        feature_metadata, ("joint_global_length_scale", "learned_length_scale")
    ) / physical_scale
    local_scale = _metadata_vector(
        feature_metadata, ("joint_local_length_scale", "local_pretrain_length_scale")
    ) / physical_scale

    records: list[dict[str, float | int]] = []
    quantile_levels = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    for prototype in range(values.shape[1]):
        field = values[:, prototype]
        center = float(np.median(field))
        scale = float(np.subtract(*np.quantile(field, [0.75, 0.25])))
        if scale <= 1.0e-12:
            scale = float(np.std(field))
        scale = max(scale, 1.0e-12)
        z = (field - center) / scale
        record: dict[str, float | int] = {"prototype": prototype}
        for level, value in zip(quantile_levels, np.quantile(z, quantile_levels)):
            record[f"field_q{int(level * 100):02d}"] = float(value)

        positive = np.maximum(z, 0.0) + 1.0e-8
        positive /= positive.sum()
        spatial_center = np.sum(normalized_coords * positive[:, None], axis=0)
        displacement = spatial_center - tissue_center
        centered = normalized_coords - spatial_center
        covariance = (centered * positive[:, None]).T @ centered
        eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
        record["centroid_displacement"] = float(np.linalg.norm(displacement))
        record["spatial_inertia"] = float(eigenvalues.sum())
        record["spatial_anisotropy"] = float(
            (eigenvalues[0] - eigenvalues[-1]) / max(eigenvalues.sum(), 1.0e-12)
        )
        radius = np.linalg.norm(centered, axis=1)
        record["radial_mean"] = float(np.sum(positive * radius))
        record["radial_sd"] = float(
            np.sqrt(np.sum(positive * np.square(radius - record["radial_mean"])))
        )

        for rank in neighbor_ranks:
            effective_rank = min(rank, neighbors.shape[1] - 1)
            neighbor_field = z[neighbors[:, effective_rank]]
            record[f"knn_correlation_k{rank}"] = _safe_correlation(z, neighbor_field)
            record[f"knn_semivariance_k{rank}"] = float(
                0.5 * np.mean(np.square(z - neighbor_field))
                / max(np.var(z), 1.0e-12)
            )

        for threshold in (0.5, 1.5):
            active = z > threshold
            prefix = f"enriched_z{str(threshold).replace('.', 'p')}"
            record[f"{prefix}_fraction"] = float(active.mean())
            if active.any():
                n_components, labels = connected_components(
                    graph[active][:, active], directed=False
                )
                component_sizes = np.bincount(labels)
                record[f"{prefix}_log_components"] = float(np.log1p(n_components))
                record[f"{prefix}_largest_component"] = float(
                    component_sizes.max() / active.sum()
                )
            else:
                record[f"{prefix}_log_components"] = 0.0
                record[f"{prefix}_largest_component"] = 0.0

        feature_weights = np.maximum(gamma[:, prototype], 0.0)
        record["prototype_feature_mass"] = float(feature_weights.sum() / gamma.shape[0])
        for name, scale_values in (
            ("global_scale", global_scale),
            ("local_scale", local_scale),
        ):
            for level in (0.25, 0.50, 0.75):
                record[f"{name}_q{int(level * 100):02d}"] = _weighted_quantile(
                    scale_values, feature_weights, level
                )
        records.append(record)
    return pd.DataFrame.from_records(records), rows


def prototype_domain_relations(
    prototype_maps: np.ndarray,
    domain_probabilities: np.ndarray,
    sample_rows: np.ndarray,
) -> np.ndarray:
    """Compute prototype enrichment across soft domains on one modality."""
    fields = np.asarray(prototype_maps[sample_rows], dtype=np.float64)
    domains = np.asarray(domain_probabilities[sample_rows], dtype=np.float64)
    if fields.shape[0] != domains.shape[0]:
        raise ValueError("Prototype fields and domain probabilities are not aligned")
    relation = fields.T @ domains
    relation /= np.maximum(domains.sum(axis=0)[None, :], 1.0e-12)
    relation -= relation.mean(axis=1, keepdims=True)
    relation /= np.maximum(relation.std(axis=1, keepdims=True), 1.0e-12)
    return relation


def robust_cross_similarity(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Compute an RBF similarity after joint robust descriptor scaling."""
    columns = [column for column in left.columns if column != "prototype"]
    if set(columns) != {column for column in right.columns if column != "prototype"}:
        raise ValueError("Descriptor columns differ between modalities")
    left_values = left[columns].to_numpy(dtype=np.float64)
    right_values = right[columns].to_numpy(dtype=np.float64)
    combined = np.vstack([left_values, right_values])
    median = np.nanmedian(combined, axis=0)
    q25, q75 = np.nanquantile(combined, [0.25, 0.75], axis=0)
    scale = q75 - q25
    fallback = np.nanstd(combined, axis=0)
    scale = np.where(scale > 1.0e-10, scale, fallback)
    keep = np.isfinite(scale) & (scale > 1.0e-10)
    if not keep.any():
        raise ValueError("All descriptor columns are constant")
    columns = [column for column, retained in zip(columns, keep) if retained]
    left_scaled = np.clip((left_values[:, keep] - median[keep]) / scale[keep], -8.0, 8.0)
    right_scaled = np.clip((right_values[:, keep] - median[keep]) / scale[keep], -8.0, 8.0)
    squared_distance = np.mean(
        np.square(left_scaled[:, None, :] - right_scaled[None, :, :]), axis=2
    )
    bandwidth = max(float(np.median(squared_distance)), 1.0e-8)
    similarity = np.exp(-0.5 * squared_distance / bandwidth)
    left_table = pd.DataFrame(left_scaled, columns=columns)
    right_table = pd.DataFrame(right_scaled, columns=columns)
    left_table.insert(0, "prototype", left["prototype"].to_numpy())
    right_table.insert(0, "prototype", right["prototype"].to_numpy())
    return similarity, left_table, right_table


def alternating_relational_coupling(
    geometry_similarity: np.ndarray,
    left_relations: np.ndarray,
    right_relations: np.ndarray,
    *,
    shape_weight: float = 0.5,
    temperature: float = 0.15,
    max_iter: int = 10,
    tolerance: float = 1.0e-6,
) -> CouplingResult:
    """Alternate prototype matching and shared-domain archetype matching."""
    similarity = np.asarray(geometry_similarity, dtype=np.float64).copy()
    left_relations = np.asarray(left_relations, dtype=np.float64)
    right_relations = np.asarray(right_relations, dtype=np.float64)
    if similarity.shape != (left_relations.shape[0], right_relations.shape[0]):
        raise ValueError("Prototype similarity and relation matrices are incompatible")
    if not 0.0 <= shape_weight <= 1.0:
        raise ValueError("shape_weight must lie in [0, 1]")

    history: list[dict[str, float | int]] = []
    domain_similarity = np.empty((left_relations.shape[1], right_relations.shape[1]))
    domain_pairs = np.empty((0, 2), dtype=np.int64)
    for iteration in range(max_iter):
        transport = _mutual_soft_transport(similarity, temperature)
        right_in_left_basis = transport @ right_relations
        domain_similarity = correlation_similarity(
            left_relations.T, right_in_left_basis.T
        )
        domain_pairs = hungarian_pairs(domain_similarity)
        right_aligned = np.zeros(
            (right_relations.shape[0], left_relations.shape[1]), dtype=np.float64
        )
        for left_domain, right_domain in domain_pairs:
            right_aligned[:, left_domain] = right_relations[:, right_domain]
        relational_similarity = correlation_similarity(left_relations, right_aligned)
        relational_similarity = np.clip((relational_similarity + 1.0) / 2.0, 0.0, 1.0)
        updated = shape_weight * geometry_similarity + (1.0 - shape_weight) * relational_similarity
        change = float(np.max(np.abs(updated - similarity)))
        history.append(
            {
                "iteration": iteration + 1,
                "max_similarity_change": change,
                "mean_geometry_similarity": float(np.mean(geometry_similarity)),
                "mean_relational_similarity": float(np.mean(relational_similarity)),
                "mean_coupled_similarity": float(np.mean(updated)),
                "mean_matched_domain_similarity": float(
                    np.mean([domain_similarity[i, j] for i, j in domain_pairs])
                ),
            }
        )
        similarity = updated
        if change <= tolerance:
            break
    return CouplingResult(
        prototype_similarity=similarity,
        domain_similarity=domain_similarity,
        domain_pairs=domain_pairs,
        history=pd.DataFrame.from_records(history),
    )


def soft_assignments_with_private(
    similarity: np.ndarray,
    *,
    private_quantile: float = 0.75,
    temperature: float = 0.10,
) -> tuple[np.ndarray, float]:
    """Return row-wise soft assignments with an explicit private component."""
    similarity = np.asarray(similarity, dtype=np.float64)
    if not 0.0 <= private_quantile <= 1.0 or temperature <= 0.0:
        raise ValueError("Invalid private assignment settings")
    private_score = float(np.quantile(similarity, private_quantile))
    logits = np.column_stack(
        [similarity, np.full(similarity.shape[0], private_score)]
    ) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities, private_score


def hungarian_pairs(similarity: np.ndarray) -> np.ndarray:
    """Maximum-weight one-to-one pairs for a rectangular similarity matrix."""
    row, column = linear_sum_assignment(-np.asarray(similarity, dtype=np.float64))
    return np.column_stack([row, column]).astype(np.int64)


def correlation_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cross-correlation of row vectors with stable constant-row handling."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left = left / np.maximum(left_norm, 1.0e-12)
    right = right / np.maximum(right_norm, 1.0e-12)
    return np.clip(left @ right.T, -1.0, 1.0)


def _mutual_soft_transport(similarity: np.ndarray, temperature: float) -> np.ndarray:
    logits = similarity / max(temperature, 1.0e-8)
    row_logits = logits - logits.max(axis=1, keepdims=True)
    row_probability = np.exp(row_logits)
    row_probability /= row_probability.sum(axis=1, keepdims=True)
    column_logits = logits - logits.max(axis=0, keepdims=True)
    column_probability = np.exp(column_logits)
    column_probability /= column_probability.sum(axis=0, keepdims=True)
    transport = np.sqrt(row_probability * column_probability)
    transport /= np.maximum(transport.sum(axis=1, keepdims=True), 1.0e-12)
    return transport


def _metadata_vector(metadata: pd.DataFrame, candidates: tuple[str, ...]) -> np.ndarray:
    for column in candidates:
        if column in metadata:
            values = pd.to_numeric(metadata[column], errors="coerce").to_numpy(dtype=np.float64)
            if np.isfinite(values).all():
                return values
    raise ValueError(f"Metadata lacks a finite spatial-scale column among {candidates}")


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not keep.any():
        return float("nan")
    values = values[keep]
    weights = weights[keep]
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    target = quantile * cumulative[-1]
    return float(values[order[min(np.searchsorted(cumulative, target), order.size - 1)]])


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
