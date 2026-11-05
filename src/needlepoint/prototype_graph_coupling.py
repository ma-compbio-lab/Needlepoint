"""Prototype graph construction and registration-free cross-section coupling.

This module complements the descriptor/domain coupling baseline in
``pseudo_multiomics.py``. It treats each prototype as a multiscale spatial
measure and matches complete prototype systems through both node-level support
geometry and prototype-to-prototype relations. It never consumes feature
identifiers, m/z values, cross-section coordinates, or pointwise correspondences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class PrototypeGraph:
    """A modality-internal graph of spatial prototypes.

    Attributes
    ----------
    node_descriptors
        One row per prototype. Columns summarize multiscale cumulative-mass
        supports and optional GP scale information.
    relations
        Array with shape ``(K, K, C)``. Each channel describes a directed or
        symmetric relation between two prototypes.
    relation_names
        Names of the relation channels in ``relations``.
    sample_rows
        Rows from the original spatial map used to build the graph.
    mass_levels
        Cumulative positive-mass support levels used in the descriptors.
    """

    node_descriptors: pd.DataFrame
    relations: np.ndarray
    relation_names: tuple[str, ...]
    sample_rows: np.ndarray
    mass_levels: tuple[float, ...]


@dataclass(frozen=True)
class PrototypeGraphCouplingResult:
    """Outputs of fused node/relational prototype coupling."""

    prototype_similarity: np.ndarray
    transport: np.ndarray
    reverse_transport: np.ndarray
    left_private_probability: np.ndarray
    right_private_probability: np.ndarray
    relational_similarity: np.ndarray
    history: pd.DataFrame


def multiscale_prototype_graph(
    prototype_maps: np.ndarray,
    coordinates: np.ndarray,
    responsibilities: np.ndarray | None = None,
    feature_metadata: pd.DataFrame | None = None,
    *,
    mass_levels: Iterable[float] = (0.25, 0.50, 0.75, 0.90),
    max_spots: int = 50_000,
    graph_k: int = 8,
    random_state: int = 0,
) -> PrototypeGraph:
    """Build multiscale support descriptors and a prototype relation graph.

    Prototype fields are converted to normalized positive spatial measures.
    For each cumulative-mass level, the smallest high-valued support containing
    that amount of mass is extracted. Descriptors summarize support extent,
    connectivity, elongation, boundary complexity, and a thickness proxy.

    Relations are computed only within one section and include field
    correlation, positive-mass overlap, multiscale support overlap,
    cross-support adjacency, centroid proximity, and directed containment.
    """

    maps = np.asarray(prototype_maps)
    coords = np.asarray(coordinates, dtype=np.float64)
    if maps.ndim != 2:
        raise ValueError("prototype_maps must have shape (spots, prototypes)")
    if coords.shape != (maps.shape[0], 2):
        raise ValueError("coordinates must have shape (spots, 2)")
    if maps.shape[0] < 2 or maps.shape[1] < 1:
        raise ValueError("At least two spots and one prototype are required")
    if max_spots < 2 or graph_k < 1:
        raise ValueError("max_spots must be >=2 and graph_k must be positive")

    levels = tuple(float(level) for level in mass_levels)
    if not levels or any(not 0.0 < level < 1.0 for level in levels):
        raise ValueError("mass_levels must be nonempty and lie strictly in (0, 1)")
    if tuple(sorted(set(levels))) != levels:
        raise ValueError("mass_levels must be unique and strictly increasing")

    rows = _sample_rows(maps.shape[0], max_spots=max_spots, random_state=random_state)
    values = np.asarray(maps[rows], dtype=np.float64)
    sampled_coords = coords[rows]
    normalized_coords, physical_scale = _normalize_coordinates(coords, sampled_coords)
    graph, edge_source, edge_target = _knn_graph(normalized_coords, graph_k=graph_k)

    z = np.empty_like(values, dtype=np.float64)
    measures = np.empty_like(values, dtype=np.float64)
    supports = np.empty((len(levels), values.shape[0], values.shape[1]), dtype=bool)
    for prototype in range(values.shape[1]):
        z[:, prototype] = _robust_standardize(values[:, prototype])
        measures[:, prototype] = _positive_measure(z[:, prototype])
        for level_index, level in enumerate(levels):
            supports[level_index, :, prototype] = cumulative_mass_support(
                measures[:, prototype], level
            )

    descriptors = _node_descriptors(
        z,
        measures,
        supports,
        levels,
        normalized_coords,
        graph,
        edge_source,
        edge_target,
        responsibilities,
        feature_metadata,
        physical_scale,
    )
    relations, relation_names = _prototype_relations(
        z,
        measures,
        supports,
        levels,
        normalized_coords,
        edge_source,
        edge_target,
    )
    return PrototypeGraph(
        node_descriptors=descriptors,
        relations=relations,
        relation_names=relation_names,
        sample_rows=rows,
        mass_levels=levels,
    )


def cumulative_mass_support(mass: np.ndarray, level: float) -> np.ndarray:
    """Return the smallest deterministic high-mass support reaching ``level``.

    Ties are resolved by row index, making the result deterministic. The input
    is normalized internally and may contain zeros, but not negative or
    non-finite values.
    """

    values = np.asarray(mass, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("mass must be a nonempty vector")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("mass must contain finite nonnegative values")
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly in (0, 1)")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("mass must have positive total")
    order = np.lexsort((np.arange(values.size), -values))
    cumulative = np.cumsum(values[order]) / total
    count = int(np.searchsorted(cumulative, level, side="left") + 1)
    support = np.zeros(values.size, dtype=bool)
    support[order[:count]] = True
    return support


def robust_descriptor_similarity(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Compute RBF node similarity after pooled robust standardization."""

    left_columns = [column for column in left if column != "prototype"]
    right_columns = [column for column in right if column != "prototype"]
    if set(left_columns) != set(right_columns):
        raise ValueError("Descriptor columns differ between prototype graphs")
    columns = left_columns
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

    retained = [column for column, flag in zip(columns, keep) if flag]
    left_scaled = np.clip((left_values[:, keep] - median[keep]) / scale[keep], -8.0, 8.0)
    right_scaled = np.clip((right_values[:, keep] - median[keep]) / scale[keep], -8.0, 8.0)
    distance = np.mean(
        np.square(left_scaled[:, None, :] - right_scaled[None, :, :]), axis=2
    )
    positive_distance = distance[distance > 1.0e-12]
    bandwidth = float(np.median(positive_distance)) if positive_distance.size else 1.0
    bandwidth = max(bandwidth, 1.0e-8)
    similarity = np.exp(-0.5 * distance / bandwidth)

    left_table = pd.DataFrame(left_scaled, columns=retained)
    right_table = pd.DataFrame(right_scaled, columns=retained)
    left_table.insert(0, "prototype", left["prototype"].to_numpy())
    right_table.insert(0, "prototype", right["prototype"].to_numpy())
    return similarity, left_table, right_table


def fused_prototype_graph_coupling(
    node_similarity: np.ndarray,
    left_relations: np.ndarray,
    right_relations: np.ndarray,
    *,
    node_weight: float = 0.35,
    relation_weights: np.ndarray | None = None,
    temperature: float = 0.10,
    marginal_relaxation: float = 0.50,
    left_marginal: np.ndarray | None = None,
    right_marginal: np.ndarray | None = None,
    max_iter: int = 30,
    sinkhorn_iter: int = 200,
    tolerance: float = 1.0e-6,
) -> PrototypeGraphCouplingResult:
    """Couple prototype graphs with node and relational evidence.

    This is a compact fused, unbalanced graph-matching routine rather than a
    full general-purpose FGW solver. Each iteration:

    1. obtains an unbalanced soft transport from the current similarity;
    2. maps every candidate prototype's relation vector through that transport;
    3. scores how well prototype-to-prototype relations are preserved; and
    4. mixes relational compatibility with the fixed node similarity.

    Unbalanced transport permits unmatched mass and many-to-many split/merge
    correspondences. Private probabilities are derived from marginal mass
    deficits rather than from an extra forced one-to-one assignment.
    """

    node = np.asarray(node_similarity, dtype=np.float64)
    left = np.asarray(left_relations, dtype=np.float64)
    right = np.asarray(right_relations, dtype=np.float64)
    if node.ndim != 2:
        raise ValueError("node_similarity must be a matrix")
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("relation arrays must have shape (K, K, channels)")
    if left.shape[:2] != (node.shape[0], node.shape[0]):
        raise ValueError("left relation graph does not match node_similarity")
    if right.shape[:2] != (node.shape[1], node.shape[1]):
        raise ValueError("right relation graph does not match node_similarity")
    if left.shape[2] != right.shape[2]:
        raise ValueError("Prototype graphs have different relation channels")
    if not np.isfinite(node).all() or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Coupling inputs must be finite")
    if not 0.0 <= node_weight <= 1.0:
        raise ValueError("node_weight must lie in [0, 1]")
    if temperature <= 0.0 or marginal_relaxation <= 0.0:
        raise ValueError("temperature and marginal_relaxation must be positive")

    channels = left.shape[2]
    if relation_weights is None:
        weights = np.ones(channels, dtype=np.float64) / channels
    else:
        weights = np.asarray(relation_weights, dtype=np.float64)
        if weights.shape != (channels,) or np.any(weights < 0.0) or weights.sum() <= 0.0:
            raise ValueError("relation_weights must be a nonnegative channel vector")
        weights = weights / weights.sum()

    similarity = np.clip(node, 0.0, 1.0).copy()
    history: list[dict[str, float | int]] = []
    relational = np.zeros_like(similarity)
    transport = np.zeros_like(similarity)
    reverse = np.zeros((node.shape[1], node.shape[0]), dtype=np.float64)
    left_private = np.zeros(node.shape[0], dtype=np.float64)
    right_private = np.zeros(node.shape[1], dtype=np.float64)

    for iteration in range(max_iter):
        transport, reverse, left_private, right_private = unbalanced_soft_transport(
            similarity,
            temperature=temperature,
            marginal_relaxation=marginal_relaxation,
            left_marginal=left_marginal,
            right_marginal=right_marginal,
            max_iter=sinkhorn_iter,
            tolerance=tolerance * 0.1,
        )
        relational = _relational_similarity(left, right, transport, reverse, weights)
        updated = node_weight * node + (1.0 - node_weight) * relational
        updated = np.clip(updated, 0.0, 1.0)
        change = float(np.max(np.abs(updated - similarity)))
        history.append(
            {
                "iteration": iteration + 1,
                "max_similarity_change": change,
                "mean_node_similarity": float(node.mean()),
                "mean_relational_similarity": float(relational.mean()),
                "mean_coupled_similarity": float(updated.mean()),
                "transport_mass": float(transport.sum()),
                "mean_left_private_probability": float(left_private.mean()),
                "mean_right_private_probability": float(right_private.mean()),
            }
        )
        similarity = updated
        if change <= tolerance:
            break

    transport, reverse, left_private, right_private = unbalanced_soft_transport(
        similarity,
        temperature=temperature,
        marginal_relaxation=marginal_relaxation,
        left_marginal=left_marginal,
        right_marginal=right_marginal,
        max_iter=sinkhorn_iter,
        tolerance=tolerance * 0.1,
    )
    return PrototypeGraphCouplingResult(
        prototype_similarity=similarity,
        transport=transport,
        reverse_transport=reverse,
        left_private_probability=left_private,
        right_private_probability=right_private,
        relational_similarity=relational,
        history=pd.DataFrame.from_records(history),
    )


def unbalanced_soft_transport(
    similarity: np.ndarray,
    *,
    temperature: float = 0.10,
    marginal_relaxation: float = 0.50,
    left_marginal: np.ndarray | None = None,
    right_marginal: np.ndarray | None = None,
    max_iter: int = 200,
    tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return unbalanced forward/reverse transports and private mass.

    The update is the KL-relaxed Sinkhorn scaling used in entropic unbalanced
    optimal transport. Marginal deficits are reported as prototype-private
    probabilities. Forward and reverse rows sum to the retained shared mass,
    so uncertain/private prototypes exert less force during relation projection.
    """

    score = np.asarray(similarity, dtype=np.float64)
    if score.ndim != 2 or score.size == 0 or not np.isfinite(score).all():
        raise ValueError("similarity must be a finite nonempty matrix")
    if temperature <= 0.0 or marginal_relaxation <= 0.0:
        raise ValueError("temperature and marginal_relaxation must be positive")

    n_left, n_right = score.shape
    a = _normalized_marginal(left_marginal, n_left, "left_marginal")
    b = _normalized_marginal(right_marginal, n_right, "right_marginal")
    cost = 1.0 - np.clip(score, 0.0, 1.0)
    kernel = np.exp(-cost / temperature)
    kernel = np.maximum(kernel, 1.0e-300)
    exponent = marginal_relaxation / (marginal_relaxation + temperature)
    u = np.ones(n_left, dtype=np.float64)
    v = np.ones(n_right, dtype=np.float64)
    for _ in range(max_iter):
        old_u = u.copy()
        old_v = v.copy()
        u = np.power(a / np.maximum(kernel @ v, 1.0e-300), exponent)
        v = np.power(b / np.maximum(kernel.T @ u, 1.0e-300), exponent)
        if max(np.max(np.abs(u - old_u)), np.max(np.abs(v - old_v))) <= tolerance:
            break

    raw = u[:, None] * kernel * v[None, :]
    row_mass = raw.sum(axis=1)
    column_mass = raw.sum(axis=0)
    left_private = np.clip(1.0 - row_mass / a, 0.0, 1.0)
    right_private = np.clip(1.0 - column_mass / b, 0.0, 1.0)
    forward = raw / np.maximum(row_mass[:, None], 1.0e-300)
    reverse = raw.T / np.maximum(column_mass[:, None], 1.0e-300)
    forward *= (1.0 - left_private)[:, None]
    reverse *= (1.0 - right_private)[:, None]
    return forward, reverse, left_private, right_private


def tempered_prototype_marginal(masses: np.ndarray, power: float) -> np.ndarray:
    """Normalize nonnegative prototype masses after power tempering.

    Power zero gives a uniform marginal, intermediate powers reduce domination
    by large components, and power one preserves the observed mass ratios.
    """

    values = np.asarray(masses, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("masses must be a finite nonempty vector")
    if np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValueError("masses must be nonnegative with positive total mass")
    if not np.isfinite(power) or power < 0.0:
        raise ValueError("power must be finite and nonnegative")
    if power == 0.0:
        return np.full(values.size, 1.0 / values.size, dtype=np.float64)
    tempered = np.power(values, power)
    return tempered / tempered.sum()


def _normalized_marginal(
    marginal: np.ndarray | None,
    expected_size: int,
    name: str,
) -> np.ndarray:
    if marginal is None:
        return np.full(expected_size, 1.0 / expected_size, dtype=np.float64)
    values = np.asarray(marginal, dtype=np.float64)
    if values.shape != (expected_size,) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a finite vector of length {expected_size}")
    if np.any(values < 0.0) or values.sum() <= 0.0:
        raise ValueError(f"{name} must be nonnegative with positive total mass")
    return values / values.sum()


def _sample_rows(n_spots: int, *, max_spots: int, random_state: int) -> np.ndarray:
    sample_size = min(n_spots, max_spots)
    if sample_size == n_spots:
        return np.arange(n_spots, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_spots, size=sample_size, replace=False))


def _normalize_coordinates(
    all_coordinates: np.ndarray,
    sampled_coordinates: np.ndarray,
) -> tuple[np.ndarray, float]:
    center = np.mean(all_coordinates, axis=0)
    physical_scale = float(
        np.sqrt(np.mean(np.square(all_coordinates - center).sum(axis=1)))
    )
    physical_scale = max(physical_scale, 1.0e-12)
    return (sampled_coordinates - center) / physical_scale, physical_scale


def _knn_graph(
    coordinates: np.ndarray,
    *,
    graph_k: int,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    # Query extra candidates so equal-distance ties at the requested k do not
    # depend on the KD-tree's internal ordering. We then select neighbors by
    # rounded distance and row identity.
    candidate_k = min(max(graph_k * 2 + 1, graph_k + 9), coordinates.shape[0])
    distances, neighbors = cKDTree(coordinates).query(
        coordinates, k=candidate_k, workers=-1
    )
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
        distances = distances[:, None]
    index_order = np.argsort(neighbors, axis=1, kind="mergesort")
    neighbors = np.take_along_axis(neighbors, index_order, axis=1)
    distances = np.take_along_axis(distances, index_order, axis=1)
    distance_order = np.argsort(np.round(distances, 12), axis=1, kind="mergesort")
    neighbors = np.take_along_axis(neighbors, distance_order, axis=1)[:, : graph_k + 1]
    source = np.repeat(np.arange(coordinates.shape[0]), max(neighbors.shape[1] - 1, 0))
    target = neighbors[:, 1:].reshape(-1)
    keep = source != target
    source = source[keep]
    target = target[keep]
    graph = sparse.csr_matrix(
        (np.ones(source.size, dtype=np.uint8), (source, target)),
        shape=(coordinates.shape[0], coordinates.shape[0]),
    ).maximum(
        sparse.csr_matrix(
            (np.ones(source.size, dtype=np.uint8), (target, source)),
            shape=(coordinates.shape[0], coordinates.shape[0]),
        )
    )
    upper_source, upper_target = sparse.triu(graph, k=1).nonzero()
    return graph, upper_source.astype(np.int64), upper_target.astype(np.int64)


def _robust_standardize(field: np.ndarray) -> np.ndarray:
    center = float(np.median(field))
    q25, q75 = np.quantile(field, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 1.0e-12:
        scale = float(np.std(field))
    scale = max(scale, 1.0e-12)
    return (field - center) / scale


def _positive_measure(z: np.ndarray) -> np.ndarray:
    positive = np.maximum(z, 0.0)
    if positive.sum() <= 1.0e-12:
        positive = z - np.min(z) + 1.0e-8
    positive = np.maximum(positive, 1.0e-12)
    return positive / positive.sum()


def _node_descriptors(
    z: np.ndarray,
    measures: np.ndarray,
    supports: np.ndarray,
    levels: tuple[float, ...],
    coordinates: np.ndarray,
    graph: sparse.csr_matrix,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    responsibilities: np.ndarray | None,
    feature_metadata: pd.DataFrame | None,
    physical_scale: float,
) -> pd.DataFrame:
    records: list[dict[str, float | int]] = []
    global_scale = local_scale = None
    gamma = None
    if responsibilities is not None and feature_metadata is not None:
        gamma = np.asarray(responsibilities, dtype=np.float64)
        if gamma.ndim != 2 or gamma.shape[1] != z.shape[1]:
            raise ValueError("responsibilities must have one column per prototype")
        if feature_metadata.shape[0] != gamma.shape[0]:
            raise ValueError("feature_metadata and responsibilities are not aligned")
        global_scale = _metadata_vector(
            feature_metadata, ("joint_global_length_scale", "learned_length_scale")
        ) / physical_scale
        local_scale = _metadata_vector(
            feature_metadata, ("joint_local_length_scale", "local_pretrain_length_scale")
        ) / physical_scale

    for prototype in range(z.shape[1]):
        record: dict[str, float | int] = {"prototype": prototype}
        measure = measures[:, prototype]
        for level_index, level in enumerate(levels):
            active = supports[level_index, :, prototype]
            prefix = f"mass{int(round(level * 100)):02d}"
            active_indices = np.flatnonzero(active)
            n_components, labels = connected_components(graph[active][:, active], directed=False)
            component_sizes = np.bincount(labels) if labels.size else np.empty(0, dtype=int)
            record[f"{prefix}_support_fraction"] = float(active.mean())
            record[f"{prefix}_log_components"] = float(np.log1p(n_components))
            record[f"{prefix}_largest_component_fraction"] = float(
                component_sizes.max() / active_indices.size if active_indices.size else 0.0
            )

            edge_crossing = np.logical_xor(active[edge_source], active[edge_target])
            boundary_nodes = np.unique(
                np.concatenate([edge_source[edge_crossing], edge_target[edge_crossing]])
            )
            record[f"{prefix}_boundary_edge_fraction"] = float(
                edge_crossing.mean() if edge_crossing.size else 0.0
            )
            record[f"{prefix}_thickness_proxy"] = float(
                active_indices.size / max(boundary_nodes.size, 1) / np.sqrt(active.size)
            )

            active_weight = measure[active]
            active_weight /= np.maximum(active_weight.sum(), 1.0e-12)
            active_coordinates = coordinates[active]
            centroid = np.sum(active_coordinates * active_weight[:, None], axis=0)
            centered = active_coordinates - centroid
            covariance = (centered * active_weight[:, None]).T @ centered
            eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
            total = max(float(eigenvalues.sum()), 1.0e-12)
            record[f"{prefix}_spatial_extent"] = total
            record[f"{prefix}_elongation"] = float(
                (eigenvalues[0] - eigenvalues[-1]) / total
            )
            radius = np.linalg.norm(centered, axis=1)
            record[f"{prefix}_radius_q50"] = _weighted_quantile(radius, active_weight, 0.50)
            record[f"{prefix}_radius_q90"] = _weighted_quantile(radius, active_weight, 0.90)

        if gamma is not None and global_scale is not None and local_scale is not None:
            feature_weights = np.maximum(gamma[:, prototype], 0.0)
            for name, values in (("global_scale", global_scale), ("local_scale", local_scale)):
                for level in (0.25, 0.50, 0.75):
                    record[f"{name}_q{int(level * 100):02d}"] = _weighted_quantile(
                        values, feature_weights, level
                    )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _prototype_relations(
    z: np.ndarray,
    measures: np.ndarray,
    supports: np.ndarray,
    levels: tuple[float, ...],
    coordinates: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    level_lookup = {round(level, 8): index for index, level in enumerate(levels)}
    q50 = level_lookup[min(level_lookup, key=lambda value: abs(value - 0.50))]
    q75 = level_lookup[min(level_lookup, key=lambda value: abs(value - 0.75))]
    names = (
        "field_correlation",
        "positive_mass_overlap",
        "support_jaccard_50",
        "support_jaccard_75",
        "support_adjacency_50",
        "centroid_proximity",
        "containment_50_in_75",
    )
    n_prototypes = z.shape[1]
    relations = np.empty((n_prototypes, n_prototypes, len(names)), dtype=np.float64)
    centroids = measures.T @ coordinates
    centroid_distance = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :], axis=2
    )
    positive_distances = centroid_distance[centroid_distance > 1.0e-12]
    distance_scale = float(np.median(positive_distances)) if positive_distances.size else 1.0

    for left in range(n_prototypes):
        for right in range(n_prototypes):
            support50_left = supports[q50, :, left]
            support50_right = supports[q50, :, right]
            support75_left = supports[q75, :, left]
            support75_right = supports[q75, :, right]
            union50 = np.logical_or(support50_left, support50_right).sum()
            union75 = np.logical_or(support75_left, support75_right).sum()
            adjacency_edges = np.logical_or(
                support50_left[edge_source] & support50_right[edge_target],
                support50_right[edge_source] & support50_left[edge_target],
            )
            incident_edges = np.logical_or(
                np.logical_or(support50_left[edge_source], support50_left[edge_target]),
                np.logical_or(support50_right[edge_source], support50_right[edge_target]),
            )
            relations[left, right] = (
                _safe_correlation(z[:, left], z[:, right]),
                float(np.minimum(measures[:, left], measures[:, right]).sum()),
                float(np.logical_and(support50_left, support50_right).sum() / max(union50, 1)),
                float(np.logical_and(support75_left, support75_right).sum() / max(union75, 1)),
                float(adjacency_edges.sum() / max(incident_edges.sum(), 1)),
                float(np.exp(-centroid_distance[left, right] / max(distance_scale, 1.0e-12))),
                float(
                    np.logical_and(support50_left, support75_right).sum()
                    / max(support50_left.sum(), 1)
                ),
            )
    return relations, names


def _relational_similarity(
    left: np.ndarray,
    right: np.ndarray,
    forward: np.ndarray,
    reverse: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    n_left, n_right = forward.shape
    result = np.zeros((n_left, n_right), dtype=np.float64)
    for channel, weight in enumerate(weights):
        left_channel = left[:, :, channel]
        right_channel = right[:, :, channel]
        for i in range(n_left):
            for j in range(n_right):
                right_in_left = forward @ right_channel[j]
                left_in_right = reverse @ left_channel[i]
                score = 0.5 * (
                    _safe_correlation(left_channel[i], right_in_left)
                    + _safe_correlation(right_channel[j], left_in_right)
                )
                result[i, j] += weight * (score + 1.0) / 2.0
    return np.clip(result, 0.0, 1.0)


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
    index = min(int(np.searchsorted(cumulative, target)), order.size - 1)
    return float(values[order[index]])


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
