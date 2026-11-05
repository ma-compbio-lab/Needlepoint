from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import kneighbors_graph


@dataclass(frozen=True)
class RegularGridGeometry:
    """Validated integer-lattice representation of two-dimensional coordinates."""

    x_index: np.ndarray
    y_index: np.ndarray
    origin_x: float
    origin_y: float
    pitch_x: float
    pitch_y: float
    n_x: int
    n_y: int
    max_error_x: float
    max_error_y: float

    @property
    def n_observed(self) -> int:
        return int(self.x_index.size)

    @property
    def n_grid_cells(self) -> int:
        return int(self.n_x * self.n_y)

    @property
    def flat_index(self) -> np.ndarray:
        return self.y_index.astype(np.int64) * self.n_x + self.x_index


@dataclass(frozen=True)
class SpatialGraphConfig:
    """Explicit graph-construction contract in coordinate/physical units.

    ``radius-knn`` is the scalable cross-resolution option: it retains at most
    ``n_neighbors`` nearest neighbors, rejects every edge beyond ``radius``,
    and uses a fixed distance bandwidth. This avoids both long gap-spanning
    kNN edges and phase-sublattice aliasing from fixed long lattice steps.
    """

    construction: str = "auto"
    n_neighbors: int = 16
    radius: float | None = None
    distance_bandwidth: float | None = None
    weighted: bool = True
    row_normalized: bool = False

    def __post_init__(self) -> None:
        if self.construction not in {
            "auto",
            "none",
            "knn",
            "radius-knn",
            "regular-grid",
            "native-grid",
        }:
            raise ValueError(
                "graph construction must be auto, none, knn, radius-knn, "
                "regular-grid, or native-grid"
            )
        if self.n_neighbors <= 0:
            raise ValueError("graph n_neighbors must be positive")
        if self.radius is not None and (
            not np.isfinite(self.radius) or self.radius <= 0.0
        ):
            raise ValueError("graph radius must be positive and finite")
        if self.distance_bandwidth is not None and (
            not np.isfinite(self.distance_bandwidth)
            or self.distance_bandwidth <= 0.0
        ):
            raise ValueError("graph distance bandwidth must be positive and finite")
        if self.construction == "radius-knn" and self.radius is None:
            raise ValueError("radius-knn construction requires a graph radius")
        if self.row_normalized:
            raise ValueError(
                "Audited LBM graphs must remain symmetric; row normalization "
                "would make undirected edge weights asymmetric"
            )

    def resolved_construction(self, *, graph_model: str | None = None) -> str:
        if self.construction != "auto":
            return self.construction
        return "none" if graph_model == "none" else "knn"

    def to_dict(self, *, graph_model: str | None = None) -> dict[str, Any]:
        return {
            "requested_construction": self.construction,
            "effective_construction": self.resolved_construction(
                graph_model=graph_model
            ),
            "n_neighbors": int(self.n_neighbors),
            "radius": None if self.radius is None else float(self.radius),
            "distance_bandwidth": (
                None
                if self.distance_bandwidth is None
                else float(self.distance_bandwidth)
            ),
            "weighted": bool(self.weighted),
            "row_normalized": bool(self.row_normalized),
        }


def _validate_coords(coords: np.ndarray, *, allow_singleton: bool = False) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    minimum = 1 if allow_singleton else 2
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] < minimum:
        raise ValueError(f"Expected coordinates with shape n_locations x 2 and n >= {minimum}")
    if not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates contain non-finite values")
    return coords


def _infer_regular_axis(
    values: np.ndarray,
    *,
    axis_name: str,
) -> tuple[np.ndarray, float, float, int, float]:
    values = np.asarray(values, dtype=np.float64)
    observed = np.unique(values)
    if observed.size < 2:
        raise ValueError(f"Regular-grid {axis_name} requires two distinct values")
    differences = np.diff(observed)
    candidates = np.unique(np.sort(differences[differences > 0.0])[:128])
    pitch = None
    for candidate in candidates:
        multiples = differences / candidate
        if np.allclose(multiples, np.rint(multiples), rtol=0.0, atol=0.05):
            pitch = float(candidate)
            break
    if pitch is None:
        raise ValueError(
            f"{axis_name} coordinates are not an integer-multiple lattice"
        )
    origin = float(observed[0])
    indices = np.rint((values - origin) / pitch).astype(np.int64)
    errors = np.abs(values - (origin + indices * pitch))
    max_error = float(errors.max(initial=0.0))
    tolerance = max(1.0e-7, pitch * 0.01)
    if max_error > tolerance:
        raise ValueError(
            f"{axis_name} coordinates deviate from the inferred lattice by "
            f"{max_error:.6g}, exceeding {tolerance:.6g}"
        )
    return indices, origin, pitch, int(indices.max(initial=-1) + 1), max_error


def infer_regular_grid(coords: np.ndarray) -> RegularGridGeometry:
    """Infer a masked rectangular lattice and reject duplicate spatial cells."""
    coords = _validate_coords(coords)
    x_index, origin_x, pitch_x, n_x, max_error_x = _infer_regular_axis(
        coords[:, 0], axis_name="x"
    )
    y_index, origin_y, pitch_y, n_y, max_error_y = _infer_regular_axis(
        coords[:, 1], axis_name="y"
    )
    geometry = RegularGridGeometry(
        x_index=x_index,
        y_index=y_index,
        origin_x=origin_x,
        origin_y=origin_y,
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        n_x=n_x,
        n_y=n_y,
        max_error_x=max_error_x,
        max_error_y=max_error_y,
    )
    _, counts = np.unique(geometry.flat_index, return_counts=True)
    duplicate_count = int(np.sum(counts - 1))
    if duplicate_count:
        raise ValueError(
            f"Regular-grid coordinates contain {duplicate_count} duplicate observation(s)"
        )
    return geometry


def row_normalize(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    row_counts = np.diff(matrix.indptr)
    row_ids = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), row_counts)
    scale = np.ones(matrix.shape[0], dtype=np.float32)
    nonzero = row_sums > 0
    scale[nonzero] = 1.0 / row_sums[nonzero]
    matrix.data *= scale[row_ids]
    return matrix


def _distance_graph(
    coords: np.ndarray,
    *,
    n_neighbors: int,
) -> sparse.csr_matrix:
    k = min(int(n_neighbors), coords.shape[0] - 1)
    graph = kneighbors_graph(
        coords,
        n_neighbors=k,
        mode="distance",
        include_self=False,
    )
    # Distance is symmetric for a retained pair; maximum takes the union of
    # directed kNN edges without dropping one-sided boundary neighbors.
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def _apply_distance_weights(
    distance_graph: sparse.csr_matrix,
    *,
    weighted: bool,
    distance_bandwidth: float | None,
) -> tuple[sparse.csr_matrix, float | None]:
    graph = distance_graph.copy().astype(np.float32)
    if not weighted:
        graph.data = np.ones_like(graph.data, dtype=np.float32)
        return graph, None
    positive = graph.data[graph.data > 0.0]
    inferred = float(np.median(positive)) if positive.size else 1.0
    bandwidth = max(
        inferred if distance_bandwidth is None else float(distance_bandwidth),
        1.0e-12,
    )
    graph.data = np.exp(-graph.data / bandwidth).astype(np.float32, copy=False)
    return graph, bandwidth


def build_knn_graph(
    coords: np.ndarray,
    n_neighbors: int = 8,
    weighted: bool = True,
    row_normalized: bool = True,
    *,
    max_distance: float | None = None,
    distance_bandwidth: float | None = None,
) -> sparse.csr_matrix:
    """Build a symmetric kNN graph, optionally capped in physical distance."""
    coords = _validate_coords(coords).astype(np.float32, copy=False)
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    if max_distance is not None and (
        not np.isfinite(max_distance) or max_distance <= 0.0
    ):
        raise ValueError("max_distance must be positive and finite")
    distances = _distance_graph(coords, n_neighbors=n_neighbors)
    if max_distance is not None and distances.nnz:
        distances.data[distances.data > float(max_distance)] = 0.0
        distances.eliminate_zeros()
    graph, _ = _apply_distance_weights(
        distances,
        weighted=weighted,
        distance_bandwidth=distance_bandwidth,
    )
    if row_normalized:
        graph = row_normalize(graph)
    return graph


_GRID_16_OFFSETS = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (-2, 0),
    (2, 0),
    (0, -2),
    (0, 2),
    (2, 1),
    (-1, 2),
    (-2, -1),
    (1, -2),
)

_GRID_4_OFFSETS = ((1, 0), (0, 1))


def build_regular_grid_graph(
    coords: np.ndarray,
    *,
    weighted: bool = True,
    row_normalized: bool = False,
    distance_bandwidth: float | None = None,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build a local 16-neighbor stencil that always includes one-cell edges.

    This construction is useful within one resolution. Its physical reach
    changes with lattice pitch, so ``radius-knn`` with a fixed radius and
    bandwidth is preferred for comparisons across resolutions.
    """
    geometry = infer_regular_grid(coords)
    flat = geometry.flat_index
    lookup = np.full(geometry.n_grid_cells, -1, dtype=np.int64)
    lookup[flat] = np.arange(geometry.n_observed, dtype=np.int64)
    rows_all = np.arange(geometry.n_observed, dtype=np.int64)
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    distances: list[np.ndarray] = []
    for dx, dy in _GRID_16_OFFSETS:
        if dy < 0 or (dy == 0 and dx < 0):
            continue
        target_x = geometry.x_index + dx
        target_y = geometry.y_index + dy
        in_bounds = (
            (target_x >= 0)
            & (target_x < geometry.n_x)
            & (target_y >= 0)
            & (target_y < geometry.n_y)
        )
        source = rows_all[in_bounds]
        target = lookup[target_y[in_bounds] * geometry.n_x + target_x[in_bounds]]
        present = target >= 0
        source = source[present]
        target = target[present]
        if source.size:
            sources.append(source)
            targets.append(target)
            distances.append(
                np.full(
                    source.size,
                    np.hypot(dx * geometry.pitch_x, dy * geometry.pitch_y),
                    dtype=np.float32,
                )
            )
    if sources:
        upper_u = np.concatenate(sources)
        upper_v = np.concatenate(targets)
        upper_distance = np.concatenate(distances)
        distance_graph = sparse.csr_matrix(
            (
                np.concatenate([upper_distance, upper_distance]),
                (np.concatenate([upper_u, upper_v]), np.concatenate([upper_v, upper_u])),
            ),
            shape=(geometry.n_observed, geometry.n_observed),
            dtype=np.float32,
        )
    else:
        distance_graph = sparse.csr_matrix(
            (geometry.n_observed, geometry.n_observed), dtype=np.float32
        )
    graph, bandwidth = _apply_distance_weights(
        distance_graph,
        weighted=weighted,
        distance_bandwidth=distance_bandwidth,
    )
    if row_normalized:
        graph = row_normalize(graph)
    audit = audit_spatial_graph(
        graph,
        coords,
        requested_construction="regular-grid",
        effective_construction="regular-grid",
        distance_bandwidth=bandwidth,
    )
    audit.update(
        {
            "pitch_x": float(geometry.pitch_x),
            "pitch_y": float(geometry.pitch_y),
            "grid_width": int(geometry.n_x),
            "grid_height": int(geometry.n_y),
            "lattice_occupancy": float(
                geometry.n_observed / max(geometry.n_grid_cells, 1)
            ),
            "fixed_coordinate_distance_support": False,
        }
    )
    return graph, audit


def build_native_grid_graph(
    coords: np.ndarray,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build the TEMI acquisition-lattice graph with immediate 4-neighbors."""

    geometry = infer_regular_grid(coords)
    flat = geometry.flat_index
    lookup = np.full(geometry.n_grid_cells, -1, dtype=np.int64)
    lookup[flat] = np.arange(geometry.n_observed, dtype=np.int64)
    rows_all = np.arange(geometry.n_observed, dtype=np.int64)
    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for dx, dy in _GRID_4_OFFSETS:
        target_x = geometry.x_index + dx
        target_y = geometry.y_index + dy
        in_bounds = (
            (target_x >= 0)
            & (target_x < geometry.n_x)
            & (target_y >= 0)
            & (target_y < geometry.n_y)
        )
        source = rows_all[in_bounds]
        target = lookup[target_y[in_bounds] * geometry.n_x + target_x[in_bounds]]
        present = target >= 0
        if np.any(present):
            sources.append(source[present])
            targets.append(target[present])
    if sources:
        upper_u = np.concatenate(sources)
        upper_v = np.concatenate(targets)
        graph = sparse.csr_matrix(
            (
                np.ones(2 * upper_u.size, dtype=np.float32),
                (
                    np.concatenate([upper_u, upper_v]),
                    np.concatenate([upper_v, upper_u]),
                ),
            ),
            shape=(geometry.n_observed, geometry.n_observed),
        )
    else:
        graph = sparse.csr_matrix(
            (geometry.n_observed, geometry.n_observed), dtype=np.float32
        )
    audit = audit_spatial_graph(
        graph,
        coords,
        requested_construction="native-grid",
        effective_construction="native-grid",
    )
    audit.update(
        {
            "pitch_x": float(geometry.pitch_x),
            "pitch_y": float(geometry.pitch_y),
            "grid_width": int(geometry.n_x),
            "grid_height": int(geometry.n_y),
            "lattice_occupancy": float(
                geometry.n_observed / max(geometry.n_grid_cells, 1)
            ),
            "fixed_coordinate_distance_support": False,
            "native_neighbor_count": 4,
        }
    )
    return graph, audit


def audit_spatial_graph(
    graph: sparse.spmatrix,
    coords: np.ndarray,
    *,
    requested_construction: str,
    effective_construction: str,
    distance_bandwidth: float | None = None,
    max_distance_samples: int = 200_000,
) -> dict[str, Any]:
    """Validate symmetry/weights and report exact connectivity diagnostics."""
    coords = _validate_coords(coords, allow_singleton=True)
    graph = sparse.csr_matrix(graph, dtype=np.float32)
    expected = (coords.shape[0], coords.shape[0])
    if graph.shape != expected:
        raise ValueError(f"Spatial graph has shape {graph.shape}; expected {expected}")
    if graph.diagonal().any():
        raise ValueError("Spatial graph must not contain self edges")
    if not np.isfinite(graph.data).all() or np.any(graph.data < 0.0):
        raise ValueError("Spatial graph must contain finite nonnegative weights")
    difference = graph - graph.T
    difference.eliminate_zeros()
    if difference.nnz and np.max(np.abs(difference.data)) > 1.0e-6:
        raise ValueError("Spatial graph must be symmetric")
    degree = np.diff(graph.indptr)
    strengths = np.asarray(graph.sum(axis=1)).ravel()
    if graph.nnz:
        n_components, component_labels = connected_components(
            graph, directed=False, return_labels=True
        )
        component_sizes = np.bincount(component_labels, minlength=n_components)
        largest_component = int(component_sizes.max(initial=0))
    else:
        n_components = int(coords.shape[0])
        largest_component = 1 if coords.shape[0] else 0
    upper = sparse.triu(graph, k=1, format="coo")
    if upper.nnz:
        if upper.nnz > max_distance_samples:
            sample = np.linspace(
                0,
                upper.nnz - 1,
                max_distance_samples,
                dtype=np.int64,
            )
            edge_u = upper.row[sample]
            edge_v = upper.col[sample]
        else:
            edge_u = upper.row
            edge_v = upper.col
        edge_distances = np.linalg.norm(coords[edge_u] - coords[edge_v], axis=1)
    else:
        edge_distances = np.empty(0, dtype=np.float64)
    n_nodes = int(coords.shape[0])
    isolates = int(np.count_nonzero(degree == 0))
    return {
        "requested_construction": requested_construction,
        "effective_construction": effective_construction,
        "n_spots": n_nodes,
        "undirected_edge_count": int(upper.nnz),
        "minimum_observed_degree": int(degree.min()) if degree.size else 0,
        "maximum_observed_degree": int(degree.max()) if degree.size else 0,
        "mean_observed_degree": float(degree.mean()) if degree.size else 0.0,
        "isolated_observation_count": isolates,
        "isolated_observation_fraction": float(isolates / max(n_nodes, 1)),
        "connected_component_count": int(n_components),
        "largest_component_count": int(largest_component),
        "largest_component_fraction": float(largest_component / max(n_nodes, 1)),
        "minimum_node_strength": float(strengths.min()) if strengths.size else 0.0,
        "median_positive_node_strength": float(
            np.median(strengths[strengths > 0.0])
        )
        if np.any(strengths > 0.0)
        else 0.0,
        "maximum_node_strength": float(strengths.max()) if strengths.size else 0.0,
        "distance_bandwidth": (
            None if distance_bandwidth is None else float(distance_bandwidth)
        ),
        "sampled_edge_count_for_distance_audit": int(edge_distances.size),
        "realized_edge_distance_minimum": (
            float(edge_distances.min()) if edge_distances.size else None
        ),
        "realized_edge_distance_median": (
            float(np.median(edge_distances)) if edge_distances.size else None
        ),
        "realized_edge_distance_95th_percentile": (
            float(np.quantile(edge_distances, 0.95)) if edge_distances.size else None
        ),
        "realized_edge_distance_maximum": (
            float(edge_distances.max()) if edge_distances.size else None
        ),
    }


def validate_graph_audit(
    audit: dict[str, Any],
    *,
    max_isolate_fraction: float,
    min_largest_component_fraction: float,
    min_radius_coverage: float = 0.0,
) -> None:
    """Apply explicit production connectivity gates to a graph audit."""
    if not 0.0 <= max_isolate_fraction <= 1.0:
        raise ValueError("max_isolate_fraction must be in [0, 1]")
    if not 0.0 <= min_largest_component_fraction <= 1.0:
        raise ValueError("min_largest_component_fraction must be in [0, 1]")
    if not 0.0 <= min_radius_coverage <= 1.0:
        raise ValueError("min_radius_coverage must be in [0, 1]")
    if audit["effective_construction"] == "none":
        return
    if audit["isolated_observation_fraction"] > max_isolate_fraction:
        raise ValueError(
            "Spatial graph isolate fraction exceeds the configured limit: "
            f"{audit['isolated_observation_fraction']:.6g} > {max_isolate_fraction:.6g}"
        )
    if audit["largest_component_fraction"] < min_largest_component_fraction:
        raise ValueError(
            "Spatial graph largest-component fraction is below the configured limit: "
            f"{audit['largest_component_fraction']:.6g} < "
            f"{min_largest_component_fraction:.6g}"
        )
    coverage = audit.get("radius_coverage_95th_fraction")
    if coverage is not None and coverage < min_radius_coverage:
        raise ValueError(
            "Spatial graph retained edges do not cover enough of the configured "
            "physical radius: "
            f"{coverage:.6g} < {min_radius_coverage:.6g}. Increase "
            "--graph-neighbors or reduce --graph-radius."
        )


def build_spatial_graph(
    coords: np.ndarray,
    *,
    config: SpatialGraphConfig,
    graph_model: str | None = None,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Build an explicit, audited graph without silent construction fallback."""
    effective = config.resolved_construction(graph_model=graph_model)
    coords = _validate_coords(coords, allow_singleton=effective == "none")
    if effective == "none":
        graph = sparse.csr_matrix((coords.shape[0], coords.shape[0]), dtype=np.float32)
        audit = audit_spatial_graph(
            graph,
            coords,
            requested_construction=config.construction,
            effective_construction="none",
        )
        audit["fixed_coordinate_distance_support"] = True
        return graph, audit
    if effective == "regular-grid":
        graph, audit = build_regular_grid_graph(
            coords,
            weighted=config.weighted,
            row_normalized=config.row_normalized,
            distance_bandwidth=config.distance_bandwidth,
        )
        audit["requested_construction"] = config.construction
        return graph, audit
    if effective == "native-grid":
        graph, audit = build_native_grid_graph(coords)
        audit["requested_construction"] = config.construction
        return graph, audit
    radius = config.radius if effective == "radius-knn" else None
    distance_graph = _distance_graph(
        coords.astype(np.float32, copy=False),
        n_neighbors=config.n_neighbors,
    )
    if radius is not None and distance_graph.nnz:
        distance_graph.data[distance_graph.data > float(radius)] = 0.0
        distance_graph.eliminate_zeros()
    default_bandwidth = (
        float(radius) / 2.0
        if effective == "radius-knn" and config.distance_bandwidth is None
        else config.distance_bandwidth
    )
    graph, bandwidth = _apply_distance_weights(
        distance_graph,
        weighted=config.weighted,
        distance_bandwidth=default_bandwidth,
    )
    if config.row_normalized:
        graph = row_normalize(graph)
    audit = audit_spatial_graph(
        graph,
        coords,
        requested_construction=config.construction,
        effective_construction=effective,
        distance_bandwidth=bandwidth,
    )
    audit.update(
        {
            "n_neighbors": int(config.n_neighbors),
            "maximum_allowed_edge_distance": (
                None if radius is None else float(radius)
            ),
            "fixed_coordinate_distance_support": bool(effective == "radius-knn"),
        }
    )
    realized_95th = audit["realized_edge_distance_95th_percentile"]
    audit["radius_coverage_95th_fraction"] = (
        None
        if radius is None or realized_95th is None
        else float(realized_95th / radius)
    )
    return graph, audit


def labels_to_indicator(labels: np.ndarray, n_labels: int | None = None) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if n_labels is None:
        n_labels = int(labels.max()) + 1 if labels.size else 0
    indicator = np.zeros((labels.size, n_labels), dtype=np.float32)
    if labels.size:
        indicator[np.arange(labels.size), labels] = 1.0
    return indicator


def moran_i(values: np.ndarray, adjacency: sparse.spmatrix) -> float:
    if not sparse.isspmatrix(adjacency):
        adjacency = sparse.csr_matrix(adjacency)
    x = np.asarray(values)
    if x.dtype.kind in {"O", "S", "U"}:
        _, x = np.unique(x, return_inverse=True)
    x = x.astype(np.float64, copy=False)
    centered = x - x.mean()
    denom = float(np.square(centered).sum())
    w_sum = float(adjacency.sum())
    if denom <= 0.0 or w_sum <= 0.0:
        return float("nan")
    numerator = float(centered @ (adjacency @ centered))
    return float((x.size / w_sum) * (numerator / denom))
