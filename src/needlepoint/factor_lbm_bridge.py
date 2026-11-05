"""Conservative continuous bridge from shared GP factors to the V-SKI LBM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from needlepoint.shared_spatial_factorization import spatially_balanced_sample


@dataclass
class FactorLBMBridge:
    """Continuous embeddings and soft initial responsibilities for the LBM."""

    feature_ids: np.ndarray
    feature_embedding: np.ndarray
    location_embedding: np.ndarray
    initial_peak_responsibilities: np.ndarray
    initial_domain_responsibilities: np.ndarray
    spots: pd.DataFrame


def build_factor_lbm_bridge(
    factor_dir: Path,
    output_dir: Path,
    *,
    n_peak_modules: int = 20,
    n_domains: int = 12,
    feature_neighbors: int = 15,
    sample_spots: int = 60_000,
    random_state: int = 0,
) -> FactorLBMBridge:
    """Build Pi/H embeddings without hard-locking factors to LBM components."""
    factor_dir = Path(factor_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loadings = pd.read_parquet(factor_dir / "shared_factor_loadings.parquet")
    pi_columns = sorted(column for column in loadings if column.startswith("Pi_"))
    if not pi_columns:
        raise ValueError("Shared-factor loadings contain no Pi columns")
    feature_embedding = loadings[pi_columns].to_numpy(dtype=np.float32)
    feature_embedding = _standardize_embedding(feature_embedding, center=False)
    if n_peak_modules > feature_embedding.shape[0]:
        raise ValueError("n_peak_modules exceeds feature count")
    peak_model = KMeans(
        n_clusters=n_peak_modules, n_init=20, random_state=random_state
    ).fit(feature_embedding)
    peak_distances = peak_model.transform(feature_embedding).astype(np.float64)
    peak_responsibilities = _soft_responsibilities(peak_distances)

    location_embedding = load_factor_maps(factor_dir, normalized=False)
    location_embedding = _standardize_embedding(location_embedding, center=True)
    if n_domains > location_embedding.shape[0]:
        raise ValueError("n_domains exceeds spot count")
    spots = pd.read_csv(factor_dir / "shared_factor_spots.csv.gz")
    if spots.shape[0] != location_embedding.shape[0]:
        raise ValueError("Factor-map rows and spot rows differ")
    sample = spatially_balanced_sample(
        spots[["x", "y"]].to_numpy(dtype=np.float64),
        min(sample_spots, spots.shape[0]),
    )
    domain_model = MiniBatchKMeans(
        n_clusters=n_domains,
        batch_size=min(8192, sample.size),
        n_init=10,
        random_state=random_state,
    ).fit(location_embedding[sample])
    domain_distances = domain_model.transform(location_embedding).astype(np.float64)
    domain_responsibilities = _soft_responsibilities(domain_distances)

    feature_graph = _embedding_knn_graph(feature_embedding, feature_neighbors)
    sparse.save_npz(output_dir / "factor_feature_neighbor_graph.npz", feature_graph)
    np.save(output_dir / "factor_feature_embedding.npy", feature_embedding)
    np.save(output_dir / "factor_location_embedding.npy", location_embedding)
    np.save(
        output_dir / "initial_peak_responsibilities.npy",
        peak_responsibilities.astype(np.float32),
    )
    np.save(
        output_dir / "initial_domain_responsibilities.npy",
        domain_responsibilities.astype(np.float32),
    )
    pd.DataFrame(
        {
            "gene": loadings["gene"].astype(str),
            "feature_position": np.arange(loadings.shape[0], dtype=np.int64),
            **{column: feature_embedding[:, index] for index, column in enumerate(pi_columns)},
        }
    ).to_csv(output_dir / "factor_feature_embedding.csv", index=False)
    spots.to_csv(output_dir / "factor_location_spots.csv.gz", index=False)
    manifest = {
        "format_version": 1,
        "source_factor_dir": str(factor_dir),
        "n_features": int(feature_embedding.shape[0]),
        "n_spots": int(location_embedding.shape[0]),
        "n_shared_factors": int(feature_embedding.shape[1]),
        "n_peak_modules": int(n_peak_modules),
        "n_domains": int(n_domains),
        "feature_neighbors": int(min(feature_neighbors, max(feature_embedding.shape[0] - 1, 1))),
        "sample_spots": int(sample.size),
        "random_state": int(random_state),
        "semantics": {
            "feature_embedding": "row-normalized shared-factor composition Pi",
            "location_embedding": "column-standardized posterior mean factor activity H",
            "responsibilities": (
                "soft k-means distance initializations; the LBM may override them"
            ),
        },
    }
    (output_dir / "factor_lbm_bridge_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return FactorLBMBridge(
        feature_ids=loadings["gene"].astype(str).to_numpy(),
        feature_embedding=feature_embedding,
        location_embedding=location_embedding,
        initial_peak_responsibilities=peak_responsibilities.astype(np.float32),
        initial_domain_responsibilities=domain_responsibilities.astype(np.float32),
        spots=spots,
    )


def load_factor_lbm_bridge(path: Path) -> FactorLBMBridge:
    path = Path(path)
    feature_table = pd.read_csv(path / "factor_feature_embedding.csv")
    spots = pd.read_csv(path / "factor_location_spots.csv.gz")
    return FactorLBMBridge(
        feature_ids=feature_table["gene"].astype(str).to_numpy(),
        feature_embedding=np.load(path / "factor_feature_embedding.npy"),
        location_embedding=np.load(path / "factor_location_embedding.npy", mmap_mode="r"),
        initial_peak_responsibilities=np.load(path / "initial_peak_responsibilities.npy"),
        initial_domain_responsibilities=np.load(path / "initial_domain_responsibilities.npy"),
        spots=spots,
    )


def load_factor_maps(factor_dir: Path, *, normalized: bool = False) -> np.ndarray:
    map_dir = Path(factor_dir) / "shared_factor_maps"
    index = pd.read_csv(map_dir / "shared_factor_map_chunks.csv").sort_values("chunk_id")
    if index.empty:
        raise ValueError("Shared-factor map index is empty")
    n_spots = int(index["spot_stop"].max())
    n_factors = int(index["n_factors"].iloc[0])
    output = np.empty((n_spots, n_factors), dtype=np.float32)
    key = "normalized_mean" if normalized else "posterior_mean"
    expected_start = 0
    for row in index.itertuples(index=False):
        if int(row.spot_start) != expected_start:
            raise ValueError("Shared-factor map chunks are not contiguous")
        with np.load(map_dir / row.file) as chunk:
            values = np.asarray(chunk[key], dtype=np.float32)
        expected = (int(row.spot_stop) - int(row.spot_start), n_factors)
        if values.shape != expected:
            raise ValueError(f"Unexpected factor-map shape in {row.file}: {values.shape}")
        output[int(row.spot_start) : int(row.spot_stop)] = values
        expected_start = int(row.spot_stop)
    if expected_start != n_spots:
        raise ValueError("Shared-factor map chunks do not cover all spots")
    return output


def _standardize_embedding(values: np.ndarray, *, center: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if not center:
        return values
    values = values - values.mean(axis=0, keepdims=True)
    values = values / np.maximum(values.std(axis=0, keepdims=True), 1.0e-6)
    row_norm = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(row_norm, 1.0e-8)).astype(np.float32)


def _soft_responsibilities(distances: np.ndarray) -> np.ndarray:
    squared = np.square(np.asarray(distances, dtype=np.float64))
    nearest = np.partition(squared, min(1, squared.shape[1] - 1), axis=1)[:, 0]
    positive = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    temperature = float(np.median(positive)) if positive.size else 1.0
    logits = -squared / max(temperature, 1.0e-8)
    logits -= logits.max(axis=1, keepdims=True)
    responsibilities = np.exp(logits)
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities


def _embedding_knn_graph(embedding: np.ndarray, n_neighbors: int) -> sparse.csr_matrix:
    count = embedding.shape[0]
    neighbors = min(max(int(n_neighbors), 1), max(count - 1, 1))
    model = NearestNeighbors(n_neighbors=min(neighbors + 1, count)).fit(embedding)
    distances, indices = model.kneighbors(embedding)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    scale = np.median(distances[distances > 0.0]) if np.any(distances > 0.0) else 1.0
    weights = np.exp(-np.square(distances / max(scale, 1.0e-8)))
    rows = np.repeat(np.arange(count, dtype=np.int64), indices.shape[1])
    graph = sparse.csr_matrix((weights.ravel(), (rows, indices.ravel())), shape=(count, count))
    return graph.maximum(graph.T).tocsr()
