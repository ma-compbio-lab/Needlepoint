"""Opt-in full-support graph-heat sketches for prototype initialization.

This is NOT SpaGFT's author implementation and is NOT a new GP likelihood.
Only the initialization is approximate; downstream full-field EM is unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
from scipy.special import ive


@dataclass
class MultiscaleSketchSettings:
    pitch_um: float = 8.0
    scales_um: tuple[float, ...] = (16.0, 64.0, 256.0)
    dimensions_per_band: tuple[int, ...] = (256, 512, 1024)
    projection_sparsity: int = 4
    projection_seed: int = 20260922
    filter_tolerance: float = 1e-5
    max_polynomial_degree: int = 2048
    feature_chunk_size: int = 32
    audit_features: int = 512
    audit_pairs: int = 2000
    distance_error_p95_max: float = 0.10
    device: str = "cuda"

    def validate(self):
        if self.pitch_um <= 0 or self.feature_chunk_size < 1:
            raise ValueError("Positive pitch and feature chunk required")
        if len(self.scales_um) != 3 or not (0 < self.scales_um[0] < self.scales_um[1] < self.scales_um[2]):
            raise ValueError("Exactly three increasing positive physical scales required")
        dims = self.dimensions_per_band
        if not dims or tuple(sorted(set(dims))) != tuple(dims):
            raise ValueError("Sketch dimensions must be increasing and unique")
        if self.projection_sparsity < 1 or any(d % self.projection_sparsity or dims[-1] % d for d in dims):
            raise ValueError("Nested dimensions must divide the maximum and contain whole hash groups")
        if not 0 < self.filter_tolerance < 1 or self.max_polynomial_degree < 1:
            raise ValueError("Invalid polynomial error budget")
        if self.audit_features < 2 or self.audit_pairs < 1 or self.distance_error_p95_max <= 0:
            raise ValueError("Invalid distance audit")


def native_lattice_adjacency(coords: np.ndarray, pitch_um: float):
    """Exact four-neighbor support, with no missing/zero-expression node deletion."""
    xy = np.asarray(coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all() or len(xy) == 0:
        raise ValueError("Expected finite nonempty N by 2 coordinates")
    normalized = (xy - xy.min(axis=0)) / pitch_um
    grid = np.rint(normalized).astype(np.int64)
    if not np.allclose(normalized, grid, atol=1e-4, rtol=0):
        raise ValueError("Coordinates do not lie on the declared physical lattice")
    width = int(grid[:, 0].max()) + 2  # extra column prevents row-wrap edges
    keys = grid[:, 1] * width + grid[:, 0]
    order = np.argsort(keys)
    ordered = keys[order]
    if np.any(np.diff(ordered) == 0):
        raise ValueError("Duplicate lattice sites")
    left, right = [], []
    for delta in (1, width):
        loc = np.searchsorted(ordered, keys + delta)
        a = np.flatnonzero(loc < len(keys))
        a = a[ordered[loc[a]] == keys[a] + delta]
        left.append(a)
        right.append(order[loc[a]])
    a, b = np.concatenate(left), np.concatenate(right)
    adjacency = sparse.csr_matrix((np.ones(2 * len(a), dtype=np.float32),
        (np.r_[a, b], np.r_[b, a])), shape=(len(xy), len(xy)))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    n_components, components = sparse.csgraph.connected_components(adjacency)
    report = dict(n_spots=len(xy), n_undirected_edges=len(a), pitch_um=float(pitch_um),
        n_components=int(n_components), n_isolates=int((degree == 0).sum()),
        largest_component_fraction=float(np.bincount(components).max() / len(xy)),
        graph="native-four-neighbor", laplacian="(D-A)/pitch_um**2",
        excluded_spots=0, coordinate_sha256=hashlib.sha256(xy.tobytes()).hexdigest())
    return adjacency, degree, report


def heat_coefficients(sigma_um, pitch_um, tolerance=1e-5, max_degree=2048):
    """Chebyshev coefficients; Bessel tail bounds uniform operator error.

    lambda_max <= 8 / h^2, mapped operator S=2L/lambda_max-I.
    exp(-sigma^2 L/2) = exp[-a(S+I)], a=2 sigma^2/h^2.
    Absolute coefficients sum to one; their omitted sum bounds truncation.
    """
    a = 2.0 * (float(sigma_um) / float(pitch_um)) ** 2
    k = np.arange(max_degree + 1)
    magnitude = 2 * ive(k, a)
    magnitude[0] *= 0.5
    tail = np.maximum(0, 1 - np.cumsum(magnitude))
    valid = np.flatnonzero(tail <= tolerance)
    if not len(valid):
        raise ValueError("Heat filter exceeds polynomial degree budget")
    degree = max(1, int(valid[0]))
    coefficients = magnitude[:degree + 1] * np.where(k[:degree + 1] % 2, -1, 1)
    return coefficients, dict(sigma_um=float(sigma_um), degree=degree,
        uniform_absolute_error_bound=float(tail[degree]), spectral_upper_bound=8 / pitch_um**2)


def torch_csr(matrix, device, dtype=None):
    import torch
    return torch.sparse_csr_tensor(torch.as_tensor(matrix.indptr, device=device),
        torch.as_tensor(matrix.indices, device=device),
        torch.as_tensor(matrix.data, device=device, dtype=dtype or torch.float32),
        size=matrix.shape, device=device)


def heat_bands(values, adjacency, degree, coefficients):
    """Three heat filters in one recurrence, followed by telescoping bands."""
    import torch
    def operator(x):
        return (degree[:, None] * x - torch.sparse.mm(adjacency, x)) * 0.25 - x
    previous = values
    current = operator(values)
    smooth = [float(c[0]) * previous + float(c[1]) * current for c in coefficients]
    for k in range(2, max(map(len, coefficients))):
        following = 2 * operator(current) - previous
        for j, c in enumerate(coefficients):
            if k < len(c):
                smooth[j].add_(following, alpha=float(c[k]))
        previous, current = current, following
    fine, middle, coarse = smooth
    return (coarse, middle - coarse, fine - middle, values - fine)


def sparse_projection(n_spots, dimensions, sparsity, seed):
    """Four independent disjoint hash groups, preserving signed coefficients."""
    rng = np.random.default_rng(seed)
    rows, columns, data = [], [], []
    group_size = dimensions // sparsity
    for group in range(sparsity):
        rows.append(group * group_size + rng.integers(group_size, size=n_spots))
        columns.append(np.arange(n_spots))
        data.append(rng.choice(np.array([-1, 1], dtype=np.float32), n_spots) / np.sqrt(sparsity))
    return sparse.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
        shape=(dimensions, n_spots), dtype=np.float32)


def fold_sketch(embedding, target_per_band, sparsity=4):
    """Nested hash folding: larger sketches audit smaller ones without refiltering."""
    x = np.asarray(embedding)
    source_per_band = x.shape[1] // 4
    if source_per_band % target_per_band or target_per_band % sparsity:
        raise ValueError("Incompatible nested sketch dimensions")
    return x.reshape(len(x), 4, sparsity, source_per_band // target_per_band,
        target_per_band // sparsity).sum(axis=3).reshape(len(x), 4 * target_per_band)


def distance_audit(gram, embedding, n_pairs, seed):
    """Random + nearest-pattern pairs on a declared stratified gene audit set."""
    from scipy.spatial.distance import cdist
    count = len(embedding)
    exact2 = np.maximum(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2 * gram, 0)
    approximate2 = cdist(embedding, embedding, metric="sqeuclidean")
    tri = np.column_stack(np.triu_indices(count, 1))
    rng = np.random.default_rng(seed)
    random_pairs = tri[rng.choice(len(tri), min(n_pairs // 2, len(tri)), replace=False)]
    # Choose nearest pairs by exact multiscale geometry, not the sketch under test.
    rank = np.argsort(exact2[tri[:, 0], tri[:, 1]])
    near = tri[rank[:min(n_pairs - len(random_pairs), len(tri))]]
    pairs = np.unique(np.vstack([random_pairs, near]), axis=0)
    exact = np.sqrt(exact2[pairs[:, 0], pairs[:, 1]])
    approx = np.sqrt(approximate2[pairs[:, 0], pairs[:, 1]])
    active = exact > 1e-7 * max(float(exact.max()), 1e-12)
    if not active.any():
        raise ValueError("No nonconstant gene pairs for sketch audit")
    error = np.abs(approx[active] - exact[active]) / exact[active]
    return dict(n_pairs=int(active.sum()), n_audit_features=count,
        distance_error_median=float(np.median(error)), distance_error_p95=float(np.quantile(error, .95)),
        distance_error_max=float(error.max()), excluded_near_identical_pairs=int((~active).sum()),
        metric="signed unweighted multiscale Euclidean distance, not full-field precision distance")


def torch_kmeans_labels(matrix, n_clusters, seed, device, max_iter=100, tolerance=1e-4):
    """One reproducible k-means++ start on the selected Torch device."""
    import torch
    x = torch.as_tensor(np.ascontiguousarray(matrix), dtype=torch.float32, device=device)
    if not 1 < n_clusters <= len(x) or not torch.isfinite(x).all():
        raise ValueError("Invalid initialization input")
    rng = np.random.default_rng(seed)
    centers = [int(rng.integers(len(x)))]
    minimum = torch.full((len(x),), float('inf'), device=device)
    for _ in range(1, n_clusters):
        minimum = torch.minimum(minimum, (x - x[centers[-1]]).square().sum(1))
        probability = minimum.double().cpu().numpy()
        probability[centers] = 0
        available = np.setdiff1d(np.arange(len(x)), centers)
        centers.append(int(rng.choice(len(x), p=probability / probability.sum()))
            if probability.sum() > 0 else int(rng.choice(available)))
    centroids = x[centers].clone()
    norm = x.square().sum(1)[:, None]
    for _ in range(max_iter):
        distance = (norm + centroids.square().sum(1)[None] - 2 * x @ centroids.T).clamp_min_(0)
        labels = distance.argmin(1)
        counts = torch.bincount(labels, minlength=n_clusters)
        updated = torch.zeros_like(centroids)
        updated.index_add_(0, labels, x)
        updated /= counts.clamp_min(1)[:, None]
        empty = torch.nonzero(counts == 0).flatten()
        if len(empty):
            farthest = distance.min(1).values.argsort(descending=True)[:len(empty)]
            updated[empty] = x[farthest]
        change = float(torch.linalg.vector_norm(updated-centroids) /
            torch.linalg.vector_norm(centroids).clamp_min(1e-12))
        centroids = updated
        if change <= tolerance:
            break
    labels = (norm + centroids.square().sum(1)[None] - 2*x@centroids.T).argmin(1)
    return labels.cpu().numpy()


def prepare_multiscale_sketch(store, output_dir: Path, settings=None):
    import torch
    settings = settings or MultiscaleSketchSettings()
    settings.validate()
    output_dir = Path(output_dir)
    cache = store.prepare()
    config = json.loads(json.dumps(asdict(settings)))
    signature = hashlib.sha256(json.dumps(dict(config=config,
        source_signature=cache['source_signature']), sort_keys=True).encode()).hexdigest()
    marker = output_dir / 'multiscale_manifest.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous['signature'] != signature:
            raise ValueError("Incompatible existing multiscale cache")
        embedding = np.load(output_dir / 'initialization_embedding.npy')
        if hashlib.sha256(embedding.tobytes()).hexdigest() != previous['embedding_sha256']:
            raise ValueError("Corrupt multiscale embedding")
        return previous
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing incomplete sketch directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    device = torch.device(settings.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.cuda.reset_peak_memory_stats(device)
    adj, deg, graph = native_lattice_adjacency(store.spots[['x','y']].to_numpy(), settings.pitch_um)
    adjacency = torch_csr(adj, device)
    degree = torch.as_tensor(deg, device=device)
    filters = [heat_coefficients(s, settings.pitch_um, settings.filter_tolerance,
        settings.max_polynomial_degree) for s in settings.scales_um]
    print(json.dumps(dict(stage='graph_and_filters', graph=graph, filters=[f[1] for f in filters])), flush=True)
    maximum = settings.dimensions_per_band[-1]
    projections = [torch_csr(sparse_projection(store.n_spots, maximum,
        settings.projection_sparsity, settings.projection_seed + b), device) for b in range(4)]
    result = np.empty((store.n_features, 4 * maximum), dtype=np.float32)
    # Balanced amplitude quantiles + random genes; never annotation-derived.
    n_audit = min(settings.audit_features, store.n_features)
    order = np.argsort(store.amplitudes)
    audit_ids = order[np.linspace(0, store.n_features-1, n_audit//2, dtype=int)]
    available = np.setdiff1d(np.arange(store.n_features), audit_ids)
    audit_ids = np.sort(np.r_[audit_ids, np.random.default_rng(settings.projection_seed).choice(
        available, n_audit-len(audit_ids), replace=False)])
    audit_index = {int(g): i for i,g in enumerate(audit_ids)}
    audit_bands = [torch.empty((store.n_spots, n_audit), device=device) for _ in range(4)]
    all_rows = np.arange(store.n_spots)
    with torch.no_grad():
        for first in range(0, store.n_features, settings.feature_chunk_size):
            ids = np.arange(first, min(first + settings.feature_chunk_size, store.n_features))
            mean, _ = store.read_block(all_rows, ids)
            if not np.isfinite(mean).all():
                raise ValueError("Nonfinite field in sketch input")
            # Every shape has mean 1. Removing this common constant improves numerics;
            # it does not change between-gene distances, including on disconnected graphs.
            values = torch.as_tensor(mean, device=device) - 1.0
            bands = heat_bands(values, adjacency, degree, [f[0] for f in filters])
            for b, band in enumerate(bands):
                result[ids, b*maximum:(b+1)*maximum] = (
                    torch.sparse.mm(projections[b], band).T / np.sqrt(store.n_spots)).cpu().numpy()
                source = [j for j,g in enumerate(ids) if int(g) in audit_index]
                if source:
                    destination = [audit_index[int(ids[j])] for j in source]
                    audit_bands[b][:, destination] = band[:, source]
            print(json.dumps(dict(stage='sketch', genes_done=int(ids[-1]+1),
                genes_total=store.n_features, elapsed_seconds=time.time()-start)), flush=True)
        gram = np.zeros((n_audit,n_audit), dtype=np.float64)
        for b in audit_bands:
            # Float64 reductions preserve small distances between similar fields.
            for first in range(0, store.n_spots, 8192):
                chunk = b[first:first+8192].double()
                gram += (chunk.T @ chunk).cpu().numpy() / store.n_spots
        del audit_bands
    audits, chosen, embedding = [], None, None
    for dimensions in settings.dimensions_per_band:
        candidate = fold_sketch(result, dimensions, settings.projection_sparsity)
        audit = distance_audit(gram, candidate[audit_ids], settings.audit_pairs, settings.projection_seed)
        audit.update(dimensions_per_band=dimensions, total_dimensions=4*dimensions)
        audits.append(audit)
        if audit['distance_error_p95'] <= settings.distance_error_p95_max:
            chosen, embedding = dimensions, candidate
            break
    if embedding is None:
        (output_dir/'failed_distance_audit.json').write_text(json.dumps(audits,indent=2))
        raise ValueError("No candidate sketch passes the predeclared distance audit")
    if not np.isfinite(embedding).all():
        raise ValueError("Nonfinite sketch")
    np.save(output_dir/'initialization_embedding.npy', embedding)
    np.save(output_dir/'audit_feature_indices.npy', audit_ids)
    np.save(output_dir/'audit_multiscale_gram.npy', gram)
    manifest = dict(schema_version=1, signature=signature, settings=config,
        cache_source_signature=cache['source_signature'], n_spots=store.n_spots,
        n_features=store.n_features, graph=graph, filters=[f[1] for f in filters],
        selected_dimensions_per_band=chosen, audits=audits,
        audit_sampling='512 or fewer amplitude-stratified/random genes; no reference labels',
        feature_order_sha256=hashlib.sha256('\n'.join(store.feature_ids).encode()).hexdigest(),
        embedding_sha256=hashlib.sha256(embedding.tobytes()).hexdigest(),
        elapsed_seconds=time.time()-start, peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated(device))
        if device.type=='cuda' else 0, purpose='initialization only; full-field posterior updates unchanged',
        uncertainty='not approximated in sketch; original marginal variances used by full-field refinement',
        dropped_spots=0, implementation='graph-heat sparse sketch; NOT author SpaGFT')
    marker.write_text(json.dumps(manifest, indent=2))
    return manifest


def load_validated_embedding(directory, store, cache_manifest):
    directory = Path(directory)
    manifest = json.loads((directory/'multiscale_manifest.json').read_text())
    feature_hash = hashlib.sha256('\n'.join(store.feature_ids).encode()).hexdigest()
    if manifest['cache_source_signature'] != cache_manifest['source_signature'] or manifest['feature_order_sha256'] != feature_hash:
        raise ValueError("Sketch source/feature order mismatch")
    data = np.load(directory/'initialization_embedding.npy')
    if data.shape != (store.n_features, 4*manifest['selected_dimensions_per_band']) or not np.isfinite(data).all():
        raise ValueError("Invalid initialization embedding")
    if hashlib.sha256(data.tobytes()).hexdigest() != manifest['embedding_sha256']:
        raise ValueError("Sketch checksum mismatch")
    return data, manifest
