"""Fine-grid counts from decoded Xenium calls; no cell segmentation dependency."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def current_panel_genes(panel):
    """Include base and add-on genes; exclude controls/deprecated codewords."""
    result = {}
    for target in panel['payload']['targets']:
        if target['type']['descriptor'] != 'gene':
            continue
        if target['source']['category'] not in ('base', 'current'):
            raise ValueError('Unknown gene-panel source category')
        data = target['type']['data']
        if data['name'] in result:
            raise ValueError('Duplicate gene name in panel')
        result[data['name']] = data.get('id', '')
    return result


def grid_geometry(bounds, pitch):
    from needlepoint.bounds import bounds_xyxy
    bounds = np.asarray(bounds_xyxy(bounds), dtype=float)
    if bounds.shape != (4,) or not np.isfinite(bounds).all() or not np.isfinite(pitch) or pitch <= 0:
        raise ValueError("Finite XYXY bounds and positive pitch required")
    x0, y0, x1, y1 = bounds
    dimensions = np.array([x1-x0, y1-y0])/pitch
    if np.any(dimensions <= 0) or not np.allclose(dimensions, np.rint(dimensions), atol=1e-8, rtol=0):
        raise ValueError("ROI sides must be positive integer multiples of pitch")
    nx, ny = np.rint(dimensions).astype(int)
    yy, xx = np.indices((ny, nx))
    coords = np.column_stack((x0+(xx.ravel()+.5)*pitch, y0+(yy.ravel()+.5)*pitch))
    return coords, (int(ny), int(nx))


def bin_transcripts(x, y, feature_indices, *, bounds, pitch, n_features):
    """Half-open XY bounds, row-major bins, all zero rows retained explicitly."""
    from needlepoint.bounds import bounds_xyxy
    bounds = bounds_xyxy(bounds)
    coords, (ny, nx) = grid_geometry(bounds, pitch)
    x, y = np.asarray(x), np.asarray(y)
    genes = np.asarray(feature_indices)
    if x.shape != y.shape or genes.shape != x.shape or x.ndim != 1:
        raise ValueError("Molecule arrays must be aligned vectors")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite molecule coordinates")
    if not np.issubdtype(genes.dtype, np.integer) or np.any((genes < 0) | (genes >= n_features)):
        raise ValueError("Unknown feature indices")
    x0, y0, x1, y1 = bounds
    keep = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    rows = (np.floor((y[keep]-y0)/pitch).astype(np.int64)*nx
            + np.floor((x[keep]-x0)/pitch).astype(np.int64))
    matrix = sparse.coo_matrix((np.ones(keep.sum(), dtype=np.int32), (rows, genes[keep])),
        shape=(ny*nx, n_features)).tocsr()
    if int(matrix.sum()) != int(keep.sum()):
        raise AssertionError("Molecule conservation failed")
    return matrix, coords


def coarsen_grid(matrix, shape, factor):
    """Exact sum aggregation of complete nested support, including empty bins."""
    ny, nx = shape
    if factor < 1 or int(factor) != factor or ny % factor or nx % factor or matrix.shape[0] != nx*ny:
        raise ValueError("Coarsening requires complete divisible grid")
    yy, xx = np.indices((ny, nx))
    rows = ((yy//factor)*(nx//factor)+xx//factor).ravel()
    aggregate = sparse.csr_matrix((np.ones(nx*ny, dtype=np.int32), (rows, np.arange(nx*ny))),
        shape=((ny//factor)*(nx//factor), nx*ny))
    result = (aggregate @ matrix).tocsr()
    if int(result.sum()) != int(matrix.sum()):
        raise AssertionError("Coarse molecule conservation failed")
    return result
