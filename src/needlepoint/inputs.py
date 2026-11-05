"""Portable spatial measurement inputs, with explicit coordinates and units."""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class SpatialData:
    """Measurements by locations, retaining all supplied observation rows.

    Coordinates are physical XY micrometres. ``measurement`` records whether
    values are counts or continuous measurements; conversion never normalizes.
    """
    counts: object
    coords: np.ndarray
    genes: object
    obs: pd.DataFrame | None = None
    var: pd.DataFrame | None = None
    measurement: str = "counts"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.counts = sparse.csr_matrix(self.counts, copy=True)
        self.counts.sum_duplicates()
        self.counts.sort_indices()
        self.coords = np.asarray(self.coords, dtype=np.float64).copy()
        self.genes = np.asarray(self.genes, dtype=str)
        n, p = self.counts.shape
        if n < 1 or p < 1:
            raise ValueError('At least one observation and feature are required')
        if self.coords.shape != (n, 2) or not np.isfinite(self.coords).all():
            raise ValueError('coords must be finite micrometre XY coordinates, shape (n_observations, 2)')
        if self.genes.shape != (p,) or len(set(self.genes)) != p or np.any(self.genes == ''):
            raise ValueError('Feature names must be nonempty, unique, and match matrix columns')
        if self.measurement not in ('counts', 'continuous'):
            raise ValueError('measurement must be counts or continuous')
        if self.counts.dtype.kind not in 'biuf':
            raise ValueError('Measurements must be real numeric values')
        if not np.isfinite(self.counts.data).all():
            raise ValueError('Measurements must be finite')
        if self.measurement == 'counts' and np.any(self.counts.data < 0):
            raise ValueError('Counts must be nonnegative')
        self.obs = pd.DataFrame(index=[str(i) for i in range(n)]) if self.obs is None else self.obs.copy()
        self.var = pd.DataFrame(index=self.genes) if self.var is None else self.var.copy()
        if len(self.obs) != n or not self.obs.index.is_unique:
            raise ValueError('obs must have one uniquely indexed row per observation')
        if len(self.var) != p:
            raise ValueError('var must have one row per feature')
        self.obs.index = self.obs.index.astype(str)
        if not self.obs.index.is_unique:
            raise ValueError('Observation IDs must remain unique after string conversion')
        self.var.index = pd.Index(self.genes)

    @property
    def shape(self):
        return self.counts.shape

    def to_anndata(self):
        import anndata as ad
        result = ad.AnnData(X=self.counts.copy(), obs=self.obs.copy(), var=self.var.copy())
        result.obsm['spatial'] = self.coords.copy()
        result.uns['needlepoint_input'] = dict(self.metadata, coordinate_units='micron',
                                               measurement=self.measurement)
        return result

    def write_h5ad(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_anndata().write_h5ad(path)
        return path

    def fingerprint(self):
        """Hash actual measurements, support, identifiers, and metadata."""
        h = hashlib.sha256()
        for a in (np.asarray(self.shape, dtype=np.int64), self.counts.indptr,
                  self.counts.indices, self.counts.data, self.coords):
            view = memoryview(np.ascontiguousarray(a)).cast('B')
            h.update(str(a.dtype).encode())
            for start in range(0, len(view), 8*1024**2):
                h.update(view[start:start+8*1024**2])
        for text in (self.measurement, self.obs.to_json(orient='split'), self.var.to_json(orient='split'),
                     json.dumps(self.metadata,sort_keys=True,
                                default=lambda v: v.tolist() if isinstance(v,np.ndarray) else v.item())):
            h.update(text.encode()); h.update(b'\0')
        return h.hexdigest()


def read_spatial(source, *, coords=None, genes=None, layer=None, coord_key='auto',
                 coordinate_scale=1.0, measurement=None, obs=None,
                 allow_measurement_override=False):
    """Read AnnData/H5AD or a dense/sparse matrix with explicit XY coordinates.

    ``coordinate_scale`` converts supplied coordinates to micrometres. No
    coordinate units are inferred, and no observation or feature is removed.
    ``.npz`` paths are SciPy sparse-matrix files and require coords and genes.
    Saved measurement semantics are preserved. Conflicting explicit semantics
    require ``allow_measurement_override=True``. Unlabelled matrices default
    to counts; ingestion never infers the likelihood from a platform name.
    """
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError('coordinate_scale must be finite and positive')
    if isinstance(source, SpatialData):
        if any(x is not None for x in (coords, genes, layer, obs)) or coordinate_scale != 1:
            raise ValueError('SpatialData already contains coordinates, identifiers and measurements')
        if measurement is None or measurement == source.measurement:
            return source
        if not allow_measurement_override:
            raise ValueError('Conflicting saved measurement; set allow_measurement_override=True explicitly')
        return SpatialData(source.counts, source.coords, source.genes, source.obs,
                           source.var, measurement, dict(source.metadata))
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.suffix == '.h5ad':
            import anndata as ad
            source = ad.read_h5ad(p)
        elif p.suffix == '.zarr':
            import anndata as ad
            source = ad.read_zarr(p)
        elif p.suffix == '.npz':
            source = sparse.load_npz(p)
        else:
            raise ValueError('Use H5AD, sparse NPZ, read_10x_mtx, or read_transcripts')
    if hasattr(source, 'obsm') and hasattr(source, 'var_names'):
        from .io.h5ad import resolve_h5ad_coordinates
        matrix = source.X if layer is None else source.layers[layer]
        if coords is None:
            coords, _ = resolve_h5ad_coordinates(source, coord_key)
        saved = source.uns.get('needlepoint_input', {})
        saved_measurement = saved.get('measurement')
        if (measurement is not None and saved_measurement is not None
                and measurement != saved_measurement and not allow_measurement_override):
            raise ValueError('Conflicting saved measurement; set allow_measurement_override=True explicitly')
        result = SpatialData(matrix, np.asarray(coords)*coordinate_scale,
                           source.var_names if genes is None else genes,
                           source.obs if obs is None else obs, source.var,
                           measurement if measurement is not None else saved_measurement or 'counts')
        result.metadata = {k:v for k,v in saved.items()
                           if k not in ('coordinate_units','measurement')}
        return result
    if coords is None or genes is None or layer is not None:
        raise ValueError('Matrix inputs require coords and genes; layer is an AnnData option')
    return SpatialData(source, np.asarray(coords)*coordinate_scale, genes, obs=obs,
                       measurement='counts' if measurement is None else measurement)


def _read_tsv(path):
    return pd.read_csv(path, sep='\t', header=None, dtype=str, keep_default_na=False)


def _aligned_positions(positions, barcodes, x, y, barcode):
    table = positions.copy() if isinstance(positions, pd.DataFrame) else (
        pd.read_parquet(positions) if Path(positions).suffix == '.parquet' else pd.read_csv(positions))
    table[barcode] = table[barcode].astype(str)
    if table[barcode].duplicated().any():
        raise ValueError('Duplicate position barcodes')
    table = table.set_index(barcode)
    missing = set(barcodes) - set(table.index)
    if missing:
        raise ValueError(f'Missing coordinates for {len(missing)} barcodes; support must be preserved')
    return table.loc[barcodes]


def read_10x_mtx(directory, *, positions, x='x', y='y', barcode='barcode',
                 coordinate_scale=1.0, feature_column=0):
    """Read 10x features × barcodes MTX and align positions by barcode.

    Positions must be a DataFrame or CSV with explicitly named columns. Supply
    feature-column 1 for gene symbols only when unique; IDs (column 0) are the
    default. Every barcode requires coordinates, including zero-count columns.
    """
    from scipy.io import mmread
    directory = Path(directory)
    def locate(name):
        for n in (name, name+'.gz'):
            if (directory/n).is_file(): return directory/n
        raise FileNotFoundError(directory/name)
    matrix_path = locate('matrix.mtx')
    with (gzip.open(matrix_path, 'rb') if matrix_path.suffix == '.gz' else matrix_path.open('rb')) as f:
        matrix = sparse.csr_matrix(mmread(f)).T.tocsr()
    features = _read_tsv(locate('features.tsv'))
    barcodes = _read_tsv(locate('barcodes.tsv'))[0].to_numpy()
    if matrix.shape != (len(barcodes), len(features)):
        raise ValueError('MTX dimensions disagree with features/barcodes')
    table = _aligned_positions(positions, barcodes, x, y, barcode)
    result = read_spatial(matrix, coords=table[[x,y]].to_numpy(), genes=features[feature_column],
                          obs=table, coordinate_scale=coordinate_scale)
    result.var['feature_id'] = features[0].to_numpy()
    if features.shape[1] > 1: result.var['feature_name'] = features[1].to_numpy()
    return result


def read_10x_h5(path, *, positions, x='x', y='y', barcode='barcode',
                 coordinate_scale=1.0, feature_key='id'):
    """Read the modern 10x /matrix CSC HDF5 layout; align all barcodes by ID.

    Supply physical positions (DataFrame/CSV/Parquet) and any unit conversion.
    This is a feature-barcode reader, not a general vendor-HDF5 reader.
    """
    import h5py
    def strings(dataset):
        return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in dataset[:]])
    with h5py.File(path, 'r') as handle:
        group = handle['matrix']
        matrix = sparse.csc_matrix((group['data'][:], group['indices'][:], group['indptr'][:]),
                                   shape=tuple(group['shape'][:])).T.tocsr()
        barcodes = strings(group['barcodes'])
        features = group['features']
        names = strings(features[feature_key])
        annotations = {key:strings(features[key]) for key in ('id', 'name', 'feature_type') if key in features}
    if matrix.shape != (len(barcodes), len(names)):
        raise ValueError('HDF5 matrix dimensions disagree with features/barcodes')
    table = _aligned_positions(positions, barcodes, x, y, barcode)
    result = read_spatial(matrix, coords=table[[x, y]].to_numpy(), genes=names, obs=table,
                          coordinate_scale=coordinate_scale)
    for key, values in annotations.items():
        result.var['feature_' + key] = values
    return result


def read_gem(source, *, bounds, pitch, genes, coordinate_scale, gene='geneID',
             count=None, x='x', y='y'):
    """Bin a Stereo-seq GEM count table on explicit full physical support.

    GEM coordinate units vary by acquisition; coordinate_scale is mandatory.
    Bounds/pitch are XYXY micrometres. Weighted entries are summed directly,
    without expanding counts into repeated molecules. Empty bins are retained.
    """
    from .bounds import bounds_xyxy
    from .io.xenium import grid_geometry
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError('coordinate_scale must be finite and positive')
    table = source.copy() if isinstance(source, pd.DataFrame) else pd.read_csv(source, sep='\t', comment='#')
    if count is None:
        candidates = [name for name in ('MIDCount', 'MIDCounts') if name in table]
        if len(candidates) != 1:
            raise ValueError('Specify the GEM count column explicitly; expected MIDCount or MIDCounts')
        count = candidates[0]
    values = pd.to_numeric(table[count], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError('GEM molecule counts must be finite nonnegative integers')
    names = np.asarray(genes, dtype=str)
    if names.ndim != 1 or not len(names) or len(set(names)) != len(names):
        raise ValueError('Supply a unique complete gene panel')
    bounds = bounds_xyxy(bounds)
    coords, (ny, nx) = grid_geometry(bounds, pitch)
    indices = pd.Index(names).get_indexer(table[gene].astype(str))
    tx, ty = table[x].to_numpy(dtype=float)*coordinate_scale, table[y].to_numpy(dtype=float)*coordinate_scale
    if not np.isfinite(tx).all() or not np.isfinite(ty).all():
        raise ValueError('GEM coordinates must be finite')
    x0, y0, x1, y1 = bounds
    keep = (indices >= 0) & (tx >= x0) & (tx < x1) & (ty >= y0) & (ty < y1)
    rows = (np.floor((ty[keep]-y0)/pitch).astype(np.int64)*nx
            + np.floor((tx[keep]-x0)/pitch).astype(np.int64))
    matrix = sparse.coo_matrix((values[keep], (rows, indices[keep])), shape=(ny*nx, len(names))).tocsr()
    result = SpatialData(matrix, coords, names)
    result.metadata['gem_input'] = dict(input_rows=len(table), retained_rows=int(keep.sum()),
        bounds_xyxy=list(bounds), pitch_um=float(pitch), coordinate_scale=float(coordinate_scale))
    return result


def read_transcripts(source, *, bounds, pitch, genes, x='x_location', y='y_location',
                     gene='feature_name', quality='qv', minimum_quality=20,
                     coordinate_scale=1.0):
    """Bin a transcript DataFrame/CSV/Parquet on an explicit full XY rectangle.

    Bounds and pitch are in micrometres. The supplied complete gene panel
    defines controls to exclude. Quality filtering and exclusion counts are
    recorded; every grid bin, including empty bins, is retained.
    """
    from .io.xenium import bin_transcripts
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError('coordinate_scale must be finite and positive')
    if quality is not None and not np.isfinite(minimum_quality):
        raise ValueError('minimum_quality must be finite')
    if isinstance(source, pd.DataFrame):
        table = source.copy()
    elif Path(source).suffix == '.parquet':
        table = pd.read_parquet(source)
    else:
        table = pd.read_csv(source)
    names = np.asarray(genes, dtype=str)
    if names.ndim != 1 or len(names) == 0 or len(set(names)) != len(names):
        raise ValueError('Supply a unique complete gene panel')
    accepted = np.ones(len(table), dtype=bool)
    if quality is not None:
        values = pd.to_numeric(table[quality], errors='raise').to_numpy()
        if not np.isfinite(values).all(): raise ValueError('Transcript quality must be finite')
        accepted &= values >= minimum_quality
    feature_names = table[gene].map(lambda value: value.decode('utf-8') if isinstance(value,bytes) else str(value))
    indices = pd.Index(names).get_indexer(feature_names)
    accepted &= indices >= 0
    tx = table.loc[accepted, x].to_numpy(dtype=float)*coordinate_scale
    ty = table.loc[accepted, y].to_numpy(dtype=float)*coordinate_scale
    matrix, coords = bin_transcripts(tx, ty, indices[accepted], bounds=bounds, pitch=pitch, n_features=len(names))
    result = SpatialData(matrix, coords, names)
    result.obs['bin_library_size'] = np.asarray(matrix.sum(axis=1)).ravel()
    # Acquisition support is supplied by the caller; a rectangle inferred from
    # observed molecules alone would lose legitimately empty boundary bins.
    result.metadata['transcript_filter'] = {'input_calls': len(table),
        'quality_and_panel_accepted': int(accepted.sum()), 'in_bounds_calls': int(matrix.sum())}
    return result
