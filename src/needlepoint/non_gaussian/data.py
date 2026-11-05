"""Event-preserving point patterns and physical-area quadrature, NumPy only."""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np


def fingerprint(*arrays, metadata=None):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(json.dumps(metadata or {}, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def _immutable(value, dtype=float):
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _bounds(bounds):
    from ..bounds import bounds_xyxy
    return bounds_xyxy(bounds)


def _inside(coords, bounds):
    x0, y0, x1, y1 = bounds
    return ((coords[:, 0] >= x0) & (coords[:, 0] <= x1)
            & (coords[:, 1] >= y0) & (coords[:, 1] <= y1))


@dataclass(frozen=True)
class PointPattern:
    """One feature's original events in an explicit acquired window.

    Repeated event coordinates are retained. Effort is known acquisition effort
    at events, not observed library totals. Area units are square micrometers.
    """
    coordinates: object
    bounds: tuple
    feature_name: str = 'feature'
    effort: object = None
    observed_area: float | None = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        xy = _immutable(self.coordinates)
        if xy.size == 0:
            xy = _immutable(np.empty((0, 2)))
        bounds = _bounds(self.bounds)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError('Event coordinates must be finite (n_events, 2)')
        if not _inside(xy, bounds).all():
            raise ValueError('Events outside the acquired window must be resolved explicitly')
        effort = np.ones(len(xy)) if self.effort is None else np.broadcast_to(self.effort, (len(xy),))
        if not np.isfinite(effort).all() or (effort <= 0).any():
            raise ValueError('Every recorded event requires finite positive effort')
        rectangle_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        area = rectangle_area if self.observed_area is None else float(self.observed_area)
        if not 0 < area <= rectangle_area * (1 + 1e-10):
            raise ValueError('observed_area must be positive and within the bounding rectangle')
        object.__setattr__(self, 'coordinates', xy)
        object.__setattr__(self, 'bounds', bounds)
        object.__setattr__(self, 'effort', _immutable(effort))
        object.__setattr__(self, 'observed_area', area)

    def fingerprint(self):
        return fingerprint(self.coordinates, self.effort, metadata=dict(
            bounds=self.bounds, observed_area=self.observed_area,
            feature_name=self.feature_name, metadata=self.metadata))


@dataclass(frozen=True)
class Quadrature:
    """Explicit integration points, physical areas, effort, and optional grid mask."""
    coordinates: object
    areas: object
    bounds: tuple
    effort: object = 1.0
    grid_shape: tuple | None = None
    acquired_mask: object = None

    def __post_init__(self):
        xy = _immutable(self.coordinates)
        bounds = _bounds(self.bounds)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0 or not np.isfinite(xy).all():
            raise ValueError('Quadrature coordinates must be finite (n_points, 2)')
        if not _inside(xy, bounds).all():
            raise ValueError('Quadrature points lie outside the acquired window')
        areas = np.broadcast_to(self.areas, (len(xy),))
        effort = np.broadcast_to(self.effort, (len(xy),))
        if not np.isfinite(areas).all() or (areas < 0).any() or areas.sum() <= 0:
            raise ValueError('Quadrature areas must be nonnegative with positive total area')
        if not np.isfinite(effort).all() or (effort < 0).any():
            raise ValueError('Quadrature effort must be finite and nonnegative')
        if self.grid_shape is not None:
            shape = tuple(int(v) for v in self.grid_shape)
            if len(shape) != 2 or min(shape) <= 0 or np.prod(shape) != len(xy):
                raise ValueError('grid_shape must match all quadrature rows, including empty cells')
            mask = np.ones(shape, bool) if self.acquired_mask is None else np.asarray(self.acquired_mask)
            if mask.dtype != bool or mask.shape != shape:
                raise ValueError('acquired_mask must be a boolean grid matching grid_shape')
            if (areas.reshape(shape)[~mask] != 0).any():
                raise ValueError('Unacquired cells must have zero integration area')
            object.__setattr__(self, 'grid_shape', shape)
            object.__setattr__(self, 'acquired_mask', _immutable(mask, bool))
        elif self.acquired_mask is not None:
            raise ValueError('acquired_mask requires grid_shape')
        object.__setattr__(self, 'coordinates', xy)
        object.__setattr__(self, 'bounds', bounds)
        object.__setattr__(self, 'areas', _immutable(areas))
        object.__setattr__(self, 'effort', _immutable(effort))

    def validate_events(self, events):
        if tuple(events.bounds) != self.bounds or not np.isclose(
                self.areas.sum(), events.observed_area, rtol=1e-8, atol=0):
            raise ValueError('Event window and quadrature area/support disagree')
        if self.grid_shape is not None and len(events.coordinates):
            x0, y0, x1, y1 = self.bounds
            ny, nx = self.grid_shape
            ix = np.minimum(((events.coordinates[:, 0] - x0) / (x1 - x0) * nx).astype(int), nx - 1)
            iy = np.minimum(((events.coordinates[:, 1] - y0) / (y1 - y0) * ny).astype(int), ny - 1)
            if not self.acquired_mask[iy, ix].all():
                raise ValueError('Events occur in explicitly unacquired cells')

    def fingerprint(self):
        mask = np.empty(0, bool) if self.acquired_mask is None else self.acquired_mask
        return fingerprint(self.coordinates, self.areas, self.effort, mask,
                           metadata=dict(bounds=self.bounds, grid_shape=self.grid_shape))


def regular_grid_quadrature(bounds, shape, *, acquired_mask=None, effort=1.0):
    """Midpoint rule on (ny, nx) cells; keep acquired empty cells and masked rows."""
    bounds = _bounds(bounds)
    ny, nx = tuple(shape)
    if int(ny) != ny or int(nx) != nx or min(ny, nx) <= 0:
        raise ValueError('shape must contain two positive integers')
    ny, nx = int(ny), int(nx)
    x0, y0, x1, y1 = bounds
    x = x0 + (np.arange(nx) + .5) * (x1 - x0) / nx
    y = y0 + (np.arange(ny) + .5) * (y1 - y0) / ny
    xx, yy = np.meshgrid(x, y)
    mask = np.ones((ny, nx), bool) if acquired_mask is None else np.asarray(acquired_mask)
    if mask.dtype != bool or mask.shape != (ny, nx):
        raise ValueError('acquired_mask must be boolean with shape (ny, nx)')
    areas = mask.ravel() * ((x1 - x0) * (y1 - y0) / (nx * ny))
    return Quadrature(np.column_stack((xx.ravel(), yy.ravel())), areas, bounds,
                      effort=np.broadcast_to(effort, (ny, nx)).ravel(),
                      grid_shape=(ny, nx), acquired_mask=mask)


def read_point_pattern(source, *, bounds, feature, x='x_location', y='y_location',
                       gene='feature_name', coordinate_scale=1.0, quality=None,
                       minimum_quality=None, observed_area=None):
    """Read original events from DataFrame/CSV/Parquet, without spatial binning.

    Bounds and observed_area are already in physical units. Quality filtering
    is opt-in and recorded. No observed events or library totals define effort.
    """
    import pandas as pd
    if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
        raise ValueError('coordinate_scale must be finite and positive')
    if isinstance(source, pd.DataFrame):
        table = source.copy()
    elif Path(source).suffix == '.parquet':
        table = pd.read_parquet(source)
    else:
        table = pd.read_csv(source)
    names = table[gene].map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
    selected = names == str(feature)
    before_quality = int(selected.sum())
    if quality is not None:
        if minimum_quality is None or not np.isfinite(minimum_quality):
            raise ValueError('Quality filtering requires a finite minimum_quality')
        values = pd.to_numeric(table[quality], errors='raise').to_numpy()
        if not np.isfinite(values).all():
            raise ValueError('Event quality must be finite')
        selected &= values >= minimum_quality
    xy = table.loc[selected, [x, y]].to_numpy(dtype=float) * coordinate_scale
    return PointPattern(xy, bounds, str(feature), observed_area=observed_area,
        metadata=dict(input_rows=len(table), retained_events=len(xy),
            other_feature_rows=int(len(table) - before_quality),
            quality_exclusions=int(before_quality - len(xy)),
            coordinate_scale=float(coordinate_scale), quality_column=quality,
            minimum_quality=minimum_quality, coordinate_units='micrometers'))
