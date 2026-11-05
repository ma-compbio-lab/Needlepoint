"""One physical acquisition-window convention for bins and point patterns."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpatialBounds:
    """XYXY bounds in micrometres: xmin, ymin, xmax, ymax.

    Named fields avoid ambiguous four-tuples. Historical LGCP XXYY tuples
    require the explicit ``from_xxyy`` conversion; their order is never guessed.
    """
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self):
        values = np.asarray(self.as_tuple(), dtype=float)
        if not np.isfinite(values).all() or self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError('Bounds must be finite positive-area (xmin, ymin, xmax, ymax) in micrometres')

    def as_tuple(self):
        return (float(self.xmin), float(self.ymin), float(self.xmax), float(self.ymax))

    @property
    def area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    @classmethod
    def from_xxyy(cls, values):
        xmin, xmax, ymin, ymax = values
        return cls(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)


def bounds_xyxy(value):
    if not isinstance(value, SpatialBounds):
        try:
            value = SpatialBounds(*map(float, value))
        except (TypeError, ValueError) as exc:
            raise ValueError('Supply SpatialBounds or (xmin, ymin, xmax, ymax)') from exc
    return value.as_tuple()
