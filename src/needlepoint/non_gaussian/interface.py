"""Fit explicit likelihoods to the common, format-independent SpatialData."""
import numpy as np
from ..inputs import read_spatial
from .vi import fit_poisson_gp, fit_tobit_gp


def fit_spatial_vi(data, prior, *, observation_model, features=None, settings=None,
                   exposure=1.0, exposure_semantics='unit', is_censored=None,
                   threshold=None, noise_sd=None, measurement_scale=None,
                   read_options=None):
    """Yield (feature_name, fit) in panel order, fitting one feature at a time.

    All common matrix readers can feed this iterator. Poisson consumes raw
    counts; Tobit requires continuous measurements and explicit censor flags.
    Feature-specific censor flags/thresholds must match the full input matrix
    (or use a scalar/one value per location). No count normalization is applied.
    LGCP uses read_point_pattern + fit_lgcp to preserve original events.
    """
    data = read_spatial(data, **(read_options or {}))
    if observation_model not in ('poisson', 'tobit'):
        raise ValueError('Use poisson or tobit; original LGCP events use fit_lgcp')
    expected = 'counts' if observation_model == 'poisson' else 'continuous'
    if data.measurement != expected:
        raise ValueError(f'{observation_model} requires measurement={expected!r}')
    names = data.genes.tolist() if features is None else list(features)
    if not names or len(set(names)) != len(names) or not set(names) <= set(data.genes):
        raise ValueError('Feature panel must be nonempty, unique and fully present')
    if observation_model == 'tobit' and any(v is None for v in
            (is_censored, threshold, noise_sd, measurement_scale)):
        raise ValueError('Tobit requires is_censored, threshold, noise_sd and measurement_scale')
    def column(value, position):
        array = np.asarray(value)
        if array.ndim == 2:
            if array.shape != data.shape:
                raise ValueError('Feature-specific observation parameters must match the full input shape')
            return array[:, position]
        return value
    input_sha256 = data.fingerprint()
    for name in names:
        position = data.genes.tolist().index(name)
        values = data.counts[:, position].toarray().ravel()
        if observation_model == 'poisson':
            fitted = fit_poisson_gp(prior, values, coordinates=data.coords,
                exposure=column(exposure, position), exposure_semantics=exposure_semantics,
                settings=settings)
        else:
            fitted = fit_tobit_gp(prior, values, coordinates=data.coords,
                is_censored=column(is_censored, position), threshold=column(threshold, position),
                noise_sd=noise_sd, measurement_scale=measurement_scale, settings=settings)
        fitted.data_metadata.update(feature_name=name, spatial_input_sha256=input_sha256)
        yield name, fitted
