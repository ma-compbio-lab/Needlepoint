"""Experimental observation-level factor models with Global+Local spatial GPs.

The spatial_factors namespace, including settings, is safe to import without
Torch. Model and likelihood implementations require Needlepoint's ``gp`` extra.
"""
from importlib import import_module

_EXPORTS = {
    "SpatialFactorSettings": ("settings", "SpatialFactorSettings"),
    "SpatialFactorResult": ("fit", "SpatialFactorResult"),
    "SpatialFactorPrediction": ("model", "SpatialFactorPrediction"),
    "GlobalLocalFactorModel": ("model", "GlobalLocalFactorModel"),
    "fit_spatial_factors": ("fit", "fit_spatial_factors"),
    "load_spatial_factors": ("fit", "load_spatial_factors"),
    "FactorObservation": ("likelihoods", "FactorObservation"),
    "PoissonObservation": ("likelihoods", "PoissonObservation"),
    "NegativeBinomialObservation": ("likelihoods", "NegativeBinomialObservation"),
    "GaussianObservation": ("likelihoods", "GaussianObservation"),
    "TobitObservation": ("likelihoods", "TobitObservation"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), symbol)
    globals()[name] = value
    return value
