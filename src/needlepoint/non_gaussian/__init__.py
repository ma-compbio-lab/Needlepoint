"""Experimental coupled Gaussian VI for Tobit, Poisson and Cox-process GPs.

Imports stay lazy so that data ingestion does not require Torch/GPyTorch.
"""
from importlib import import_module

_EXPORTS = {
    'export_coupled_maps': ('exports', 'export_coupled_maps'),
    'fit_spatial_vi': ('interface', 'fit_spatial_vi'),
    'CoupledVISettings': ('vi', 'CoupledVISettings'),
    'CoupledVIResult': ('vi', 'CoupledVIResult'),
    'VIConvergenceError': ('vi', 'VIConvergenceError'),
    'fit_tobit_gp': ('vi', 'fit_tobit_gp'),
    'fit_poisson_gp': ('vi', 'fit_poisson_gp'),
    'fit_lgcp': ('vi', 'fit_lgcp'),
    'fit_coupled_vi': ('vi', 'fit_coupled_vi'),
    'fit_kernel_parameters': ('vi', 'fit_kernel_parameters'),
    'DensePrior': ('operators', 'DensePrior'),
    'DenseEvaluation': ('operators', 'DenseEvaluation'),
    'StructuredSpatialPrior': ('operators', 'StructuredSpatialPrior'),
    'build_spatial_prior': ('operators', 'build_spatial_prior'),
    'PointPattern': ('data', 'PointPattern'),
    'Quadrature': ('data', 'Quadrature'),
    'read_point_pattern': ('data', 'read_point_pattern'),
    'regular_grid_quadrature': ('data', 'regular_grid_quadrature'),
    'TobitLikelihood': ('likelihoods', 'TobitLikelihood'),
    'PoissonLikelihood': ('likelihoods', 'PoissonLikelihood'),
    'LGCPIntegral': ('likelihoods', 'LGCPIntegral'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(f'{__name__}.{module}'), symbol)
    globals()[name] = value
    return value
