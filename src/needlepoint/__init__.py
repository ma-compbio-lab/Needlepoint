"""Shared spatial contrast, GP, prototype, and LBM implementation.

The package deliberately avoids eager model imports.  This keeps lightweight
commands such as provenance inspection usable without CUDA/GPyTorch and makes
optional dependency boundaries explicit.
"""

__version__ = "0.2.0"

_PUBLIC = {
    "SpatialFactorSettings": ("spatial_factors", "SpatialFactorSettings"),
    "SpatialFactorResult": ("spatial_factors", "SpatialFactorResult"),
    "fit_spatial_factors": ("spatial_factors", "fit_spatial_factors"),
    "load_spatial_factors": ("spatial_factors", "load_spatial_factors"),
    "SpatialBounds": ("bounds", "SpatialBounds"),
    "VIStorageSettings": ("storage", "VIStorageSettings"),
    "export_coupled_maps": ("non_gaussian.exports", "export_coupled_maps"),
    "fit_spatial_vi": ("non_gaussian.interface", "fit_spatial_vi"),
    "SpatialData": ("inputs", "SpatialData"),
    "read_spatial": ("inputs", "read_spatial"),
    "read_10x_mtx": ("inputs", "read_10x_mtx"),
    "read_10x_h5": ("inputs", "read_10x_h5"),
    "read_gem": ("inputs", "read_gem"),
    "read_transcripts": ("inputs", "read_transcripts"),
    "Resources": ("resources", "Resources"),
    "WorkflowConfig": ("workflow", "WorkflowConfig"),
    "Workflow": ("workflow", "Workflow"),
    "WorkflowResult": ("workflow", "WorkflowResult"),
    "SpatialFeatureDataset": ("data", "SpatialFeatureDataset"),
    "normalize_counts": ("api", "normalize_counts"),
    "default_workflow": ("api", "default_workflow"),
    "JointGlobalLocalGPSettings": ("gpytorch_joint_global_local", "JointGlobalLocalGPSettings"),
    "fit_gp": ("gpytorch_joint_global_local", "fit_joint_global_local_gp_spatial_maps"),
    "CoupledVISettings": ("non_gaussian", "CoupledVISettings"),
    "fit_tobit_gp": ("non_gaussian", "fit_tobit_gp"),
    "fit_poisson_gp": ("non_gaussian", "fit_poisson_gp"),
    "fit_lgcp": ("non_gaussian", "fit_lgcp"),
    "SpatialPrototypeSettings": ("spatial_prototypes", "SpatialPrototypeSettings"),
    "fit_prototypes": ("spatial_prototypes", "fit_spatial_prototypes"),
    "merge_prototypes": ("spatial_prototypes", "merge_spatial_prototype_directory"),
    "VSKILBM": ("vski_lbm", "VSKILBM"),
    "VSKIMapStore": ("vski_lbm", "VSKIMapStore"),
    "require_production_gp_artifacts": ("artifacts", "require_production_gp_artifacts"),
}
__all__ = ["__version__", *_PUBLIC]


def __getattr__(name):
    from importlib import import_module
    if name not in _PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = _PUBLIC[name]
    value = getattr(import_module(f"{__name__}.{module}"), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
