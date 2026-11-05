"""Shared GP export policy; storage precision never selects training precision."""
from dataclasses import dataclass
import argparse
import numpy as np


# Types are explicit: "covariance" contains "variance", but cross-covariances
# are signed.  This registry also lets the separate conditioning audit share
# the export contract without importing a model backend.
GP_MAP_FIELD_TYPES = {
    'posterior_mean': 'mean',
    'posterior_variance': 'variance',
    'posterior_predictive_variance': 'variance',
    'global_posterior_mean': 'mean',
    'local_posterior_mean': 'mean',
    'global_posterior_variance': 'variance',
    'local_posterior_variance': 'variance',
    'global_local_posterior_covariance': 'signed_covariance',
    'latent_log_rate_mean': 'mean',
    'latent_log_rate_variance': 'variance',
}


@dataclass
class GPStorageSettings:
    output_dtype: str = "float16"
    retain_predictive_variance: bool = False
    retain_component_maps: bool = False
    retain_latent_maps: bool = False

    def __post_init__(self):
        if self.output_dtype not in ("float16", "float32"):
            raise ValueError("output_dtype must be float16 or float32")


@dataclass(frozen=True)
class VIStorageSettings:
    """Compact coupled-VI exports; numerical fitting and state stay unchanged."""
    output_dtype: str = 'float32'
    field_scale: str = 'latent_gp'
    retain_component_maps: bool = False
    retain_numerical_maps: bool = False
    retain_observation_maps: bool = False
    retain_state: bool = True

    def __post_init__(self):
        if self.output_dtype not in ('float16', 'float32', 'float64'):
            raise ValueError('output_dtype must be float16, float32 or float64')
        if self.field_scale not in ('latent_gp', 'latent_predictor', 'rate', 'observed'):
            raise ValueError('Unknown VI field_scale')


def encode_vi_maps(settings, maps):
    """Keep one selected moment pair; component/diagnostic maps are opt-in."""
    pairs = {'latent_gp': ('latent_gp_mean', 'latent_gp_variance'),
             'latent_predictor': ('latent_predictor_mean', 'latent_gp_variance'),
             'rate': ('rate_mean', 'rate_variance'),
             'observed': ('observed_mean', 'observed_variance')}
    required = set(pairs[settings.field_scale])
    if not required <= maps.keys():
        raise ValueError(f'{settings.field_scale} moments are unavailable for this likelihood')
    keep = required | {'coordinates_um'}
    if settings.retain_component_maps:
        keep.update(('global_mean', 'local_mean', 'global_variance', 'local_variance',
                     'global_local_covariance'))
    if settings.retain_numerical_maps:
        keep.update(('variance_mcse', 'variance_lower', 'variance_upper', 'analytic_conditional_variance'))
    if settings.retain_observation_maps:
        keep.update(('observed_mean', 'observed_variance', 'censoring_probability'))
    result = {}
    for name in sorted(keep & maps.keys()):
        value = np.asarray(maps[name])
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite VI export {name}')
        if ('variance' in name and 'covariance' not in name or name == 'censoring_probability') and (value < 0).any():
            raise ValueError(f'Negative VI export {name}')
        with np.errstate(over='ignore', under='ignore'):
            cast = value.astype('float64' if name == 'coordinates_um' else settings.output_dtype)
        if not np.isfinite(cast).all():
            raise ValueError(f'{name} overflows {settings.output_dtype}; choose a wider output_dtype')
        result[name] = cast
    return result


def add_storage_arguments(parser):
    parser.add_argument('--output-dtype', choices=['float16', 'float32'], default='float16',
                        help='Saved GP map precision; training precision is independent.')
    for name in ('predictive-variance', 'component-maps', 'latent-maps'):
        parser.add_argument('--retain-' + name, action=argparse.BooleanOptionalAction, default=False)


def storage_kwargs(args):
    return {key: getattr(args, key, default) for key, default in vars(GPStorageSettings()).items()}


def storage_manifest(settings):
    return dict(storage_kwargs(settings), policy='posterior-moments-with-explicit-extras-v1',
                quantization='IEEE round-to-nearest; small values may underflow; finite overflow rejected',
                required_maps=['posterior_mean', 'posterior_variance'],
                archive_layout='npz-stored-memory-mappable')


def encode_gp_maps(settings, arrays):
    """Select exports and check them after casting; never clip overflowing maps."""
    dtype = np.dtype(settings.output_dtype)
    if dtype not in (np.dtype('float16'), np.dtype('float32')):
        raise ValueError('output_dtype must be float16 or float32')
    required = {'posterior_mean', 'posterior_variance'}
    if not required <= arrays.keys():
        raise ValueError('Both posterior moments are required')
    keep = set(required)
    if settings.retain_predictive_variance:
        keep.add('posterior_predictive_variance')
    if settings.retain_component_maps:
        keep.update(('global_posterior_mean', 'global_posterior_variance',
                     'local_posterior_mean', 'local_posterior_variance',
                     'global_local_posterior_covariance'))
    if settings.retain_latent_maps:
        keep.update(('latent_log_rate_mean', 'latent_log_rate_variance'))
    result = {}
    for name, values in arrays.items():
        if name not in keep:
            continue
        values = np.asarray(values)
        if (not np.isfinite(values).all()
                or (GP_MAP_FIELD_TYPES[name] == 'variance' and (values < 0).any())):
            raise ValueError(f'Invalid GP export {name}')
        with np.errstate(over='ignore', under='ignore'):
            cast = values.astype(dtype, copy=False)
        if not np.isfinite(cast).all():
            raise ValueError(f'{name} overflows {dtype}; rerun with --output-dtype float32')
        result[name] = cast
    return result


def map_array_count(options, *, joint=True):
    return (2 + bool(options.get('retain_predictive_variance', False))
            + (4 if joint and options.get('retain_component_maps', False) else 0)
            + (2 if joint and options.get('retain_latent_maps', False) else 0))


def estimate_workflow_storage(n_spots, n_features, options):
    """Uncompressed phase peaks; downstream full-map caches are opt-in."""
    unit = int(n_spots) * int(n_features)
    gp = unit * np.dtype(options.get('output_dtype', 'float16')).itemsize * map_array_count(options)
    shards = gp if options.get('retain_shard_maps', False) else 0
    # Prototype caches survive until the LBM stage only when explicitly retained.
    downstream = (unit * 4 * (4 if options.get('retain_caches', False) else 2)
                  if options.get('materialize_downstream_maps', False) else 0)
    return dict(gp_map_bytes=gp, merge_peak_bytes=2*gp,
                downstream_peak_bytes=gp+shards+downstream,
                reserve_bytes=100*1024**3,
                estimated_required_free_bytes=max(2*gp, gp+shards+downstream)+100*1024**3)


def cleanup_workflow_stage(root, merged_gp, stage, options):
    """Prune only run-owned intermediates, after their consuming stage succeeded.

    Imported/symlinked shards and external caches are never removed. Failed
    stages never call this function. Essential audit metadata stays in place.
    """
    import shutil
    from pathlib import Path
    from needlepoint.artifacts import atomic_write_json, sha256_file, audit_gp_artifact
    root = Path(root).resolve()
    removed = []
    if stage == 'fit' and not options.get('retain_shard_maps', False):
        merged_gp = Path(merged_gp)
        audit = audit_gp_artifact(merged_gp)
        if audit.status != 'production':
            raise ValueError('Refusing shard cleanup before a production GP audit')
        import pandas as pd
        chunks = pd.read_csv(merged_gp/'vski_gp_map_chunks.csv')
        # Validate all merged copies before removing any source arrays.
        for row in chunks.to_dict('records'):
            path = merged_gp/row['file']
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('Shard cleanup requires independent run-owned merged copies')
            if sha256_file(path) != row['file_sha256']:
                raise ValueError('Merged GP checksum mismatch; keeping all shard maps')
        sources = set()
        for row in chunks.to_dict('records'):
            shard = Path(row['source_shard_dir'])
            if not shard.resolve().is_relative_to(root/'gp/shards') or shard.is_symlink():
                continue
            data = shard/'fitted_maps' if (shard/'fitted_maps').is_dir() else shard
            index = pd.read_csv(data/'vski_gp_map_chunks.csv')
            match = index[index.chunk_id == row['source_chunk_id']]
            if len(match) != 1:
                raise ValueError('Cannot identify source chunk; keeping shard maps')
            item = match.iloc[0]
            path = data/item['file']
            if path.is_symlink() or not path.resolve().is_relative_to(root/'gp/shards'):
                raise ValueError('Refusing external shard cleanup')
            if path.exists() and sha256_file(path) != item['file_sha256']:
                raise ValueError('Source GP checksum mismatch; keeping shard maps')
            sources.add(path)
        for path in sorted(sources):
            if path.exists():
                path.unlink()
                removed.append(str(path.relative_to(root)))
    if not options.get('retain_caches', False):
        names = {'screen': ['caches/scoring_null'],
                 'fit': ['caches/vnngp_geometry', 'length_scale/checkpoints', 'length_scale/merged/checkpoints'],
                 'prototypes': ['caches/prototype_maps'],
                 'lbm': ['caches/lbm_maps', 'caches/graph_knn16.npz', 'caches/graph_knn16.npz.json']}.get(stage, [])
        for name in names:
            path = root/name
            if not path.exists() or path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            if path.is_dir():
                # Do not follow imported cache trees.
                if any(p.is_symlink() for p in path.rglob('*')):
                    continue
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(name)
    (root/'provenance').mkdir(parents=True, exist_ok=True)
    atomic_write_json(root/'provenance'/f'{stage}_storage_cleanup.json',
                      dict(stage=stage, removed=removed, retention=options))


def cleanup_public_workflow_cache(root, *, retain_caches=False):
    """Prune the public Workflow's owned geometry cache before final receipts.

    This helper is called only after the GP process and production audit pass.
    Inputs, maps and frozen source are never candidates for deletion.
    """
    from pathlib import Path
    import shutil
    from .artifacts import atomic_write_json
    root = Path(root).resolve()
    cache = root/'caches/geometry'
    receipt = dict(retain_caches=retain_caches, removed=[], removed_bytes=0)
    if (not retain_caches and cache.is_dir() and not cache.is_symlink()
            and cache.resolve().is_relative_to(root)
            and not any(p.is_symlink() for p in cache.rglob('*'))):
        receipt['removed_bytes'] = sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
        shutil.rmtree(cache)
        receipt['removed'] = ['caches/geometry']
    (root/'provenance').mkdir(exist_ok=True)
    path = root/'provenance/gp_storage_cleanup.json'
    atomic_write_json(path, receipt)
    return dict(receipt, receipt=str(path))
