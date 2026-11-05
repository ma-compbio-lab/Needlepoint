"""Strict input audits for the opt-in shared/private prototype-LBM experiment."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .artifacts import require_production_gp_artifacts, map_file_fingerprints, sha256_file
from .multidataset_lbm import ExistingCachedVSKIMapStore


def audited_existing_cache(vski_dir, metadata_path, cache_dir):
    """Reject non-production, stale-source, misordered or partial cached maps."""
    vski_dir, metadata_path, cache_dir = map(Path,(vski_dir,metadata_path,cache_dir))
    store=ExistingCachedVSKIMapStore(vski_dir,metadata_path,cache_dir)
    audits=require_production_gp_artifacts([vski_dir])
    manifest_path=cache_dir/'lbm_map_cache_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    paths=[Path(x['path']) for x in manifest.get('source_map_files',[])]
    if not paths:
        raise ValueError('Cache has no source map fingerprints')
    if map_file_fingerprints(paths)!=manifest['source_map_files']:
        raise ValueError('Cached posterior source files have changed')
    positions=store.metadata.feature_position.to_numpy(dtype=int)
    if not np.array_equal(positions,np.arange(store.n_features)) or store.metadata.gene.duplicated().any():
        raise ValueError('Feature axis is not unique and exhaustive')
    if len(store.spots)!=manifest['n_spots'] or store.spots.spot_index.duplicated().any():
        raise ValueError('Spot axis is not unique and exhaustive')
    if 'map_row' in store.spots and not np.array_equal(store.spots.map_row,np.arange(store.n_spots)):
        raise ValueError('Map rows do not match cache rows')
    if not manifest.get('uses_posterior_variance'):
        raise ValueError('GP uncertainty is required')
    for start in range(0,store.n_spots,8192):
        y=np.asarray(store._mean[start:start+8192]); v=np.asarray(store._variance[start:start+8192])
        if not np.isfinite(y).all() or not np.isfinite(v).all() or (v<0).any():
            raise ValueError('Invalid GP moments in cache')
    audit={'gp_audits':[item.to_dict() for item in audits],'cache_manifest_sha256':sha256_file(manifest_path),
           'metadata_sha256':sha256_file(metadata_path),'n_spots':store.n_spots,
           'n_features':store.n_features,'all_entries_validated':True,
           'source_fingerprints_current':True,'cache_manifest':manifest}
    return store,audit


def aligned_prototype_memberships(directory, metadata):
    """Align memberships by explicit feature IDs, never by incidental order."""
    directory=Path(directory)
    manifest_path=directory/'prototype_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    table=pd.read_csv(directory/'prototype_assignments.csv')
    gamma=np.load(directory/'prototype_responsibilities.npy')
    if table.gene.duplicated().any() or metadata.gene.duplicated().any():
        raise ValueError('Prototype feature IDs must be unique')
    if gamma.shape[0]!=len(table) or set(table.gene)!=set(metadata.gene):
        raise ValueError('Prototype coverage differs from GP panel')
    ordered_hash=hashlib.sha256('\n'.join(table.gene.astype(str)).encode()).hexdigest()
    if ordered_hash != manifest['input']['feature_order_sha256']:
        raise ValueError('Prototype row labels do not match their recorded feature hash')
    lookup=pd.Series(np.arange(len(table)),index=table.gene.astype(str))
    positions=lookup.loc[metadata.gene.astype(str)].to_numpy()
    gamma=np.asarray(gamma[positions],dtype=np.float64)
    if not np.isfinite(gamma).all() or (gamma<0).any() or not np.allclose(gamma.sum(1),1,atol=1e-5):
        raise ValueError('Invalid prototype memberships')
    for column in ['map_column','map_chunk_file']:
        if column in table and column in metadata:
            if not np.array_equal(table.iloc[positions][column].astype(str),metadata[column].astype(str)):
                raise ValueError('Prototype and GP feature-source identities differ')
    return gamma,{'manifest_sha256':sha256_file(manifest_path),
                  'memberships_sha256':sha256_file(directory/'prototype_responsibilities.npy'),
                  'n_prototypes':gamma.shape[1],'n_features':gamma.shape[0],
                  'feature_ids_verified':True,'manifest':manifest}


def strict_gp_reliability(metadata):
    """Signal/(signal+noise); no neutral fallback for this controlled experiment."""
    values=metadata[['joint_global_outputscale','joint_local_outputscale','joint_noise_variance']].to_numpy(float)
    if not np.isfinite(values).all() or (values<0).any() or (values.sum(1)<=0).any():
        raise ValueError('Missing or invalid signal/noise GP metadata')
    return np.clip(values[:,:2].sum(1)/values.sum(1),1e-6,1.)


def assert_same_spot_axis(left, right):
    for column in ['map_row','spot_index','x','y']:
        if not np.array_equal(left[column].to_numpy(),right[column].to_numpy()):
            raise ValueError(f'Shared/private spot identity mismatch: {column}')


def assert_same_peak_crosswalk(left, right):
    """Exact IDs/order, permitting only decimal serialization roundoff in m/z/ppm."""
    differences={}
    float_columns={'left_peak_mz','right_peak_mz','delta_mz','delta_ppm','absolute_delta_ppm'}
    if len(left)!=len(right):
        raise ValueError('Common prototype crosswalk row count mismatch')
    for column in left:
        a=left[column].to_numpy(); b=right[column].to_numpy()
        if column in float_columns:
            if not np.allclose(a,b,rtol=0,atol=1e-10,equal_nan=False):
                raise ValueError(f'Common prototype crosswalk mismatch: {column}')
            differences[column]=float(np.max(np.abs(a-b)))
        elif not np.array_equal(a,b):
            raise ValueError(f'Common prototype crosswalk identity mismatch: {column}')
    return {'identities_exact':True,'decimal_roundoff_atol':1e-10,'max_absolute_differences':differences}
