"""Explicit coupled-VI adapter to the existing chunked posterior-map contract."""
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import atomic_write_json, sha256_file
from ..posterior_blocks import atomic_savez_maps
from ..storage import VIStorageSettings, encode_vi_maps


def export_coupled_maps(fits, output_directory, *, field_scale, field_units,
                        fit_purpose='evaluation', complete_observation_support=False,
                        output_dtype='float32'):
    """Stream fitted features into maps consumed by prototype and LBM readers.

    ``fits`` is a mapping or iterable of (name, CoupledVIResult). Features must
    share the native evaluation coordinates, likelihood and eligible support.
    The caller explicitly declares production intent and complete acquired
    observation coverage; numerical convergence alone does not establish it.
    No parameters are refitted and no posterior variance is recombined.
    """
    if fit_purpose not in ('production', 'evaluation'):
        raise ValueError('fit_purpose must be production or evaluation')
    if not isinstance(complete_observation_support, bool):
        raise ValueError('complete_observation_support must be an explicit boolean')
    if fit_purpose == 'production' and not complete_observation_support:
        raise ValueError('Production export requires complete_observation_support=True')
    if not isinstance(field_units, str) or not field_units.strip():
        raise ValueError('Declare field_units explicitly')
    if field_scale not in ('latent_predictor', 'rate', 'observed'):
        raise ValueError('Choose latent_predictor, rate or observed field_scale explicitly')
    if fit_purpose == 'production' and field_scale == 'observed':
        raise ValueError('Production prototype/LBM maps require posterior field moments; observed moments include sampling noise')
    storage = VIStorageSettings(output_dtype=output_dtype, field_scale=field_scale)
    pairs = {'latent_predictor': ('latent_predictor_mean', 'latent_gp_variance'),
             'rate': ('rate_mean', 'rate_variance'), 'observed': ('observed_mean', 'observed_variance')}
    mean_key, variance_key = pairs[field_scale]
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(output)
    records, chunks, names = [], [], set()
    coordinates = eligible = likelihood = None
    items = fits.items() if isinstance(fits, Mapping) else fits
    for index, (name, fitted) in enumerate(items):
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Export feature names must be nonempty unique strings')
        if not fitted.converged:
            raise ValueError('Cannot export an unconverged fit')
        xy = getattr(fitted.evaluation, 'coordinates', None)
        if xy is None:
            raise ValueError('Export requires declared physical evaluation coordinates')
        current = fitted.likelihood.name
        if current == 'poisson':
            mask = fitted.likelihood.exposure.detach().cpu().numpy() > 0
        elif current == 'lgcp':
            mask = (fitted.quadrature.areas * fitted.quadrature.effort) > 0
        else:
            mask = np.ones(len(xy), dtype=bool)
        if coordinates is None:
            coordinates, eligible, likelihood = np.asarray(xy), mask, current
        elif (current != likelihood or not np.array_equal(coordinates, xy)
                or not np.array_equal(eligible, mask)):
            raise ValueError('Features must share likelihood, coordinates/order and eligible support')
        if fit_purpose == 'production' and not eligible.any():
            raise ValueError('Production export requires positive likelihood support')
        maps = fitted.predict(effort=1.0 if current == 'lgcp' and field_scale == 'rate' else None)
        encoded = encode_vi_maps(storage, maps)
        variance_audit = fitted.audit_variances(
            np.linspace(0, len(xy)-1, min(8, len(xy)), dtype=int),
            estimated_variance=maps['latent_gp_variance'])
        # The selected field is already in physical units: the existing affine
        # reconstruction contract is exactly identity, not a new normalization.
        output.mkdir(parents=True, exist_ok=True)
        file = f'vski_gp_maps_chunk_{index:04d}.npz'
        atomic_savez_maps(output/file, posterior_mean=encoded[mean_key][:, None],
                         posterior_variance=encoded[variance_key][:, None],
                         gene=np.asarray([name]), feature_position=np.asarray([index]))
        nonnegative = (field_scale == 'rate' or field_scale == 'observed' and
                       (current == 'poisson' or (fitted.likelihood.threshold >= 0).all().item()))
        records.append(dict(gene=name, feature_position=index, map_chunk_id=index,
            map_chunk_file=file, map_column=0, standardization_mean=0., standardization_std=1.,
            observation_model=current, posterior_map_scale=field_scale, field_units=field_units,
            posterior_nonnegative=nonnegative, modality=fitted.data_metadata.get('measurement', current),
            normalization_method='none', variance_semantics=('latent_field' if field_scale != 'observed'
                else 'observation_including_sampling_noise'), posterior_identity=fitted.posterior_identity))
        chunks.append(dict(chunk_id=index, file=file, n_spots=len(xy), n_features=1,
            feature_start=index, feature_stop=index+1, file_sha256=sha256_file(output/file),
            file_size_bytes=(output/file).stat().st_size, posterior_identity=fitted.posterior_identity,
            selected_latent_variance_audit=variance_audit))
        names.add(name)
    if not records:
        raise ValueError('At least one fitted feature is required')
    n, trained = len(coordinates), int(eligible.sum())
    pd.DataFrame(records).to_csv(output/'vski_gp_model_metadata.csv', index=False)
    pd.DataFrame([{k:v for k,v in row.items() if k != 'selected_latent_variance_audit'}
                  for row in chunks]).to_csv(output/'vski_gp_map_chunks.csv', index=False)
    pd.DataFrame(dict(map_row=np.arange(n), spot_index=np.arange(n),
        x=coordinates[:, 0], y=coordinates[:, 1], used_for_training=eligible,
        prediction_only=~eligible, likelihood_eligible=eligible)).to_csv(
            output/'vski_gp_map_spots.csv.gz', index=False)
    manifest = dict(schema_version=1, experimental=True, method='coupled_conditional_gaussian_vi',
        observation_model=likelihood, posterior_map_scale=field_scale, field_units=field_units,
        coordinate_units='micron', variance_includes_global_local_cross_covariance=True,
        n_training_spots=trained, n_prediction_spots=n, n_features=len(records),
        fit_purpose=fit_purpose, downstream_ready=fit_purpose == 'production',
        n_likelihood_eligible_spots=trained, full_likelihood_coverage=bool(complete_observation_support),
        likelihood_coverage_fraction=1.0 if complete_observation_support else None,
        full_training_coverage=trained == n, training_coverage_fraction=trained/n,
        complete_prediction_support=True, coverage_declaration='explicit_caller_acquired_support',
        parameter_uncertainty_included=False, storage=asdict(storage), chunks=chunks,
        metadata_file='vski_gp_model_metadata.csv', spot_file='vski_gp_map_spots.csv.gz',
        chunk_index_file='vski_gp_map_chunks.csv')
    atomic_write_json(output/'vski_gp_maps_manifest.json', manifest)
    return output
