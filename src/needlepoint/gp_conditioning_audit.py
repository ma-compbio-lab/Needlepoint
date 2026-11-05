"""Numerical audits for opt-in frozen Gaussian conditioning outputs.

These checks neither replace ``artifacts.audit_gp_artifact`` nor confer
production or downstream eligibility.  In particular, finite negative Gaussian
means pass this numerical audit while the nonnegative downstream store may
reject them.  No Torch/GPyTorch imports are needed to inspect saved moments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .storage import GP_MAP_FIELD_TYPES


CONDITIONING_MAP_FIELDS = (
    'posterior_mean', 'posterior_variance', 'posterior_predictive_variance',
    'global_posterior_mean', 'local_posterior_mean',
    'global_posterior_variance', 'local_posterior_variance',
    'global_local_posterior_covariance',
)


def _solver_audit(receipts):
    if receipts is None:
        return {'status': 'unverified', 'n_rhs': 0, 'failures': ['No solver receipts supplied']}
    if isinstance(receipts, dict):
        receipts = [receipts] if 'converged' in receipts else list(receipts.values())
    receipts = list(receipts)
    if not receipts:
        return {'status': 'unverified', 'n_rhs': 0, 'failures': ['No solver receipts supplied']}
    failures, n_rhs, max_relative, max_absolute = [], 0, 0.0, 0.0
    for index, item in enumerate(receipts):
        if not isinstance(item, dict):
            failures.append(f'Receipt {index} is not a mapping')
            continue
        try:
            converged = np.asarray(item['converged'])
            iterations = np.asarray(item['iterations'])
            relative = np.asarray(item['relative_residual'], dtype=np.float64)
            absolute = np.asarray(item['absolute_residual'], dtype=np.float64)
            shape = converged.shape
            if (converged.dtype.kind != 'b' or converged.size == 0
                    or any(v.shape != shape for v in (iterations, relative, absolute))):
                raise ValueError('per-RHS receipt shapes/types are inconsistent')
            n_rhs += converged.size
            if (not np.isfinite(relative).all() or not np.isfinite(absolute).all()
                    or not np.isfinite(iterations).all() or np.any(iterations < 0)
                    or np.any(iterations != np.floor(iterations))
                    or np.any(relative < 0) or np.any(absolute < 0)):
                raise ValueError('invalid residuals or iteration counts')
            if item.get('status') != 'converged' or not converged.all():
                failures.append(f'Receipt {index} contains an unconverged solve')
            if any(reason for reason in item.get('failure_reasons', [])):
                failures.append(f'Receipt {index} reports a failure reason')
            max_relative = max(max_relative, float(relative.max()))
            max_absolute = max(max_absolute, float(absolute.max()))
        except (KeyError, TypeError, ValueError) as error:
            failures.append(f'Receipt {index}: {error}')
    return {'status': 'failed' if failures else 'passed', 'n_rhs': int(n_rhs),
            'max_relative_residual': max_relative, 'max_absolute_residual': max_absolute,
            'failures': failures}


def audit_conditioning_moments(arrays, observation_noise_variance, *,
                               solver_receipts=None, statistical_status=None,
                               rtol=1e-5, atol=1e-8, raise_on_failure=False):
    """Check coupled Gaussian moment identities and independent solver evidence.

    All arrays and observation noise must use the same units.  Noise may be a
    scalar or broadcast over the feature axis.  Component identities are checked
    against the scale of their terms so that cancellation in a small total
    variance does not produce a misleading float32 storage failure.  Variances
    must nevertheless be nonnegative, without clipping or a sign tolerance.

    ``status`` covers moment/solver checks only.  Statistical accuracy is a
    separate gate, and support eligibility is deliberately not evaluated.
    """
    if not np.isfinite([rtol, atol]).all() or rtol < 0 or atol < 0:
        raise ValueError('Audit tolerances must be finite and nonnegative')
    failures, checks, values = [], {}, {}
    for name in CONDITIONING_MAP_FIELDS:
        if name not in arrays:
            failures.append(f'Missing conditioning map {name}')
            continue
        value = np.asarray(arrays[name], dtype=np.float64)
        values[name] = value
        if not value.size or not np.isfinite(value).all():
            failures.append(f'Nonfinite or empty conditioning map {name}')
        if GP_MAP_FIELD_TYPES[name] == 'variance' and np.any(value < 0):
            failures.append(f'Negative variance in {name}')
    if len({v.shape for v in values.values()}) > 1:
        failures.append('Conditioning map shapes disagree')
    noise = np.asarray(observation_noise_variance, dtype=np.float64)
    if not noise.size or not np.isfinite(noise).all() or np.any(noise <= 0):
        failures.append('Observation noise variance must be finite and positive')
    if not failures:
        total = values['posterior_variance']
        global_var = values['global_posterior_variance']
        local_var = values['local_posterior_variance']
        cross = values['global_local_posterior_covariance']
        reconstructed = global_var + local_var + 2 * cross
        scale = global_var + local_var + 2 * np.abs(cross)
        error = np.abs(total - reconstructed)
        checks['component_identity'] = bool(np.all(error <= atol + rtol * scale))
        checks['component_identity_max_absolute_error'] = float(error.max())
        bound = np.sqrt(global_var) * np.sqrt(local_var)
        excess = np.maximum(np.abs(cross) - bound, 0)
        checks['cross_covariance_cauchy_bound'] = bool(np.all(excess <= atol + rtol * bound))
        checks['cross_covariance_max_excess'] = float(excess.max())
        try:
            expected = np.broadcast_to(total + noise, total.shape)
            observed = values['posterior_predictive_variance']
            error = np.abs(observed - expected)
            checks['predictive_variance_identity'] = bool(
                np.all(error <= atol + rtol * np.maximum(observed, expected)))
            checks['predictive_identity_max_absolute_error'] = float(error.max())
        except ValueError:
            checks['predictive_variance_identity'] = False
            failures.append('Observation noise does not broadcast to the moment shape')
        for name in ('component_identity', 'cross_covariance_cauchy_bound',
                     'predictive_variance_identity'):
            if not checks[name]:
                failures.append(f'Failed {name}')
    solver = _solver_audit(solver_receipts)
    statistical = dict(statistical_status or {})
    target = statistical.get('accuracy_target_passed')
    statistical['accuracy_gate'] = ('passed' if target is True else 'failed'
                                     if target is False else 'unverified')
    numerical_status = ('failed' if failures or solver['status'] == 'failed'
                        else 'unverified' if solver['status'] == 'unverified' else 'passed')
    result = {
        'schema_version': 1, 'audit': 'frozen_joint_conditioning_numerical',
        'status': numerical_status, 'moment_status': 'failed' if failures else 'passed',
        'rtol': float(rtol), 'atol': float(atol), 'checks': checks, 'failures': failures,
        'solver': solver, 'statistical': statistical,
        'support_eligibility_evaluated': False,
        'downstream_eligibility_evaluated': False,
        'negative_mean_count': int(np.sum(values.get('posterior_mean', np.array([])) < 0)),
        'note': ('Numerical identities and solver convergence do not certify Monte Carlo '
                 'accuracy, support eligibility, nonnegative downstream input, '
                 'hyperparameter accuracy, or biological calibration.'),
    }
    if raise_on_failure and numerical_status == 'failed':
        raise ValueError('; '.join(failures + solver['failures']))
    return result


def audit_conditioning_artifact(directory, *, rtol=None, atol=None):
    """Read native chunk files and run numerical checks without a support audit.

    The manifest must name observation noise in exported-map units and provide
    the solver/statistical receipts.  File hashes, when declared, are checked.
    No missing cross-covariances are fabricated for legacy factorized outputs.
    """
    from .artifacts import sha256_file

    directory = Path(directory).resolve()
    manifest_path = directory / 'vski_gp_maps_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('posterior_conditioning') != 'coupled':
        raise ValueError('Artifact does not declare coupled posterior conditioning')
    noise = np.asarray(manifest['observation_noise_variance'], dtype=np.float64)
    receipts = manifest.get('solver_receipts', manifest.get('conditioning_solver_receipts_file'))
    if isinstance(receipts, str):
        path = (directory / receipts).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Solver receipt path leaves artifact directory')
        receipts = json.loads(path.read_text())
    if isinstance(receipts, dict) and 'receipts' in receipts:
        receipts = receipts['receipts']
    statistical = manifest.get('statistical_status',
        manifest.get('variance_statistical_status', manifest.get('variance_estimator')))
    dtype = manifest.get('output_dtype', manifest.get('storage', {}).get('output_dtype', 'float32'))
    tolerances = dict(rtol=(1e-3 if dtype == 'float16' else 1e-5) if rtol is None else rtol,
                      atol=(1e-5 if dtype == 'float16' else 1e-8) if atol is None else atol)
    chunks = []
    for chunk in manifest['chunks']:
        path = (directory / chunk['file']).resolve()
        if not path.is_relative_to(directory):
            raise ValueError('Map chunk path leaves artifact directory')
        if chunk.get('file_sha256') and sha256_file(path) != chunk['file_sha256']:
            raise ValueError('Conditioning map chunk checksum mismatch')
        chunk_noise = noise
        if noise.ndim and noise.size > 1:
            chunk_noise = noise[int(chunk['feature_start']):int(chunk['feature_stop'])]
        with np.load(path, allow_pickle=False) as arrays:
            receipt = audit_conditioning_moments(arrays, chunk_noise,
                solver_receipts=receipts, statistical_status=statistical, **tolerances)
        chunks.append(dict(file=chunk['file'], **receipt))
    status = ('failed' if not chunks or any(c['status'] == 'failed' for c in chunks)
              else 'unverified' if any(c['status'] == 'unverified' for c in chunks) else 'passed')
    return {'schema_version': 1, 'audit': 'frozen_joint_conditioning_numerical',
            'status': status, 'manifest_sha256': sha256_file(manifest_path), 'chunks': chunks,
            'support_eligibility_evaluated': False, 'downstream_eligibility_evaluated': False}
