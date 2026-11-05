"""Opt-in, frozen Gaussian conditioning; independent-column matrix-free PCG.

The solver never changes the model jitter and never differentiates through a fit.
Residual tolerances are numerical checks, not covariance accuracy certificates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import torch


@dataclass(frozen=True)
class FrozenConditioningSettings:
    operator_backend: str = 'structured'
    work_dtype: str = 'float64'
    output_dtype: str = 'float32'
    device: str = 'cpu'
    solver: str = 'pcg'
    preconditioner: str = 'scaled_block_jacobi'
    rtol: float = 1e-9
    atol: float = 1e-12
    max_iterations: int = 4000
    residual_refresh: int = 25
    fixed_draws: int = 256
    rhs_batch_size: int = 8
    query_batch_size: int = 2048
    feature_batch_size: int = 1
    audit_rows: tuple[int, ...] = ()
    audit_relative_tolerance: float = .001
    random_seed: int = 20261013
    memory_cap_bytes: int = 8 * 1024**3
    mean_mode: str = 'joint'
    failure_policy: str = 'raise'

    def __post_init__(self):
        if self.operator_backend != 'structured' or self.solver != 'pcg':
            raise ValueError('Only structured operators and independent-column PCG are implemented')
        if self.work_dtype not in ('float32', 'float64') or self.output_dtype != 'float32':
            raise ValueError('Use float32/float64 work and audited float32 output')
        if self.preconditioner not in ('identity', 'scaled_block_jacobi'):
            raise ValueError('Unknown fixed SPD preconditioner')
        if self.mean_mode not in ('joint', 'preserve_variational'):
            raise ValueError('Unknown mean policy')
        if self.failure_policy not in ('raise', 'record'):
            raise ValueError('failure_policy must be raise or record')
        if self.feature_batch_size != 1:
            raise ValueError('Feature batching has not been benchmarked; use one feature')
        for name in ('max_iterations', 'residual_refresh', 'fixed_draws', 'rhs_batch_size',
                     'query_batch_size', 'memory_cap_bytes'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not (math.isfinite(self.rtol) and math.isfinite(self.atol) and
                0 <= self.rtol < 1 and 0 <= self.atol and 0 < self.audit_relative_tolerance < 1):
            raise ValueError('Invalid numerical tolerance')


@dataclass
class SolveResult:
    solution: torch.Tensor
    receipt: dict


class ConditioningSolveError(RuntimeError):
    def __init__(self, receipt):
        super().__init__('Frozen conditioning PCG failed: ' + str(receipt['failure_reasons']))
        self.receipt = receipt


@torch.no_grad()
def independent_pcg(matmul, rhs, settings=None, *, inverse_preconditioner=None, initial=None):
    """Batched independent CG, with true residual checks for every RHS.

    A fixed linear SPD inverse preconditioner is required. This is not block CG.
    Explicit residuals are recomputed every refresh interval. Recurrences restart
    only for material residual drift or a false recursive convergence claim;
    replacing accurate residuals every interval would discard useful conjugacy.
    """
    settings = settings or FrozenConditioningSettings()
    vector = rhs.ndim == 1
    b = rhs.detach().reshape(rhs.shape[0], -1)
    if b.ndim != 2 or not b.is_floating_point() or not torch.isfinite(b).all():
        raise ValueError('RHS must be a finite floating vector or matrix')
    if rhs.ndim not in (1, 2):
        raise ValueError('RHS must be vector or matrix')
    x = torch.zeros_like(b) if initial is None else initial.detach().reshape_as(b).clone()
    if not torch.isfinite(x).all():
        raise ValueError('Initial iterate must be finite')
    apply_preconditioner = inverse_preconditioner or (lambda r: r)
    start = time.perf_counter()
    bnorm = torch.linalg.vector_norm(b, dim=0)
    threshold = torch.maximum(settings.rtol * bnorm, bnorm.new_full(bnorm.shape, settings.atol))
    r = b - matmul(x)
    residual_norm = torch.linalg.vector_norm(r, dim=0)
    active = residual_norm > threshold
    failed = torch.zeros_like(active)
    iterations = torch.zeros(b.shape[1], dtype=torch.int64, device=b.device)
    reasons = [None] * b.shape[1]
    z = apply_preconditioner(r)
    rho = (r * z).sum(0)
    p = torch.where(active[None, :], z, torch.zeros_like(z))
    matvec_count = 1
    for step in range(1, settings.max_iterations + 1):
        if not active.any():
            break
        ap = matmul(p)
        matvec_count += 1
        curvature = (p * ap).sum(0)
        bad = active & (~torch.isfinite(curvature) | (curvature <= 0) |
                        ~torch.isfinite(rho) | (rho <= 0))
        for i in bad.nonzero().flatten().tolist():
            reasons[i] = 'nonfinite_or_nonpositive_curvature_or_preconditioner'
        failed |= bad
        active &= ~bad
        if not active.any():
            break
        alpha = torch.where(active, rho / torch.where(active, curvature, torch.ones_like(curvature)), 0)
        x += p * alpha
        iterations[active] = step
        r -= ap * alpha
        recursive_norm = torch.linalg.vector_norm(r, dim=0)
        # Verify a recursive convergence claim immediately, and periodically
        # replace all residuals. Only explicit residuals can stop a column.
        refresh = step % settings.residual_refresh == 0
        candidates = active & (recursive_norm <= threshold)
        restart = torch.zeros_like(active)
        residual_norm = recursive_norm
        if refresh or candidates.any():
            explicit = b - matmul(x)
            matvec_count += 1
            checked = active if refresh else candidates
            true_norm = torch.linalg.vector_norm(explicit, dim=0)
            drift = torch.linalg.vector_norm(explicit-r, dim=0)
            # Preserve CG conjugacy when the recurrence still tracks reality.
            replace = checked & ((drift > .01*true_norm.clamp_min(settings.atol)) | candidates)
            r[:, replace] = explicit[:, replace]
            restart = replace
            residual_norm = torch.where(checked, true_norm, recursive_norm)
        bad = active & (~torch.isfinite(residual_norm) | ~torch.isfinite(x).all(0))
        for i in bad.nonzero().flatten().tolist():
            reasons[i] = 'nonfinite_iterate_or_residual'
        failed |= bad
        active &= ~bad
        active &= residual_norm > threshold
        if not active.any():
            break
        z = apply_preconditioner(r)
        new_rho = (r * z).sum(0)
        beta = torch.where(active, new_rho / torch.where(rho != 0, rho, torch.ones_like(rho)), 0)
        beta = torch.where(restart, torch.zeros_like(beta), beta)
        p = torch.where(active[None, :], z + p * beta, torch.zeros_like(p))
        rho = new_rho
    r = b - matmul(x)
    matvec_count += 1
    residual_norm = torch.linalg.vector_norm(r, dim=0)
    converged = (residual_norm <= threshold) & ~failed & torch.isfinite(residual_norm)
    for i in (~converged).nonzero().flatten().tolist():
        reasons[i] = reasons[i] or 'maximum_iterations_or_final_true_residual'
    relative = torch.where(bnorm > 0, residual_norm / bnorm.clamp_min(torch.finfo(b.dtype).tiny), residual_norm)
    receipt = dict(status='converged' if converged.all() else 'failed',
                   algorithm='batched_independent_pcg', converged=converged.cpu().tolist(),
                   iterations=iterations.cpu().tolist(), relative_residual=relative.cpu().tolist(),
                   absolute_residual=residual_norm.cpu().tolist(), rhs_norm=bnorm.cpu().tolist(),
                   failure_reasons=reasons, matvec_calls=matvec_count,
                   elapsed_seconds=time.perf_counter()-start, rtol=settings.rtol, atol=settings.atol,
                   residual_refresh=settings.residual_refresh, residual_replacement_relative_drift=.01,
                   work_dtype=str(b.dtype),
                   zero_initialization=initial is None)
    if receipt['status'] == 'failed' and settings.failure_policy == 'raise':
        raise ConditioningSolveError(receipt)
    return SolveResult(x[:, 0] if vector else x, receipt)


def make_preconditioner(operators, settings):
    if settings.preconditioner == 'identity':
        return None, dict(name='identity')
    diagonal = operators.scaled_jacobi_diagonal
    if callable(diagonal):
        diagonal = diagonal()
    if not torch.isfinite(diagonal).all() or (diagonal <= 0).any():
        raise ValueError('Non-SPD Jacobi diagonal')
    def apply(rhs):
        return rhs / diagonal[:, None] if rhs.ndim == 2 else rhs / diagonal
    return apply, dict(name='scaled_block_jacobi',
                      global_rule='1 + N * diag(Ku) / (M * noise_variance)',
                      local_rule='diag(R_local.T R_local) + 1/noise_variance',
                      diagonal_min=float(diagonal.min()), diagonal_max=float(diagonal.max()),
                      global_diagonal=float(diagonal[0]))


@torch.no_grad()
def solve_precision(operators, rhs, settings=None, *, initial=None):
    settings = settings or FrozenConditioningSettings()
    inverse, receipt = make_preconditioner(operators, settings)
    result = independent_pcg(operators.precision_matmul, rhs, settings,
                             inverse_preconditioner=inverse, initial=initial)
    result.receipt['preconditioner'] = receipt
    return result


@torch.no_grad()
def solve_joint_mean(snapshot, operators, solver_settings=None):
    """Return mixed-coordinate posterior mean and explicit solve receipt."""
    if snapshot.training_values is None:
        raise ValueError('Posterior means require hash-verified standardized training values')
    values = torch.tensor(snapshot.training_values, dtype=operators.dtype, device=operators.device)
    prior_mean = operators.query.prior_mean[torch.as_tensor(snapshot.training_query_rows.copy(),
                                                           device=operators.device)]
    rhs = operators.training_t_matmul(values-prior_mean) / operators.noise_variance
    return solve_precision(operators, rhs, solver_settings)


@torch.no_grad()
def reference_variances(operators, query_rows, solver_settings=None):
    """Selected marginals with deterministic variational lower/upper bounds.

    The upper bound uses Q0^-1 only here; it is never used inside a precision
    action or standard PCG iteration. Floating point slack is not added to bounds.
    """
    settings = solver_settings or FrozenConditioningSettings()
    rows = torch.as_tensor(query_rows, dtype=torch.long, device=operators.device)
    if rows.ndim != 1 or rows.numel() == 0 or len(torch.unique(rows)) != len(rows):
        raise ValueError('Audit rows must be a nonempty list of unique query rows')
    if (rows < 0).any() or (rows >= operators.query.size).any():
        raise ValueError('Audit rows outside query range')
    values, lower, upper, receipts = [], [], [], []
    for batch in rows.split(settings.rhs_batch_size):
        eye = torch.eye(len(batch), dtype=operators.dtype, device=operators.device)
        h = operators.query.t_matmul(eye, rows=batch)
        solved = solve_precision(operators, h, settings)
        pa = operators.precision_matmul(solved.solution)
        r = h-pa
        ha = (h * solved.solution).sum(0)
        lo = 2*ha - (solved.solution*pa).sum(0) + operators.query.residual[batch]
        energy = (r*operators.prior_inverse_matmul(r)).sum(0)
        if not torch.isfinite(energy).all() or (energy < 0).any():
            raise ValueError('Invalid residual prior energy in reference enclosure')
        values.append(ha+operators.query.residual[batch]); lower.append(lo); upper.append(lo+energy)
        receipts.append(solved.receipt)
    value, lo, hi = map(torch.cat, (values, lower, upper))
    floor = torch.finfo(value.dtype).eps * max(1., float(operators.global_prior.diagonal))
    relative_width = (hi-lo)/value.abs().clamp_min(floor)
    return dict(query_rows=rows.cpu().numpy(), variance=value.cpu().numpy(),
                lower=lo.cpu().numpy(), upper=hi.cpu().numpy(),
                relative_width=relative_width.cpu().numpy(),
                enclosure_target_passed=bool((relative_width <= settings.audit_relative_tolerance).all()),
                solver_receipts=receipts, reference_scale_floor=floor)


def condition_frozen_gaussian_fit(snapshot, output_dir, settings=None, *, record_draw_counts=(), progress=None):
    """Condition one immutable feature snapshot into a NEW native map directory.

    Numerical audits and support audits are separate receipts. A hybrid output
    carries the exact source canonical mean and exports the joint mean alongside
    it; component means remain explicitly the coherent joint component means.
    """
    import json
    from pathlib import Path
    import shutil
    import numpy as np
    import pandas as pd
    from needlepoint.artifacts import atomic_write_json, sha256_file, audit_gp_artifact
    from needlepoint.storage import GPStorageSettings, encode_gp_maps, storage_manifest
    from needlepoint.posterior_blocks import atomic_savez_maps
    from needlepoint.gp_prior_operators import build_frozen_prior_operators
    from needlepoint.gp_variance_estimation import estimate_joint_moments
    from needlepoint.gp_conditioning_audit import audit_conditioning_moments

    from collections.abc import Mapping
    def plain(value):
        if isinstance(value, Mapping): return {k:plain(v) for k,v in value.items()}
        if isinstance(value, (tuple,list)): return [plain(v) for v in value]
        return value

    settings = settings or FrozenConditioningSettings()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Conditioning requires a new empty output directory')
    source = Path(snapshot.source_provenance['source_dir'])
    if output.resolve() == source.resolve() or output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Conditioned artifacts must be separate from the frozen source')
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        n,p,k = len(snapshot.training_coordinates),len(snapshot.query_coordinates),snapshot.local_neighbors
        m = len(snapshot.grid_axes[0])*len(snapshot.grid_axes[1])
        itemsize = 8 if settings.work_dtype=='float64' else 4
        # Conservative accounting includes source arrays, cached sparse factors,
        # batched local Cholesky work, PCG workspaces, and map accumulators.
        estimated_memory = itemsize*(16*(n+p)*k + 16*p*snapshot.interpolation_weights.shape[1] +
            16*m + 8*min(512,n)*k*k + 20*(n+m)*settings.rhs_batch_size + 24*p)
        if estimated_memory > settings.memory_cap_bytes:
            raise MemoryError(f'Estimated conditioning workspace {estimated_memory} exceeds memory cap')
        operators = build_frozen_prior_operators(snapshot, dtype=getattr(torch,settings.work_dtype), device=settings.device)
        build_seconds = time.perf_counter()-start
        mean_result = solve_joint_mean(snapshot, operators, settings)
        gm, lm = operators.query.matmul_parts(mean_result.solution)
        gm = gm + operators.query.prior_mean
        joint_mean = (gm+lm).cpu().numpy()
        source_mean = snapshot.source_variational_mean
        if settings.mean_mode == 'preserve_variational' and source_mean is None:
            raise ValueError('Hybrid mode requires verified source variational means')
        refs = reference_variances(operators, settings.audit_rows, settings) if settings.audit_rows else None
        moments = estimate_joint_moments(operators, variance_settings=settings,
                                        record_draw_counts=record_draw_counts, progress=progress)
        arrays = {k:v for k,v in moments['arrays'].items() if k in (
            'posterior_variance','global_posterior_variance','local_posterior_variance',
            'global_local_posterior_covariance')}
        arrays.update(posterior_mean=joint_mean if settings.mean_mode=='joint' else np.asarray(source_mean),
                      global_posterior_mean=gm.cpu().numpy(), local_posterior_mean=lm.cpu().numpy(),
                      posterior_predictive_variance=arrays['posterior_variance']+operators.noise_variance)
        storage = GPStorageSettings(output_dtype=settings.output_dtype,retain_component_maps=True,
                                    retain_predictive_variance=True)
        encoded = encode_gp_maps(storage, {k:v[:,None] for k,v in arrays.items()})
        all_receipts = [mean_result.receipt]+([] if refs is None else refs['solver_receipts'])+moments['solver_receipts']
        audit = audit_conditioning_moments(encoded,operators.noise_variance,solver_receipts=all_receipts,
                                          statistical_status=moments['statistical_status'],rtol=2e-5,atol=1e-7,
                                          raise_on_failure=True)
        encoded.update(gene=np.asarray([snapshot.feature_name]),
                       feature_position=np.asarray([snapshot.feature_position]))
        mapfile = output/'vski_gp_maps_chunk_0000.npz'
        atomic_savez_maps(mapfile,**encoded)
        diagnostics = {key:val for key,val in moments['arrays'].items() if key not in arrays}
        diagnostics['joint_posterior_mean'] = joint_mean
        if source_mean is not None:
            diagnostics['source_variational_mean'] = source_mean
            diagnostics['posterior_mean_difference'] = joint_mean-source_mean
        if refs is not None:
            diagnostics.update(variance_reference_rows=refs['query_rows'],variance_reference=refs['variance'],
                               variance_reference_lower=refs['lower'],variance_reference_upper=refs['upper'],
                               variance_reference_relative_width=refs['relative_width'])
        np.savez(output/'conditioning_diagnostics.npz',**diagnostics)
        for count,prefix in moments['prefixes'].items():
            np.savez(output/f'variance_prefix_{count}.npz',**prefix)
        original = json.loads((source/'vski_gp_maps_manifest.json').read_text())
        manifest = {k:v for k,v in original.items() if k not in (
            'chunks','variational_family','component_covariance_under_q','inference_note',
            'training_history_file','pretraining_history_file','hyperparameter_history_file','stage_timing_file')}
        metadata = pd.read_csv(source/original.get('metadata_file','vski_gp_model_metadata.csv'))
        metadata = metadata.loc[metadata.gene.astype(str)==snapshot.feature_name].copy()
        if len(metadata)!=1:
            raise ValueError('Source feature metadata must match exactly one frozen feature')
        metadata['feature_position'] = snapshot.feature_position
        metadata['map_chunk_id'] = 0
        metadata['map_chunk_file'] = mapfile.name
        metadata['map_column'] = 0
        metadata.to_csv(output/'vski_gp_model_metadata.csv',index=False)
        shutil.copy2(source/original.get('spot_file','vski_gp_map_spots.csv.gz'),output/'vski_gp_map_spots.csv.gz')
        chunk = dict(chunk_id=0,file=mapfile.name,n_spots=operators.query.size,
                     n_training_spots=operators.local_prior.size,n_features=1,feature_start=0,feature_stop=1,
                     file_size_bytes=mapfile.stat().st_size,file_sha256=sha256_file(mapfile))
        pd.DataFrame([chunk]).to_csv(output/'vski_gp_map_chunks.csv',index=False)
        manifest['source_variational_global_covariance'] = manifest.pop('global_covariance',None)
        manifest.update(training_method='original staged variational training',
                        inference_backend='frozen_kernel_joint_gaussian_precision',
                        posterior_conditioning='coupled',hyperparameters_frozen=True,
                        hyperparameter_uncertainty_included=False,
                        mean_mode=settings.mean_mode,moments_are_hybrid=settings.mean_mode!='joint',
                        component_mean_policy='joint component means; canonical mean policy given by mean_mode',
                        covariance_target='same fitted SKI/VNNGP prior',
                        local_predictive_residual_convention=snapshot.query_convention,
                        latent_variance='global_variance + local_variance + 2*global_local_covariance',
                        inference_note='numerical frozen-prior Gaussian conditioning; fixed-budget Monte Carlo diagonal',
                        solver_numerical_status='converged' if all(r['status']=='converged' for r in all_receipts) else 'failed',
                        variance_estimator=moments['statistical_status'],
                        observation_noise_variance=operators.noise_variance,
                        final_joint_noise_standardized=operators.noise_variance,
                        source_snapshot_identity=snapshot.target_identity,
                        source_snapshot_provenance=plain(snapshot.source_provenance),
                        conditioning_settings=asdict(settings), storage=storage_manifest(storage),
                        n_features=1,chunks=[chunk],metadata_file='vski_gp_model_metadata.csv',
                        spot_file='vski_gp_map_spots.csv.gz',chunk_index_file='vski_gp_map_chunks.csv',
                        output_dtype=settings.output_dtype, elapsed_seconds=time.perf_counter()-start,
                        conditioning_operator_build_seconds=build_seconds,
                        estimated_conditioning_workspace_bytes=estimated_memory,
                        memory_cap_scope='conservative numeric workspace estimate; excludes Python/library runtime',
                        conditioning_diagnostics_file='conditioning_diagnostics.npz',
                        conditioning_numerical_audit_file='conditioning_numerical_audit.json',
                        conditioning_solver_receipts_file='conditioning_solver_receipts.json')
        atomic_write_json(output/'vski_gp_maps_manifest.json',manifest)
        atomic_write_json(output/'conditioning_numerical_audit.json',audit)
        atomic_write_json(output/'conditioning_solver_receipts.json',dict(receipts=all_receipts))
        atomic_write_json(output/'conditioning_support_audit.json',audit_gp_artifact(output).to_dict())
        atomic_write_json(output/'frozen_snapshot.json',dict(target_identity=snapshot.target_identity,
                           provenance=plain(snapshot.source_provenance),schema_version=snapshot.schema_version,
                           convention_version=snapshot.convention_version))
        atomic_write_json(output/'complete.json',dict(status='complete',target_identity=snapshot.target_identity,
                          artifact_hashes={p.name:sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}))
        return dict(manifest=manifest,numerical_audit=audit,reference=refs,moments=moments,
                    mean_receipt=mean_result.receipt,output_dir=str(output))
    except Exception as error:
        atomic_write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
                          solver_receipt=getattr(error,'receipt',None),settings=asdict(settings)))
        raise
