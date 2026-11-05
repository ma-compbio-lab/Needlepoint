"""Streaming, fixed-budget perturb-and-solve joint posterior moments.

Only marginal maps are exported. Unique withheld-site residuals follow the
source's independent finite-query conditional convention. No sample-path API.
"""
from __future__ import annotations

import hashlib
import math
import time
import numpy as np
import torch
from scipy.stats import chi2

from needlepoint.gp_joint_conditioning import FrozenConditioningSettings, solve_precision


def _seed_words(operators, seed, draw, stream):
    # Model identity excludes observed y and q family/rank. Query order is not
    # used: reordering predictions must preserve common-random numerical draws.
    snapshot = operators.snapshot
    key = f'{snapshot.feature_position}:{snapshot.feature_name}'
    digest = hashlib.sha256(key.encode()).digest()
    return [int(seed), int.from_bytes(digest[:4], 'little'), int(draw), int(stream)]


@torch.no_grad()
def perturbation_rhs(operators, draw_indices, seed):
    """Independent deterministic RHSs with covariance equal to joint precision."""
    draw_indices = tuple(draw_indices)
    if not draw_indices:
        raise ValueError('At least one draw index is required')
    m, n = operators.global_prior.size, operators.local_prior.size
    arrays = []
    for stream, size in enumerate((m, n, n)):
        cols = [np.random.default_rng(np.random.SeedSequence(_seed_words(operators, seed, i, stream)))
                .standard_normal(size) for i in draw_indices]
        arrays.append(torch.as_tensor(np.column_stack(cols), dtype=operators.dtype,
                                      device=operators.device))
    xg, xl, xy = arrays
    prior = torch.cat((xg, operators.local_prior.whiten_local_t(xl)), dim=0)
    return prior + operators.training_t_matmul(xy)/math.sqrt(operators.noise_variance)


def _summarize(accum, residual, draws):
    global_var, local_propagated, cross, propagated = [v/draws for v in accum]
    # Query residual is known and integrated analytically, not sampled.
    local_var = local_propagated + residual
    total = propagated + residual
    se = propagated * math.sqrt(2/draws)
    lower = draws*propagated/chi2.ppf(.975, draws) + residual
    upper = draws*propagated/chi2.ppf(.025, draws) + residual
    arrays = dict(global_posterior_variance=global_var, local_posterior_variance=local_var,
                  global_local_posterior_covariance=cross, posterior_variance=total,
                  variance_monte_carlo_se=se, variance_pointwise_lower=lower,
                  variance_pointwise_upper=upper, propagated_variance=propagated)
    return {key: value.cpu().numpy().copy() for key, value in arrays.items()}


@torch.no_grad()
def estimate_joint_moments(operators, query_batches=None, variance_settings=None, *,
                           record_draw_counts=(), progress=None):
    """Estimate all-site component moments using the SAME draws.

    RHS batching and query batching change memory, not random draws. Optional
    prespecified prefix budgets reuse draws, explicitly yielding correlated
    budget comparisons. Independent estimator seeds remain independent.
    """
    settings = variance_settings or FrozenConditioningSettings()
    p = operators.query.size
    if query_batches is None:
        batches = [torch.arange(i, min(i+settings.query_batch_size, p), device=operators.device)
                   for i in range(0, p, settings.query_batch_size)]
    else:
        batches = [torch.as_tensor(rows, dtype=torch.long, device=operators.device) for rows in query_batches]
        joined = torch.cat(batches)
        if len(joined) != p or not torch.equal(torch.sort(joined).values, torch.arange(p, device=operators.device)):
            raise ValueError('Query batches must partition every query row exactly once')
    budgets = sorted(set([int(v) for v in record_draw_counts] + [settings.fixed_draws]))
    if budgets[0] <= 0 or budgets[-1] > settings.fixed_draws:
        raise ValueError('Prefix budgets must be positive and no greater than fixed_draws')
    # Conservative workspace estimate (operator arrays accounted separately).
    needed = (4*p + 16*operators.dimension*settings.rhs_batch_size +
              4*min(settings.query_batch_size,p)*settings.rhs_batch_size)*torch.tensor([],dtype=operators.dtype).element_size()
    if needed > settings.memory_cap_bytes:
        raise MemoryError(f'Estimated estimator workspace {needed} exceeds configured cap')
    accum = [torch.zeros(p, dtype=operators.dtype, device=operators.device) for _ in range(4)]
    receipts, prefix = [], {}
    start = time.perf_counter()
    done = 0
    for budget in budgets:
        while done < budget:
            stop = min(done+settings.rhs_batch_size, budget)
            rhs = perturbation_rhs(operators, range(done,stop), settings.random_seed)
            solved = solve_precision(operators, rhs, settings)
            receipts.append(dict(draw_start=done, draw_stop=stop, **solved.receipt))
            # Reuse one structured global root across every query block.
            m = operators.global_prior.size
            global_deviations = operators.global_prior.sqrt_matmul(solved.solution[:m])
            for rows in batches:
                dg = operators.query.interpolation.matmul(global_deviations, rows=rows)
                dl = operators.query.local_interpolation.matmul(solved.solution[m:], rows=rows)
                accum[0][rows] += dg.square().sum(1)
                accum[1][rows] += dl.square().sum(1)
                accum[2][rows] += (dg*dl).sum(1)
                accum[3][rows] += (dg+dl).square().sum(1)
            done = stop
            if progress is not None:
                progress(done, settings.fixed_draws, receipts[-1])
        prefix[budget] = _summarize(accum, operators.query.residual, done)
    status = dict(method='joint_perturb_and_solve', fixed_draws=settings.fixed_draws,
                  random_seed=settings.random_seed, pointwise_relative_sd=math.sqrt(2/settings.fixed_draws),
                  intervals='pointwise chi-square 95%; propagated variance plus analytic residual',
                  simultaneous_coverage=False, adaptive_stopping=False,
                  high_accuracy_target_passed=None,
                  finite_query_convention='independent residual per unique withheld coordinate',
                  zero_centered_denominator='S', prefix_budgets_correlated=True)
    return dict(arrays=prefix[settings.fixed_draws], prefixes=prefix, solver_receipts=receipts,
                statistical_status=status, elapsed_seconds=time.perf_counter()-start,
                estimated_workspace_bytes=needed)
