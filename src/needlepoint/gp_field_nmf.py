"""Graph-free Frobenius NMF of fixed GP fields; optional fixed precision weights.

This is not the authors' KL Neighborhood NMF or graph-regularized SEPAR solver.
Their useful initialization/normalization conventions are explicitly separated
from the objective. No fitted GP or prototype is mutated.
"""
from dataclasses import dataclass, asdict
import time
import numpy as np
import torch


@dataclass
class FieldNMFSettings:
    rank: int = 20
    seed: int = 0
    warm_starts: int = 3
    warm_iterations: int = 50
    max_iterations: int = 500
    check_every: int = 10
    tolerance: float = 1e-5
    patience: int = 5
    chunk_rows: int = 32768
    floor: float = 1e-12


@torch.no_grad()
def normalize_signatures(w, h, floor=1e-12):
    """H rows sum to one; W absorbs the inverse rescaling, preserving WH."""
    scale = h.sum(1)
    dead = scale <= floor
    if dead.any():
        # A dead component contributes nothing; keep it dead, with a valid signature.
        w[:, dead] = 0
        h[dead] = 1 / h.shape[1]
        scale = h.sum(1)
    h.div_(scale[:, None])
    w.mul_(scale[None, :])
    return w, h


@torch.no_grad()
def multiplicative_step(x, w, h, q=None, chunk_rows=32768, floor=1e-12):
    """One exact full-data multiplicative alternating sweep, up to FP arithmetic."""
    if q is None:
        hh = h @ h.T
        for lo in range(0, len(x), chunk_rows):
            hi = min(lo + chunk_rows, len(x))
            w[lo:hi].mul_((x[lo:hi] @ h.T) / (w[lo:hi] @ hh).clamp_min(floor))
        h.mul_((w.T @ x) / ((w.T @ w) @ h).clamp_min(floor))
    else:
        for lo in range(0, len(x), chunk_rows):
            hi = min(lo + chunk_rows, len(x))
            b, p, a = x[lo:hi], q[lo:hi], w[lo:hi]
            a.mul_(((p * b) @ h.T) / ((p * (a @ h)) @ h.T).clamp_min(floor))
        numerator, denominator = torch.zeros_like(h), torch.zeros_like(h)
        for lo in range(0, len(x), chunk_rows):
            hi = min(lo + chunk_rows, len(x))
            b, p, a = x[lo:hi], q[lo:hi], w[lo:hi]
            numerator.add_(a.T @ (p * b))
            denominator.add_(a.T @ (p * (a @ h)))
        h.mul_(numerator / denominator.clamp_min(floor))
    normalize_signatures(w, h, floor)
    return w, h


@torch.no_grad()
def objective(x, w, h, q=None, chunk_rows=32768):
    total = 0.
    for lo in range(0, len(x), chunk_rows):
        hi = min(lo + chunk_rows, len(x))
        residual = (x[lo:hi] - w[lo:hi] @ h).square()
        if q is not None:
            residual.mul_(q[lo:hi])
        total += float(residual.sum(dtype=torch.float64))
    return .5 * total / x.numel()


@torch.no_grad()
def warm_start(x, settings, callback=None):
    if x.ndim != 2 or not 1 < settings.rank <= min(x.shape):
        raise ValueError("Invalid input/rank")
    if settings.warm_starts < 1 or settings.warm_iterations < 1:
        raise ValueError("Warm-start budgets must be positive")
    generator = torch.Generator(device=x.device).manual_seed(settings.seed)
    best, histories = None, []
    # Equal initial total activity scale across ranks, no spot subsampling.
    mass = x.sum(1, keepdim=True)
    for start in range(settings.warm_starts):
        w = torch.rand((len(x), settings.rank), device=x.device, dtype=x.dtype, generator=generator).clamp_min(1e-6)
        w.mul_(mass / w.sum(1, keepdim=True))
        h = torch.rand((settings.rank, x.shape[1]), device=x.device, dtype=x.dtype, generator=generator).clamp_min(1e-6)
        h.div_(h.sum(1, keepdim=True))
        for iteration in range(settings.warm_iterations):
            multiplicative_step(x, w, h, chunk_rows=settings.chunk_rows, floor=settings.floor)
        loss = objective(x, w, h, chunk_rows=settings.chunk_rows)
        if not np.isfinite(loss):
            raise FloatingPointError("Nonfinite warm-start loss")
        histories.append(dict(start=start, loss=loss))
        if best is None or loss < best[0]:
            best = (loss, w.clone(), h.clone(), start)
        if callback:
            callback(dict(stage="warm_start", **histories[-1]))
    return best[1], best[2], dict(starts=histories, best_start=best[3])


@torch.no_grad()
def fit(x, settings, q=None, initial=None, callback=None):
    if settings.max_iterations < 1 or settings.check_every < 1 or settings.patience < 1:
        raise ValueError("Invalid iteration settings")
    if (x < 0).any() or not torch.isfinite(x).all():
        raise ValueError("Expected finite nonnegative target")
    if q is not None and (q.shape != x.shape or not torch.isfinite(q).all() or (q <= 0).any()):
        raise ValueError("Expected positive finite precision weights")
    t0 = time.monotonic()
    if initial is None:
        w, h, init_info = warm_start(x, settings, callback)
    else:
        w, h = initial[0].clone(), initial[1].clone()
        init_info = dict(shared_initialization=True)
    previous = objective(x, w, h, q, settings.chunk_rows)
    history = [dict(iteration=0, loss=previous, seconds=0.)]
    stable, converged = 0, False
    for iteration in range(1, settings.max_iterations + 1):
        multiplicative_step(x, w, h, q, settings.chunk_rows, settings.floor)
        if iteration % settings.check_every == 0 or iteration == settings.max_iterations:
            loss = objective(x, w, h, q, settings.chunk_rows)
            change = (previous - loss) / max(abs(previous), 1e-30)
            if not np.isfinite(loss) or change < -1e-5:
                raise FloatingPointError(f"Invalid/nonmonotone loss: {previous} -> {loss}")
            stable = stable + 1 if abs(change) <= settings.tolerance else 0
            row = dict(iteration=iteration, loss=loss, relative_decrease=change, seconds=time.monotonic()-t0)
            history.append(row)
            if callback:
                callback(dict(stage="fitting", **row))
            if stable >= settings.patience:
                converged = True
                break
            previous = loss
    return w, h, dict(settings=asdict(settings), initialization=init_info, history=history,
                      converged=converged, iterations=iteration, elapsed_seconds=time.monotonic()-t0)


@torch.no_grad()
def precision_weights(variance, chunk_rows=32768):
    """Fixed surrogate weights: 1/(V + 0.1 mean_i(V) + 1e-6), mean norm, clip."""
    if (variance < 0).any() or not torch.isfinite(variance).all():
        raise ValueError("Invalid posterior variance")
    floor = .1 * variance.mean(0) + 1e-6
    for lo in range(0, len(variance), chunk_rows):
        hi = min(lo + chunk_rows, len(variance))
        variance[lo:hi].add_(floor).reciprocal_()
    normalizer = float(variance.mean(dtype=torch.float64))
    variance.div_(normalizer).clamp_(.01, 100.)
    return variance, dict(floor_min=float(floor.min()), floor_max=float(floor.max()),
                          precision_normalizer=normalizer, clip=[.01, 100.])
