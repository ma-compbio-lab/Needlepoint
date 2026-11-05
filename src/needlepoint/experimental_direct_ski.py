"""Experimental dense-grid reference for full-observation SKI screens.

The determinant lemma and Woodbury identity reduce N-observation quantities to
M-grid systems, without approximating the existing SKI covariance. Dense O(M^2)
storage and O(M^3) LU are intentional reference costs, not production defaults.
"""
from __future__ import annotations


def direct_ski_reference(kernel_gram, kernel_projection, projection, squared_norm,
                         *, n_observations, fraction, diagonal):
    """Return the exact-SKI logdet and inverse quadratics up to FP64 LU error.

    Inputs are K @ (W.T @ W), K @ (W.T @ y), W.T @ y, and y.T @ y. Grid
    columns with identically zero W may be omitted from *all* inputs together;
    no observation row is ever omitted. The nonsymmetric grid system is solved
    by pivoted LU, never by ordinary Euclidean CG.
    """
    import math
    import torch

    tensors = [kernel_gram, kernel_projection, projection, squared_norm]
    if any(x.dtype != torch.float64 for x in tensors):
        raise ValueError('Direct SKI reference requires float64 inputs')
    if fraction <= 0 or diagonal <= 0 or n_observations < 1:
        raise ValueError('Positive covariance parameters and observation count required')
    if any(not torch.isfinite(x).all() for x in tensors):
        raise ValueError('Nonfinite direct SKI inputs')
    m = kernel_gram.shape[0]
    matrix = (fraction/diagonal)*kernel_gram
    matrix.diagonal().add_(1.)
    lu, pivots, info = torch.linalg.lu_factor_ex(matrix)
    if int(info) != 0:
        raise FloatingPointError(f'Grid LU factorization failed: info={int(info)}')
    lu_diagonal = lu.diagonal()
    swaps = (pivots != torch.arange(1,m+1,device=pivots.device)).sum()
    determinant_sign = lu_diagonal.sign().prod() * (1-2*(swaps % 2))
    if float(determinant_sign) != 1.:
        raise FloatingPointError('Nonpositive SKI covariance determinant')
    logdet = n_observations*math.log(diagonal)+lu_diagonal.abs().log().sum()
    rhs = (fraction/diagonal)*kernel_projection
    zero_target = squared_norm == 0
    if torch.any(zero_target):
        if torch.any(projection[:, zero_target] != 0):
            raise ValueError("Zero target norm requires an exactly zero projection")
        # FFT-backed kernel products can leave tiny roundoff in a zero RHS
        # inside a large multi-RHS batch. Its analytic value is exactly zero.
        rhs[:, zero_target] = 0
    correction = torch.linalg.lu_solve(lu,pivots,rhs)
    correction[:, zero_target] = 0
    grid_residual = (matrix@correction-rhs).norm(dim=0)/rhs.norm(dim=0).clamp_min(1e-30)
    quad = (squared_norm-(projection*correction).sum(0))/diagonal
    if not torch.isfinite(quad).all() or not torch.isfinite(logdet):
        raise FloatingPointError('Nonfinite direct SKI result')
    return dict(logdet=logdet, inv_quad=quad, grid_solution=-correction/diagonal,
                rhs_coefficient=1./diagonal, grid_relative_residual=grid_residual,
                determinant_sign=float(determinant_sign), active_grid_size=m)


def direct_observation_residual(kernel_matmul, gram_matmul, projection, squared_norm,
                                grid_solution, *, fraction, diagonal):
    """Recompute the original observation residual using exact sufficient stats.

    For a direct solution x = W u + y/d, Cx-y = W r, where
    r = rho K (G u + h/d) + d u. Thus ||Cx-y||^2 = r.T G r.
    For a nonzero target this is relative to ||y||. For an exactly zero target,
    return the absolute residual: a relative residual is undefined there.
    This is not permission to ignore a bad zero-target solution.
    Unlike a CG recurrence, this evaluates the final solution in the original
    operator. It avoids constructing N-vectors and never discards an observation.
    """
    import torch

    residual = (fraction*kernel_matmul(gram_matmul(grid_solution)+projection/diagonal)
                + diagonal*grid_solution)
    norm2 = (residual*gram_matmul(residual)).sum(0)
    if not torch.isfinite(norm2).all() or torch.any(norm2 < 0):
        raise FloatingPointError('Invalid direct observation residual norm')
    target_norm = squared_norm.sqrt()
    denominator = torch.where(target_norm > 0, target_norm, torch.ones_like(target_norm))
    return norm2.sqrt()/denominator
