"""Exact structured RBF-grid KL with a degeneracy-safe analytic backward.

Float64 spectral work, no full inducing covariance, no persistent factor cache.
The backward differentiates the matrix function, not individual eigenvectors.
Consequently equal/repeated axis eigenvalues need no arbitrary gap threshold.
"""
from __future__ import annotations

import torch

FACTOR_STABILIZATION = 1e-10


def grid_prior_factors(grid, lengthscale, n_features):
    grid = grid.to(torch.float64)
    length = lengthscale.reshape(n_features, -1).to(torch.float64)
    if length.shape[-1] == 1:
        length = length.expand(n_features, 2)
    if length.shape[-1] != 2 or grid.shape[-1] != 2:
        raise ValueError("structured grid KL requires a two-dimensional RBF grid")
    identity = torch.eye(len(grid), device=grid.device, dtype=grid.dtype)
    factors = []
    for axis in range(2):
        squared = (grid[:, axis, None] - grid[None, :, axis]).square()
        factors.append(torch.exp(-0.5*squared/length[:, axis, None, None].square())
                       + FACTOR_STABILIZATION*identity)
    return factors


class _PriorTerms(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mean, diagonal, root, prior_mean, kx, ky, scale, jitter):
        f, m = mean.shape
        q = kx.shape[-1]
        lx, ux = torch.linalg.eigh(kx)
        ly, uy = torch.linalg.eigh(ky)
        eigen = scale[:, None, None]*ly[:, :, None]*lx[:, None, :] + jitter
        if torch.any(eigen <= 0):
            raise FloatingPointError("nonpositive structured prior spectrum")
        inv = eigen.reciprocal()
        centered = (mean-prior_mean[:, None]).reshape(f, q, q)
        projected_mean = uy.transpose(-1, -2) @ centered @ ux
        root_grid = root.reshape(f, q, q, -1)
        projected_root = torch.einsum('fyi,fyxr,fxj->fijr', uy, root_grid, ux)
        inverse_diagonal = uy.square() @ inv @ ux.square().transpose(-1, -2)
        d = diagonal.reshape(f, q, q)
        result = 0.5*((d*inverse_diagonal).sum((-2,-1))
            +(projected_mean.square()*inv).sum((-2,-1))
            +(projected_root.square()*inv.unsqueeze(-1)).sum((-3,-2,-1))
            -m + eigen.log().sum((-2,-1)))
        ctx.save_for_backward(diagonal, root, lx, ux, ly, uy, inv, scale,
                              projected_mean, projected_root, inverse_diagonal, kx, ky)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        (diagonal, root, lx, ux, ly, uy, inv, scale, pm, pb, diag_inv, kx, ky
         ) = ctx.saved_tensors
        f, m = diagonal.shape
        q = kx.shape[-1]
        solved_mean = uy @ (pm*inv) @ ux.transpose(-1,-2)
        solved_root = torch.einsum('fyi,fijr,fxj->fyxr', uy, pb*inv.unsqueeze(-1), ux)
        d = diagonal.reshape(f, q, q)

        # In each y eigenmode, transform the x-diagonal of D. This contracts
        # tr(K^-1 D K^-1 dK) in O(q^4), never allocating q^2 by q^2.
        dx = torch.einsum('fyi,fyx->fix', uy.square(), d)
        tx = torch.einsum('fxj,fix,fxk->fijk', ux, dx, ux)
        tx = (tx * inv.unsqueeze(-1)*inv.unsqueeze(-2)*ly[:, :, None, None]).sum(1)
        dy = torch.einsum('fxj,fyx->fjy', ux.square(), d)
        ty = torch.einsum('fyi,fjy,fyk->fjik', uy, dy, uy)
        inv_t = inv.transpose(-1, -2)
        ty = (ty*inv_t.unsqueeze(-1)*inv_t.unsqueeze(-2)*lx[:, :, None, None]).sum(1)
        px = ux @ torch.diag_embed((ly[:, :, None]*inv).sum(1)) @ ux.transpose(-1,-2)
        py = uy @ torch.diag_embed((lx[:, None, :]*inv).sum(2)) @ uy.transpose(-1,-2)
        ax = solved_mean.transpose(-1,-2) @ ky @ solved_mean
        ay = solved_mean @ kx @ solved_mean.transpose(-1,-2)
        if root.shape[-1]:
            ax = ax + torch.einsum('fyxr,fyv,fvzr->fxz', solved_root, ky, solved_root)
            ay = ay + torch.einsum('fyxr,fxz,fvzr->fyv', solved_root, kx, solved_root)
        gx = 0.5*scale[:,None,None]*(px - ux@tx@ux.transpose(-1,-2) - ax)
        gy = 0.5*scale[:,None,None]*(py - uy@ty@uy.transpose(-1,-2) - ay)
        projected_d = uy.square().transpose(-1,-2) @ d @ ux.square()
        diag_t = projected_d + pm.square() + pb.square().sum(-1)
        gs = 0.5*((ly[:,:,None]*lx[:,None,:])*(inv-diag_t*inv.square())).sum((-2,-1))
        w = grad_output
        return (solved_mean.reshape(f,m)*w[:,None],
                0.5*diag_inv.reshape(f,m)*w[:,None],
                solved_root.reshape_as(root)*w[:,None,None],
                -solved_mean.sum((-2,-1))*w,
                gx*w[:,None,None], gy*w[:,None,None], gs*w, None)


def diagonal_root_logdet(diagonal, root):
    """Matrix determinant lemma, including the rank-zero limit."""
    result = diagonal.log().sum(-1)
    rank = root.shape[-1]
    if rank:
        gram = root.transpose(-1,-2) @ (root/diagonal.unsqueeze(-1))
        identity = torch.eye(rank, dtype=root.dtype, device=root.device)
        factor = torch.linalg.cholesky(gram+identity)
        result = result + 2*factor.diagonal(dim1=-2,dim2=-1).log().sum(-1)
    return result


def structured_gaussian_grid_kl(*, variational_mean, diagonal, root, grid,
                                prior_mean, lengthscale, outputscale, jitter=1e-3):
    f, m = variational_mean.shape
    if m != len(grid)**2 or root.shape[:2] != (f,m):
        raise ValueError("covariance does not match grid dimensions")
    if jitter <= 0:
        raise ValueError("prior jitter must be positive")
    kx, ky = grid_prior_factors(grid, lengthscale, f)
    args = [t.to(torch.float64) for t in
            (variational_mean,diagonal,root,prior_mean.reshape(f),outputscale.reshape(f))]
    mean,d,b,m0,scale = args
    prior = _PriorTerms.apply(mean,d,b,m0,kx,ky,scale,float(jitter))
    return (prior-0.5*diagonal_root_logdet(d,b)).to(variational_mean.dtype)
