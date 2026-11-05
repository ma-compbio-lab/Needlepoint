"""Opt-in inducing-value diagonal-plus-low-rank Gaussian covariance.

This module has optional Torch imports; the package root does not import it.
"""
from __future__ import annotations

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import _VariationalDistribution
from linear_operator.operators import (DiagLinearOperator, LowRankRootLinearOperator,
                                       InterpolatedLinearOperator)
from linear_operator.utils.interpolation import left_interp

COVARIANCE_VERSION = "inducing-values-diag-plus-root-v1"


def inverse_softplus(value):
    return value + torch.log(-torch.expm1(-value))


class LowRankDiagonalVariationalDistribution(_VariationalDistribution):
    """q(u)=N(m,diag(d)+BBᵀ), without an M by M parameter or intermediate."""

    def __init__(self, num_inducing_points, rank, batch_shape=torch.Size(),
                 initial_fraction=0.01, init_seed=0, variance_floor=1e-10,
                 mean_init_std=1e-3):
        if not 0 <= rank <= num_inducing_points:
            raise ValueError("rank must be between zero and the inducing dimension")
        if not 0 < initial_fraction < 1 or variance_floor <= 0:
            raise ValueError("initial_fraction must be in (0,1); floor must be positive")
        super().__init__(num_inducing_points, batch_shape, mean_init_std)
        self.rank = int(rank)
        self.initial_fraction = float(initial_fraction)
        self.init_seed = int(init_seed)
        self.variance_floor = float(variance_floor)
        self.register_parameter("variational_mean", torch.nn.Parameter(
            torch.zeros(*batch_shape, num_inducing_points)))
        self.register_parameter("raw_diagonal", torch.nn.Parameter(
            torch.ones(*batch_shape, num_inducing_points)))
        self.register_parameter("covariance_root", torch.nn.Parameter(
            torch.zeros(*batch_shape, num_inducing_points, rank)))

    @property
    def diagonal_variance(self):
        # Squared scale matches the legacy mean-field optimizer geometry while
        # an explicit floor keeps d strictly positive at a scale crossing zero.
        return self.raw_diagonal.square() + self.variance_floor

    @property
    def marginal_variance(self):
        return self.diagonal_variance + self.covariance_root.square().sum(-1)

    def forward(self):
        covariance = DiagLinearOperator(self.diagonal_variance)
        if self.rank:
            covariance = covariance + LowRankRootLinearOperator(self.covariance_root)
        return MultivariateNormal(self.variational_mean, covariance)

    def initialize_variational_distribution(self, prior_dist):
        # Match the legacy inducing marginal initialization and mean RNG stream.
        with torch.no_grad():
            self.variational_mean.copy_(prior_dist.mean)
            self.variational_mean.add_(torch.randn_like(prior_dist.mean), alpha=self.mean_init_std)
            v0 = prior_dist.variance
            fraction = self.initial_fraction if self.rank else 0.0
            self.raw_diagonal.copy_(
                ((1-fraction)*v0-self.variance_floor).clamp_min(self.variance_floor).sqrt())
            if self.rank:
                generator = torch.Generator(device=self.covariance_root.device)
                generator.manual_seed(self.init_seed)
                directions = torch.randn(self.covariance_root.shape, generator=generator,
                    device=self.covariance_root.device, dtype=self.covariance_root.dtype)
                directions /= directions.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                self.covariance_root.copy_(directions * (fraction*v0).sqrt().unsqueeze(-1))
            self.initialization_statistics = dict(seed=self.init_seed,fraction=fraction,
                variance_floor=self.variance_floor,
                inducing_marginal_min=float(self.marginal_variance.min()),
                inducing_marginal_mean=float(self.marginal_variance.mean()),
                inducing_marginal_max=float(self.marginal_variance.max()),
                diagonal_min=float(self.diagonal_variance.min()),
                root_norm=float(self.covariance_root.norm()))


def interpolation_marginals(mean, diagonal, root, indices, weights):
    """Exact signed sparse Wm and diag(W(diag(d)+BBᵀ)Wᵀ).

    Combining duplicate indices is essential for the diagonal contribution.
    The c by c equality tensor is bounded by the cubic stencil size, not M.
    """
    predictive_mean = left_interp(indices, weights, mean.unsqueeze(-1)).squeeze(-1)
    selected = diagonal[..., indices] if indices.ndim == 2 else torch.gather(
        diagonal.unsqueeze(-2).expand(*indices.shape[:-1], diagonal.shape[-1]), -1, indices)
    equal = indices.unsqueeze(-1) == indices.unsqueeze(-2)
    combined = (equal * weights.unsqueeze(-2)).sum(-1)
    variance = (selected * weights * combined).sum(-1)
    if root.shape[-1]:
        projected_root = left_interp(indices, weights, root)
        variance = variance + projected_root.square().sum(-1)
    return predictive_mean, variance


class ExactDiagonalInterpolationOperator(InterpolatedLinearOperator):
    """Coherent W D Wᵀ operator with exact, inexpensive marginal contraction."""

    def _diagonal(self):
        indices, weights = self.left_interp_indices, self.left_interp_values
        d = self.base_linear_op.diagonal()
        # Strategy supplies feature-expanded stencils, which can be identical
        # across features but should not be assumed so by this operator.
        selected = torch.gather(d.unsqueeze(-2).expand(*indices.shape[:-1], d.shape[-1]),
                                -1, indices)
        equal = indices.unsqueeze(-1) == indices.unsqueeze(-2)
        combined = (equal*weights.unsqueeze(-2)).sum(-1)
        return (selected*weights*combined).sum(-1)

    def to_dense(self):
        if self.shape[-1] > 1024 or self.shape[-2] > 1024:
            raise RuntimeError("pilot guard: do not materialize large SKI query covariance")
        return super().to_dense()
