"""Numerical parameterizations of the unchanged Gaussian variational family."""
import torch
from torch import nn
from torch.nn.utils import parametrize


class PositiveVariationalScale(nn.Module):
    """Store rho, expose sd=softplus(rho), without an artificial variance floor.

    In raw standard-deviation coordinates, the Gaussian entropy derivative is
    proportional to 1/sd. It is bounded as sd tends to zero in rho coordinates.
    The right inverse preserves the initial distribution, including legacy
    signed scales whose square defines the same covariance.
    """

    def forward(self, value):
        return torch.nn.functional.softplus(value)

    def right_inverse(self, value):
        positive = value.abs()
        if torch.any(positive == 0):
            raise ValueError("A zero variational scale cannot define a Gaussian density")
        return positive + torch.log(-torch.expm1(-positive))


def parameterize_variational_scale(distribution):
    parametrize.register_parametrization(distribution, "_variational_stddev", PositiveVariationalScale())
