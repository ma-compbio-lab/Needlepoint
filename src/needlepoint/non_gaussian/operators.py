"""Fixed spatial priors and weighted precision actions for coupled VI.

The dense reference and structured SKI/VNNGP backend have the same interface.
Kernel fitting rebuilds these immutable numerical actions; it does not attempt
autodifferentiation through the frozen Gaussian experiment code.
"""
import numpy as np
import torch

from .data import fingerprint


def _tensor(value, *, device='cpu'):
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float64, device=device).clone()
    return torch.as_tensor(np.asarray(value).copy(), dtype=torch.float64, device=device).clone()


class DenseEvaluation:
    def __init__(self, matrix, *, global_dimension=0, coordinates=None, device='cpu'):
        self.matrix = _tensor(matrix, device=device)
        if self.matrix.ndim != 2 or not torch.isfinite(self.matrix).all():
            raise ValueError('Evaluation matrix must be finite and two dimensional')
        self.size, self.dimension = self.matrix.shape
        self.global_dimension = int(global_dimension)
        if not 0 <= self.global_dimension <= self.dimension:
            raise ValueError('Invalid global_dimension')
        self.device = self.matrix.device
        self.dtype = self.matrix.dtype
        self.coordinates = None if coordinates is None else np.asarray(coordinates, float).copy()
        if self.coordinates is not None and (self.coordinates.shape != (self.size, 2)
                                             or not np.isfinite(self.coordinates).all()):
            raise ValueError('coordinates must match evaluation rows')

    def matmul(self, rhs, rows=None):
        matrix = self.matrix if rows is None else self.matrix[rows]
        return matrix @ rhs

    def t_matmul(self, rhs, rows=None):
        matrix = self.matrix if rows is None else self.matrix[rows]
        return matrix.T @ rhs

    def matmul_parts(self, rhs, rows=None):
        matrix = self.matrix if rows is None else self.matrix[rows]
        m = self.global_dimension
        return matrix[:, :m] @ rhs[:m], matrix[:, m:] @ rhs[m:]

    def anchor_local(self):
        if self.global_dimension == self.dimension:
            return torch.full((self.size,), -1, dtype=torch.long, device=self.device)
        return self.matrix[:, self.global_dimension:].abs().argmax(dim=1)

    def fingerprint(self):
        arrays = [self.matrix.cpu().numpy()]
        if self.coordinates is not None:
            arrays.append(self.coordinates)
        return fingerprint(*arrays, metadata=dict(global_dimension=self.global_dimension))


class DensePrior:
    """Explicit SPD prior precision for small mathematical reference problems."""
    def __init__(self, precision, *, global_dimension=0, device='cpu'):
        self.precision = _tensor(precision, device=device)
        q = self.precision
        if q.ndim != 2 or q.shape[0] != q.shape[1] or len(q) == 0:
            raise ValueError('Prior precision must be a nonempty square matrix')
        if not torch.isfinite(q).all() or not torch.allclose(q, q.T, rtol=1e-12, atol=1e-12):
            raise ValueError('Prior precision must be finite and symmetric')
        self.cholesky = torch.linalg.cholesky(q)
        self.dimension = len(q)
        self.global_dimension = int(global_dimension)
        if not 0 <= self.global_dimension <= self.dimension:
            raise ValueError('Invalid global_dimension')
        self.device, self.dtype = q.device, q.dtype
        self.logdet = 2 * self.cholesky.diagonal().log().sum()

    def precision_matmul(self, rhs):
        return self.precision @ rhs

    def precision_noise(self, white):
        return self.cholesky @ white

    def unwhiten(self, rhs):
        vector = rhs.ndim == 1
        result = torch.linalg.solve_triangular(self.cholesky.T, rhs.reshape(self.dimension, -1), upper=True)
        return result[:, 0] if vector else result

    def unwhiten_t(self, rhs):
        vector = rhs.ndim == 1
        result = torch.linalg.solve_triangular(self.cholesky, rhs.reshape(self.dimension, -1), upper=False)
        return result[:, 0] if vector else result

    def inverse_matmul(self, rhs):
        return self.unwhiten(self.unwhiten_t(rhs))

    def fingerprint(self):
        return fingerprint(self.precision.cpu().numpy(), metadata=dict(global_dimension=self.global_dimension))


class SpatialEvaluation:
    """One deterministic field at native, event, quadrature and output sites.

    Local interpolation has no independent per-query residual. Consequently
    repeated and nearby evaluations share the same latent random variables.
    """
    def __init__(self, prior, coordinates, global_interpolation, local_interpolation):
        self.prior = prior
        self.coordinates = np.asarray(coordinates, float).copy()
        self.global_interpolation = global_interpolation
        self.local_interpolation = local_interpolation
        self.dimension = prior.dimension
        self.global_dimension = prior.global_dimension
        self.size = len(self.coordinates)
        self.dtype, self.device = prior.dtype, prior.device

    def matmul_parts(self, rhs, rows=None):
        m = self.global_dimension
        global_value = self.global_interpolation.matmul(self.prior.global_prior.sqrt_matmul(rhs[:m]), rows)
        local_value = self.local_interpolation.matmul(rhs[m:], rows)
        return global_value, local_value

    def matmul(self, rhs, rows=None):
        g, l = self.matmul_parts(rhs, rows)
        return g + l

    def t_matmul(self, rhs, rows=None):
        g = self.prior.global_prior.sqrt_t_matmul(self.global_interpolation.t_matmul(rhs, rows))
        l = self.local_interpolation.t_matmul(rhs, rows)
        return torch.cat((g, l))

    def anchor_local(self):
        weights = self.local_interpolation.weights
        return self.local_interpolation.indices.gather(1, weights.abs().argmax(1)[:, None]).flatten()

    def fingerprint(self):
        return fingerprint(self.coordinates, self.global_interpolation.indices.cpu().numpy(),
            self.global_interpolation.weights.cpu().numpy(), self.local_interpolation.indices.cpu().numpy(),
            self.local_interpolation.weights.cpu().numpy(),
            metadata=dict(convention='deterministic-cubic-ski-linear-local-v1'))


class StructuredSpatialPrior:
    """Independent global SKI and local VNNGP priors, coupled after conditioning."""
    def __init__(self, global_prior, local_prior):
        if global_prior.device != local_prior.device or global_prior.dtype != torch.float64:
            raise ValueError('Coupled VI requires matching float64 prior operators')
        if local_prior.dtype != torch.float64:
            raise ValueError('Coupled VI requires float64 local operators')
        self.global_prior, self.local_prior = global_prior, local_prior
        self.global_dimension = global_prior.size
        self.dimension = global_prior.size + local_prior.size
        self.device, self.dtype = global_prior.device, global_prior.dtype
        self.logdet = (-2 * local_prior.initial_cholesky.diagonal().log().sum()
                      - local_prior.conditional_variance.log().sum())
        self._triangulation = None
        self._tree = None

    @classmethod
    def from_frozen(cls, operators):
        """Reuse declared priors only; Gaussian likelihood/means are not imported."""
        return cls(operators.global_prior, operators.local_prior)

    def precision_matmul(self, rhs):
        m = self.global_dimension
        return torch.cat((rhs[:m], self.local_prior.precision_matmul(rhs[m:])))

    def precision_noise(self, white):
        m = self.global_dimension
        return torch.cat((white[:m], self.local_prior.whiten_local_t(white[m:])))

    def unwhiten(self, rhs):
        m = self.global_dimension
        local = self.local_prior
        value = rhs.reshape(self.dimension, -1)
        output = value.clone()
        out = output[m:]
        out[:local.k] = local.initial_cholesky @ value[m:m + local.k]
        for i in range(local.k, local.size):
            j = i - local.k
            out[i] = (value[m + i] / local.inverse_sd[j]
                      + (local.coefficients[j, :, None] * out[local.indices[j]]).sum(0))
        return output[:, 0] if rhs.ndim == 1 else output

    def unwhiten_t(self, rhs):
        m = self.global_dimension
        local = self.local_prior
        value = rhs.reshape(self.dimension, -1).clone()
        out = value.clone()
        for i in range(local.size - 1, local.k - 1, -1):
            j = i - local.k
            out[m + i] = value[m + i] / local.inverse_sd[j]
            value[m:].index_add_(0, local.indices[j], local.coefficients[j, :, None] * value[m + i].clone())
        out[m:m + local.k] = local.initial_cholesky.T @ value[m:m + local.k]
        return out[:, 0] if rhs.ndim == 1 else out

    def inverse_matmul(self, rhs):
        m = self.global_dimension
        return torch.cat((rhs[:m], self.local_prior.prior_solve(rhs[m:])))

    def evaluation(self, coordinates):
        from scipy.spatial import cKDTree, Delaunay
        from gpytorch.utils.interpolation import Interpolation
        from ..gp_prior_operators import SparseInterpolation
        xy = np.asarray(coordinates, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError('Evaluation coordinates must be finite (n, 2)')
        support = self.local_prior.coordinates.cpu().numpy()
        if self._tree is None:
            self._tree = cKDTree(support)
        distance, indices = self._tree.query(xy)
        local_indices = np.repeat(indices[:, None], 3, axis=1)
        local_weights = np.zeros((len(xy), 3)); local_weights[:, 0] = 1
        off = distance != 0
        if off.any():
            if self._triangulation is None:
                self._triangulation = Delaunay(support)
            tri = self._triangulation
            simplex = tri.find_simplex(xy[off])
            if (simplex < 0).any():
                raise ValueError('Local interpolation requires query points inside the support mesh')
            transform = tri.transform[simplex]
            barycentric = np.einsum('nij,nj->ni', transform[:, :2], xy[off] - transform[:, 2])
            local_indices[off] = tri.simplices[simplex]
            local_weights[off] = np.column_stack((barycentric, 1 - barycentric.sum(1)))
        with torch.no_grad():
            if len(xy):
                indices_g, weights_g = Interpolation().interpolate(
                    list(self.global_prior.axes), _tensor(xy, device=self.device))
            else:
                indices_g = torch.empty((0, 16), dtype=torch.long, device=self.device)
                weights_g = torch.empty((0, 16), dtype=self.dtype, device=self.device)
        global_op = SparseInterpolation(indices_g, weights_g, self.global_dimension,
                                         dtype=self.dtype, device=self.device)
        local_op = SparseInterpolation(local_indices, local_weights, len(support),
                                        dtype=self.dtype, device=self.device)
        return SpatialEvaluation(self, xy, global_op, local_op)

    def fingerprint(self):
        g, l = self.global_prior, self.local_prior
        return fingerprint(*(a.cpu().numpy() for a in [*g.axes, g.ux, g.uy, g.eigenvalues,
            l.coordinates, l.indices, l.initial_cholesky, l.coefficients, l.conditional_variance]),
            metadata=dict(prior='independent-ski-vnngp', query='deterministic-linear-local'))


def build_spatial_prior(coordinates, *, global_lengthscale, local_lengthscale,
                        global_scale=1.0, local_scale=1.0, grid_shape=(16, 16),
                        local_neighbors=16, global_jitter=1e-3, local_jitter=1e-3,
                        axis_stabilization=1e-10, device='cpu'):
    """Construct explicit fixed kernels, Morton DAG, cubic SKI and local mesh.

    Coordinates are micrometers; no measurement normalization is performed.
    Every supplied support row is retained. Numerical factors are rebuilt when
    kernel parameters change, while ordering and geometry remain deterministic.
    """
    from ..gp_prior_operators import FrozenGridPrior, FrozenLocalPrior
    from ..gpytorch_global_local import morton_order, build_predecessor_neighbor_dag
    xy = np.asarray(coordinates, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
        raise ValueError('Supply at least two finite spatial support coordinates')
    if len(np.unique(xy, axis=0)) != len(xy):
        raise ValueError('Spatial support coordinates must be unique; events may repeat')
    shape = tuple(int(v) for v in grid_shape)
    if len(shape) != 2 or min(shape) < 4 or tuple(grid_shape) != shape:
        raise ValueError('grid_shape must contain two integers at least 4')
    axes = []
    for dimension, count in enumerate(shape):
        low, high = xy[:, dimension].min(), xy[:, dimension].max()
        width = max(high - low, 1.0)
        step = width / (count - 3)
        axes.append(low - step + np.arange(count) * step)
    order = morton_order(xy)
    k = min(int(local_neighbors), len(xy) - 1)
    if k < 1 or int(local_neighbors) != local_neighbors:
        raise ValueError('local_neighbors must be a positive integer')
    dag = build_predecessor_neighbor_dag(xy[order], k)
    g = FrozenGridPrior(axes, global_lengthscale, global_scale, global_jitter,
        axis_stabilization=axis_stabilization, dtype=torch.float64, device=device)
    l = FrozenLocalPrior(xy[order], dag, local_lengthscale, local_scale, local_jitter,
                         dtype=torch.float64, device=device)
    return StructuredSpatialPrior(g, l)


class WeightedPrecision:
    """P=Q0+A.T diag(information) A, frozen for each solve or sample batch."""
    def __init__(self, prior, evaluation, information):
        self.prior, self.evaluation = prior, evaluation
        self.information = information.detach().clone()
        self.dimension = prior.dimension
        self.device, self.dtype = prior.device, prior.dtype
        if (evaluation.dimension != self.dimension or evaluation.device != self.device
                or evaluation.global_dimension != prior.global_dimension
                or self.information.shape != (evaluation.size,)
                or not torch.isfinite(self.information).all() or (self.information < 0).any()):
            raise ValueError('Information must be a finite nonnegative vector on the evaluation support')

    def matmul(self, rhs):
        value = self.evaluation.matmul(rhs)
        weights = self.information if rhs.ndim == 1 else self.information[:, None]
        return self.prior.precision_matmul(rhs) + self.evaluation.t_matmul(weights * value)

    def forcing(self, generator, count):
        white = torch.randn((self.dimension, count), dtype=self.dtype, device=self.device, generator=generator)
        observation = torch.randn((self.evaluation.size, count), dtype=self.dtype,
                                  device=self.device, generator=generator)
        return self.prior.precision_noise(white) + self.evaluation.t_matmul(
            self.information.sqrt()[:, None] * observation)

    def whitened_matmul(self, rhs):
        return self.prior.unwhiten_t(self.matmul(self.prior.unwhiten(rhs)))

    def preconditioner_diagonal(self):
        """Fixed positive Jacobi scaling; it never substitutes for the target."""
        if isinstance(self.prior, DensePrior):
            diagonal = self.prior.precision.diagonal().clone()
        else:
            diagonal = torch.cat((torch.ones(self.prior.global_dimension,
                dtype=self.dtype, device=self.device), self.prior.local_prior.precision_diagonal.clone()))
        if isinstance(self.evaluation, DenseEvaluation):
            diagonal += (self.evaluation.matrix.square() * self.information[:, None]).sum(0)
        else:
            m = self.prior.global_dimension
            diagonal[:m] += self.information.sum() * self.prior.global_prior.diagonal / m
            local = self.evaluation.local_interpolation
            diagonal[m:].index_add_(0, local.indices.flatten(),
                (self.information[:, None] * local.weights.square()).flatten())
        if not torch.isfinite(diagonal).all() or (diagonal <= 0).any():
            raise FloatingPointError('Invalid fixed precision preconditioner')
        return diagonal
