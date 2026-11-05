"""Frozen SKI/VNNGP prior actions for opt-in joint Gaussian conditioning.

No full inducing or training covariance is constructed. The global full-spectrum
symmetric root costs O(M**1.5) per RHS on a square grid; local factor construction
costs O(N k**3), storage O(N k), and multiplication O(N k) per RHS. Sparse local
prior solves traverse DAG dependencies and may be inefficient on GPUs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

LOCAL_QUERY_CONVENTION = "unique_queries_independent_withheld_conditional_residual_v1"


def _tensor(value, *, dtype, device):
    # clone makes the operator independent of caller mutation and detaches graphs.
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=dtype, device=device).clone()
    return torch.tensor(value, dtype=dtype, device=device)


def _columns(value, size, *, dtype, device):
    value = (value.detach().to(dtype=dtype, device=device) if isinstance(value, torch.Tensor)
             else torch.tensor(value, dtype=dtype, device=device))
    vector = value.ndim == 1
    if vector:
        value = value[:, None]
    if value.ndim != 2 or value.shape[0] != size:
        raise ValueError(f"RHS must have shape ({size},) or ({size}, n_rhs)")
    if not torch.isfinite(value).all():
        raise ValueError("RHS must be finite")
    return value, vector


def _restore(value, vector):
    return value[:, 0] if vector else value


def _positive(name, value, *, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


class FrozenGridPrior:
    """Exact cached tensor-product RBF prior, x-fast inducing ordering.

    ``grid_axes=(x,y)`` may have unequal sizes and spacings. All axis eigenvalues
    participate, including stabilized small modes. The root is symmetric.
    """

    @torch.no_grad()
    def __init__(self, grid_axes, lengthscale, outputscale, jitter, *,
                 axis_stabilization=1e-10, dtype=torch.float64, device="cpu"):
        self.dtype, self.device = dtype, torch.device(device)
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("frozen operators require float32 or float64")
        if len(grid_axes) != 2:
            raise ValueError("two grid axes are required")
        self.axes = tuple(_tensor(a, dtype=dtype, device=device) for a in grid_axes)
        if any(a.ndim != 1 or len(a) < 2 or not torch.isfinite(a).all()
               or not torch.all(a[1:] > a[:-1]) for a in self.axes):
            raise ValueError("grid axes must be finite, strictly increasing vectors")
        length = _tensor(lengthscale, dtype=dtype, device=device).reshape(-1)
        if length.numel() == 1:
            length = length.expand(2)
        if length.numel() != 2 or not torch.isfinite(length).all() or torch.any(length <= 0):
            raise ValueError("global lengthscale must have one or two positive values")
        self.outputscale = _positive("global outputscale", outputscale)
        self.jitter = _positive("global jitter", jitter)
        self.axis_stabilization = _positive("axis stabilization", axis_stabilization, allow_zero=True)
        self.nx, self.ny = len(self.axes[0]), len(self.axes[1])
        self.size = self.nx * self.ny
        eigenvalues, eigenvectors = [], []
        for axis, ell in zip(self.axes, length):
            kernel = torch.exp(-0.5 * ((axis[:, None] - axis[None, :]) / ell).square())
            kernel.diagonal().add_(self.axis_stabilization)
            vals, vecs = torch.linalg.eigh(kernel)
            eigenvalues.append(vals)
            eigenvectors.append(vecs)
        self.ux, self.uy = eigenvectors
        self.eigenvalues = (self.outputscale * eigenvalues[1][:, None] * eigenvalues[0][None, :]
                            + self.jitter)
        if not torch.isfinite(self.eigenvalues).all() or torch.any(self.eigenvalues <= 0):
            raise FloatingPointError("frozen global prior has nonpositive spectrum; jitter was not changed")
        self.diagonal = self.outputscale * (1 + self.axis_stabilization)**2 + self.jitter

    @torch.no_grad()
    def _spectral_action(self, rhs, power):
        value, vector = _columns(rhs, self.size, dtype=self.dtype, device=self.device)
        grids = value.T.reshape(-1, self.ny, self.nx)
        projected = self.uy.T @ grids @ self.ux
        answer = self.uy @ (projected * self.eigenvalues.pow(power)) @ self.ux.T
        return _restore(answer.reshape(-1, self.size).T, vector)

    def matmul(self, rhs):
        return self._spectral_action(rhs, 1)

    def sqrt_matmul(self, rhs):
        return self._spectral_action(rhs, 0.5)

    def sqrt_t_matmul(self, rhs):
        return self._spectral_action(rhs, 0.5)

    def inv_matmul(self, rhs):
        return self._spectral_action(rhs, -1)


class SparseInterpolation:
    """Signed sparse row action with duplicate indices combined by scatter-add."""

    @torch.no_grad()
    def __init__(self, indices, weights, n_columns, *, dtype=torch.float64,
                 device="cpu", chunk_size=2048):
        self.dtype, self.device = dtype, torch.device(device)
        self.indices = _tensor(indices, dtype=torch.long, device=device)
        self.weights = _tensor(weights, dtype=dtype, device=device)
        self.n_columns = int(n_columns)
        self.chunk_size = int(chunk_size)
        if (self.indices.ndim != 2 or self.indices.shape != self.weights.shape
                or self.n_columns < 1 or self.chunk_size < 1):
            raise ValueError("invalid sparse interpolation dimensions")
        if (not torch.isfinite(self.weights).all() or torch.any(self.indices < 0)
                or torch.any(self.indices >= self.n_columns)):
            raise ValueError("nonfinite weights or out-of-range sparse indices")
        self.size = self.indices.shape[0]
        self.shape = (self.size, self.n_columns)

    def _rows(self, rows):
        if rows is None:
            return self.indices, self.weights
        rows = (rows.detach().to(dtype=torch.long, device=self.device) if isinstance(rows, torch.Tensor)
                else torch.tensor(rows, dtype=torch.long, device=self.device))
        if rows.ndim != 1 or torch.any(rows < 0) or torch.any(rows >= self.size):
            raise ValueError("query rows must be an in-range vector")
        return self.indices[rows], self.weights[rows]

    @torch.no_grad()
    def matmul(self, rhs, rows=None):
        value, vector = _columns(rhs, self.n_columns, dtype=self.dtype, device=self.device)
        indices, weights = self._rows(rows)
        result = torch.empty((len(indices), value.shape[1]), dtype=self.dtype, device=self.device)
        for start in range(0, len(indices), self.chunk_size):
            stop = min(start + self.chunk_size, len(indices))
            result[start:stop] = (value[indices[start:stop]] * weights[start:stop, :, None]).sum(1)
        return _restore(result, vector)

    @torch.no_grad()
    def t_matmul(self, rhs, rows=None):
        indices, weights = self._rows(rows)
        value, vector = _columns(rhs, len(indices), dtype=self.dtype, device=self.device)
        result = torch.zeros((self.n_columns, value.shape[1]), dtype=self.dtype, device=self.device)
        for start in range(0, len(indices), self.chunk_size):
            stop = min(start + self.chunk_size, len(indices))
            contributions = weights[start:stop, :, None] * value[start:stop, None, :]
            result.index_add_(0, indices[start:stop].reshape(-1), contributions.reshape(-1, value.shape[1]))
        return _restore(result, vector)

    def subset(self, rows):
        indices, weights = self._rows(rows)
        return SparseInterpolation(indices, weights, self.n_columns,
                                   dtype=self.dtype, device=self.device, chunk_size=self.chunk_size)


def _matern32(x, z, scale, length):
    distance = torch.linalg.vector_norm((x[..., :, None, :] - z[..., None, :, :]) / length, dim=-1)
    distance = math.sqrt(3.) * distance
    return scale * (1 + distance) * torch.exp(-distance)


class FrozenLocalPrior:
    """Frozen Matérn-3/2 VNNGP prior square-root precision factor R."""

    @torch.no_grad()
    def __init__(self, coordinates, predecessor_indices, lengthscale, outputscale,
                 jitter, *, dtype=torch.float64, device="cpu", chunk_size=512):
        self.dtype, self.device = dtype, torch.device(device)
        self.coordinates = _tensor(coordinates, dtype=dtype, device=device)
        self.indices = _tensor(predecessor_indices, dtype=torch.long, device=device)
        self.lengthscale = _positive("local lengthscale", lengthscale)
        self.outputscale = _positive("local outputscale", outputscale)
        self.jitter = _positive("local jitter", jitter)
        self.chunk_size = int(chunk_size)
        if (self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2
                or not torch.isfinite(self.coordinates).all() or self.indices.ndim != 2
                or self.chunk_size < 1):
            raise ValueError("invalid local coordinates or DAG")
        self.size, self.k = len(self.coordinates), self.indices.shape[1]
        if not 0 < self.k < self.size or len(self.indices) != self.size-self.k:
            raise ValueError("DAG must have shape (N-k,k), with 0<k<N")
        row = torch.arange(self.k, self.size, device=self.device)[:, None]
        if torch.any(self.indices < 0) or torch.any(self.indices >= row):
            raise ValueError("local DAG must contain only strict predecessors")
        if self.k > 1 and torch.any(torch.sort(self.indices, dim=1).values[:, 1:] ==
                                    torch.sort(self.indices, dim=1).values[:, :-1]):
            raise ValueError("local DAG has repeated neighbors")
        eye = torch.eye(self.k, dtype=dtype, device=self.device)
        c0 = _matern32(self.coordinates[:self.k], self.coordinates[:self.k],
                       self.outputscale, self.lengthscale) + self.jitter*eye
        self.initial_cholesky = torch.linalg.cholesky(c0)
        self.coefficients = torch.empty_like(self.indices, dtype=dtype)
        self.conditional_variance = torch.empty(self.size-self.k, dtype=dtype, device=self.device)
        for start in range(0, len(self.indices), self.chunk_size):
            stop = min(start+self.chunk_size, len(self.indices))
            neighbors = self.coordinates[self.indices[start:stop]]
            kernel = _matern32(neighbors, neighbors, self.outputscale, self.lengthscale) + self.jitter*eye
            cross = _matern32(neighbors, self.coordinates[start+self.k:stop+self.k, None],
                              self.outputscale, self.lengthscale).squeeze(-1)
            coefficients = torch.cholesky_solve(cross[..., None], torch.linalg.cholesky(kernel)).squeeze(-1)
            self.coefficients[start:stop] = coefficients
            self.conditional_variance[start:stop] = self.outputscale+self.jitter-(cross*coefficients).sum(-1)
        if (not torch.isfinite(self.conditional_variance).all()
                or torch.any(self.conditional_variance <= 0)):
            raise FloatingPointError("nonpositive local conditional variance; jitter was not changed")
        self.inverse_sd = self.conditional_variance.rsqrt()
        r0 = torch.linalg.solve_triangular(self.initial_cholesky, eye, upper=False)
        self.precision_diagonal = torch.zeros(self.size, dtype=dtype, device=self.device)
        self.precision_diagonal[:self.k] = r0.square().sum(0)
        self.precision_diagonal[self.k:] = self.conditional_variance.reciprocal()
        self.precision_diagonal.index_add_(0, self.indices.reshape(-1),
            (self.coefficients.square()/self.conditional_variance[:, None]).reshape(-1))

    @torch.no_grad()
    def whiten_local(self, rhs):
        value, vector = _columns(rhs, self.size, dtype=self.dtype, device=self.device)
        result = torch.empty_like(value)
        result[:self.k] = torch.linalg.solve_triangular(self.initial_cholesky, value[:self.k], upper=False)
        for start in range(0, len(self.indices), self.chunk_size):
            stop = min(start+self.chunk_size, len(self.indices))
            prediction = (value[self.indices[start:stop]]*self.coefficients[start:stop, :, None]).sum(1)
            result[start+self.k:stop+self.k] = (value[start+self.k:stop+self.k]-prediction)*self.inverse_sd[start:stop, None]
        return _restore(result, vector)

    @torch.no_grad()
    def whiten_local_t(self, rhs):
        value, vector = _columns(rhs, self.size, dtype=self.dtype, device=self.device)
        result = torch.zeros_like(value)
        result[:self.k] = torch.linalg.solve_triangular(self.initial_cholesky.T, value[:self.k], upper=True)
        result[self.k:] = value[self.k:]*self.inverse_sd[:, None]
        for start in range(0, len(self.indices), self.chunk_size):
            stop = min(start+self.chunk_size, len(self.indices))
            # Use the untouched own-row contribution: earlier scatters can modify result.
            contribution = (-self.coefficients[start:stop, :, None]
                * (value[start+self.k:stop+self.k]*self.inverse_sd[start:stop, None])[:, None, :])
            result.index_add_(0, self.indices[start:stop].reshape(-1), contribution.reshape(-1, value.shape[1]))
        return _restore(result, vector)

    def precision_matmul(self, rhs):
        return self.whiten_local_t(self.whiten_local(rhs))

    @torch.no_grad()
    def prior_solve(self, rhs):
        """Apply Q^-1 via two DAG triangular passes, without forming Q or Q^-1."""
        value, vector = _columns(rhs, self.size, dtype=self.dtype, device=self.device)
        work = value.clone()
        intermediate = torch.empty_like(value)
        for i in range(self.size-1, self.k-1, -1):
            j = i-self.k
            intermediate[i] = work[i]/self.inverse_sd[j]
            work.index_add_(0, self.indices[j], self.coefficients[j, :, None]*work[i].clone())
        intermediate[:self.k] = self.initial_cholesky.T@work[:self.k]
        result = torch.empty_like(value)
        result[:self.k] = self.initial_cholesky@intermediate[:self.k]
        for i in range(self.k, self.size):
            j = i-self.k
            result[i] = intermediate[i]/self.inverse_sd[j]+(self.coefficients[j, :, None]*result[self.indices[j]]).sum(0)
        return _restore(result, vector)


class FrozenQueryOperator:
    """h=[W Lg,B] with independent withheld conditional residuals.

    Global and local outputs here are centered; ``prior_mean`` supplies W mu_u.
    Duplicate query coordinates are rejected rather than assigned independent
    residuals. Query rows may be selected/reordered freely after construction.
    """

    @torch.no_grad()
    def __init__(self, global_prior, local_prior, interpolation, query_coordinates,
                 training_query_rows, query_neighbor_indices, global_mean):
        self.global_prior, self.local_prior = global_prior, local_prior
        self.interpolation = interpolation
        self.dtype, self.device = global_prior.dtype, global_prior.device
        self.size = interpolation.size
        self.dimension = global_prior.size+local_prior.size
        coordinates = _tensor(query_coordinates, dtype=self.dtype, device=self.device)
        if coordinates.shape != (self.size, 2) or not torch.isfinite(coordinates).all():
            raise ValueError("invalid query coordinates")
        if len(torch.unique(coordinates, dim=0)) != self.size:
            raise ValueError("duplicate query coordinates must be deduplicated with an explicit row mapping")
        rows = _tensor(training_query_rows, dtype=torch.long, device=self.device)
        if (rows.shape != (local_prior.size,) or len(torch.unique(rows)) != len(rows)
                or torch.any(rows < 0) or torch.any(rows >= self.size)):
            raise ValueError("invalid training-to-query mapping")
        if not torch.equal(coordinates[rows], local_prior.coordinates):
            raise ValueError("training coordinates disagree with training-to-query mapping")
        neighbors = _tensor(query_neighbor_indices, dtype=torch.long, device=self.device)
        if neighbors.shape != (self.size, local_prior.k):
            raise ValueError("query neighbor table must have shape (P,k)")
        mask = torch.ones(self.size, dtype=torch.bool, device=self.device)
        mask[rows] = False
        withheld = torch.nonzero(mask).flatten()
        if torch.any(neighbors[withheld] < 0) or torch.any(neighbors[withheld] >= local_prior.size):
            raise ValueError("withheld query neighbors are out of range")
        local_indices = torch.zeros_like(neighbors)
        local_weights = torch.zeros(neighbors.shape, dtype=self.dtype, device=self.device)
        local_indices[rows, 0] = torch.arange(local_prior.size, device=self.device)
        local_weights[rows, 0] = 1
        local_indices[withheld] = neighbors[withheld]
        self.residual = torch.zeros(self.size, dtype=self.dtype, device=self.device)
        eye = torch.eye(local_prior.k, dtype=self.dtype, device=self.device)
        for start in range(0, len(withheld), local_prior.chunk_size):
            selection = withheld[start:start+local_prior.chunk_size]
            x = local_prior.coordinates[neighbors[selection]]
            kernel = _matern32(x, x, local_prior.outputscale, local_prior.lengthscale)+local_prior.jitter*eye
            cross = _matern32(x, coordinates[selection, None], local_prior.outputscale, local_prior.lengthscale).squeeze(-1)
            coeff = torch.cholesky_solve(cross[..., None], torch.linalg.cholesky(kernel)).squeeze(-1)
            local_weights[selection] = coeff
            self.residual[selection] = local_prior.outputscale-(cross*coeff).sum(-1)
        if not torch.isfinite(self.residual).all() or torch.any(self.residual < 0):
            raise FloatingPointError("negative withheld conditional residual; jitter was not changed")
        self.local_interpolation = SparseInterpolation(local_indices, local_weights,
            local_prior.size, dtype=self.dtype, device=self.device)
        mean = float(global_mean)
        if not math.isfinite(mean):
            raise ValueError("global prior mean must be finite")
        self.prior_mean = interpolation.matmul(torch.full((global_prior.size,), mean,
                                                         dtype=self.dtype, device=self.device))
        self.convention = LOCAL_QUERY_CONVENTION

    @torch.no_grad()
    def matmul_parts(self, rhs, rows=None):
        value, vector = _columns(rhs, self.dimension, dtype=self.dtype, device=self.device)
        m = self.global_prior.size
        global_part = self.interpolation.matmul(self.global_prior.sqrt_matmul(value[:m]), rows)
        local_part = self.local_interpolation.matmul(value[m:], rows)
        return _restore(global_part, vector), _restore(local_part, vector)

    def matmul(self, rhs, rows=None):
        g, l = self.matmul_parts(rhs, rows)
        return g+l

    @torch.no_grad()
    def t_matmul(self, rhs, rows=None):
        n_rows = self.size if rows is None else len(rows)
        value, vector = _columns(rhs, n_rows, dtype=self.dtype, device=self.device)
        g = self.global_prior.sqrt_t_matmul(self.interpolation.t_matmul(value, rows))
        l = self.local_interpolation.t_matmul(value, rows)
        return _restore(torch.cat((g, l)), vector)


@dataclass
class FrozenPriorOperators:
    global_prior: FrozenGridPrior
    local_prior: FrozenLocalPrior
    train_interpolation: SparseInterpolation
    query: FrozenQueryOperator
    noise_variance: float
    snapshot: object = None

    def __post_init__(self):
        self.noise_variance = _positive("joint noise variance", self.noise_variance)
        self.dimension = self.global_prior.size+self.local_prior.size
        self.dtype, self.device = self.global_prior.dtype, self.global_prior.device
        # Scalar global and exact local diagonal scaling: this is a fixed SPD
        # approximation, not the exact global diagonal in symmetric-root coordinates.
        global_scale = 1+(self.local_prior.size/self.global_prior.size)*self.global_prior.diagonal/self.noise_variance
        self.scaled_jacobi_diagonal = torch.cat((torch.full((self.global_prior.size,), global_scale,
            dtype=self.dtype, device=self.device), self.local_prior.precision_diagonal+1/self.noise_variance))
        self.scaled_jacobi_description = "global=1+(N/M)*diag(Ku)/noise; local=diag(Ql)+1/noise"

    @torch.no_grad()
    def training_matmul(self, rhs):
        value, vector = _columns(rhs, self.dimension, dtype=self.dtype, device=self.device)
        m = self.global_prior.size
        answer = self.train_interpolation.matmul(self.global_prior.sqrt_matmul(value[:m]))+value[m:]
        return _restore(answer, vector)

    @torch.no_grad()
    def training_t_matmul(self, rhs):
        value, vector = _columns(rhs, self.local_prior.size, dtype=self.dtype, device=self.device)
        answer = torch.cat((self.global_prior.sqrt_t_matmul(self.train_interpolation.t_matmul(value)), value))
        return _restore(answer, vector)

    @torch.no_grad()
    def precision_matmul(self, rhs):
        value, vector = _columns(rhs, self.dimension, dtype=self.dtype, device=self.device)
        m = self.global_prior.size
        prior = torch.cat((value[:m], self.local_prior.precision_matmul(value[m:])))
        result = prior+self.training_t_matmul(self.training_matmul(value))/self.noise_variance
        return _restore(result, vector)

    @torch.no_grad()
    def prior_inverse_matmul(self, rhs):
        value, vector = _columns(rhs, self.dimension, dtype=self.dtype, device=self.device)
        m = self.global_prior.size
        return _restore(torch.cat((value[:m], self.local_prior.prior_solve(value[m:]))), vector)


@torch.no_grad()
def build_frozen_prior_operators(snapshot, dtype=torch.float64, device="cpu"):
    """Build immutable numeric actions from one verified FrozenGaussianFit."""
    if isinstance(dtype, str):
        if dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        dtype = getattr(torch, dtype)
    global_prior = FrozenGridPrior(snapshot.grid_axes, snapshot.global_lengthscale,
        snapshot.global_scale, snapshot.global_jitter, axis_stabilization=snapshot.axis_stabilization,
        dtype=dtype, device=device)
    local_prior = FrozenLocalPrior(snapshot.training_coordinates, snapshot.local_dag,
        snapshot.local_lengthscale, snapshot.local_scale, snapshot.local_jitter, dtype=dtype, device=device)
    interpolation = SparseInterpolation(snapshot.interpolation_indices, snapshot.interpolation_weights,
        global_prior.size, dtype=dtype, device=device)
    query = FrozenQueryOperator(global_prior, local_prior, interpolation, snapshot.query_coordinates,
        snapshot.training_query_rows, snapshot.query_neighbor_indices, snapshot.global_mean)
    return FrozenPriorOperators(global_prior, local_prior, interpolation.subset(snapshot.training_query_rows),
                                query, snapshot.noise_variance, snapshot)
