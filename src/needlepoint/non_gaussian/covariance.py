"""Dense reference and matrix-free weighted Gaussian covariance calculations."""
import math

import numpy as np
import torch

from ..gp_joint_conditioning import FrozenConditioningSettings, independent_pcg
from .operators import WeightedPrecision


class GaussianCovariance:
    def __init__(self, prior, evaluation, information, settings):
        self.precision = WeightedPrecision(prior, evaluation, information)
        self.prior, self.settings = prior, settings
        self.dimension, self.device, self.dtype = prior.dimension, prior.device, prior.dtype
        self.solve_receipts = []
        self.factor = None
        self._logdet = None
        self.preconditioner_diagonal = self.precision.preconditioner_diagonal()
        if settings.backend == 'dense':
            self._check_dense()
            eye = torch.eye(self.dimension, dtype=self.dtype, device=self.device)
            matrix = self.precision.matmul(eye)
            self.factor = torch.linalg.cholesky((matrix + matrix.T) / 2)

    def _check_dense(self):
        if self.dimension > self.settings.dense_max_dimension:
            raise ValueError('Dense covariance exceeds dense_max_dimension; select the pcg backend')
        if 8 * self.dimension**2 * 5 > self.settings.memory_cap_bytes:
            raise MemoryError('Dense covariance workspace exceeds memory_cap_bytes')

    def solve(self, rhs):
        if 8 * rhs.numel() * 12 > self.settings.memory_cap_bytes:
            raise MemoryError('Solve workspace exceeds memory_cap_bytes; reduce rhs_batch_size')
        if self.factor is not None:
            answer = torch.cholesky_solve(rhs.reshape(self.dimension, -1), self.factor)
            return answer[:, 0] if rhs.ndim == 1 else answer
        options = FrozenConditioningSettings(rtol=self.settings.pcg_rtol,
            atol=self.settings.pcg_atol, max_iterations=self.settings.pcg_max_iterations,
            residual_refresh=self.settings.residual_refresh)
        def preconditioner(value):
            diagonal = self.preconditioner_diagonal
            return value / (diagonal[:, None] if value.ndim == 2 else diagonal)
        result = independent_pcg(self.precision.matmul, rhs, options,
                                 inverse_preconditioner=preconditioner)
        self.solve_receipts.append(result.receipt)
        return result.solution

    def logdet_ratio(self):
        if self._logdet is not None:
            return self._logdet
        if self.factor is not None:
            result = (float(2 * self.factor.diagonal().log().sum() - self.prior.logdet), 0.0)
        elif self.settings.logdet_method == 'dense':
            self._check_dense()
            eye = torch.eye(self.dimension, dtype=self.dtype, device=self.device)
            matrix = self.precision.matmul(eye)
            result = (float(2 * torch.linalg.cholesky((matrix + matrix.T) / 2).diagonal().log().sum()
                            - self.prior.logdet), 0.0)
        else:
            result = self._slq()
        self._logdet = result
        return result

    def _slq(self):
        """Rademacher SLQ on R0^-T P R0^-1, whose logdet is the required ratio.

        Full reorthogonalization limits loss of Krylov orthogonality. The
        returned SE measures probe variability, not Lanczos truncation bias.
        """
        generator = torch.Generator(device=self.device).manual_seed(self.settings.seed + 17)
        values = []
        steps = min(self.settings.lanczos_steps, self.dimension)
        workspace = 8 * self.dimension * (steps + 12)
        if workspace > self.settings.memory_cap_bytes:
            raise MemoryError('Lanczos workspace exceeds memory_cap_bytes')
        for _ in range(self.settings.logdet_probes):
            probe = torch.randint(0, 2, (self.dimension,), generator=generator, device=self.device)
            q = (2 * probe.to(self.dtype) - 1) / math.sqrt(self.dimension)
            basis, alpha, beta = [], [], []
            for j in range(steps):
                basis.append(q)
                z = self.precision.whitened_matmul(q)
                a = q @ z
                z -= a * q
                if j:
                    z -= beta[-1] * basis[-2]
                # A second pass improves small-spectrum accuracy in float64.
                for _pass in range(2):
                    for previous in basis:
                        z -= (previous @ z) * previous
                b = torch.linalg.vector_norm(z)
                alpha.append(a)
                if j == steps - 1 or float(b) < 1e-13:
                    break
                beta.append(b)
                q = z / b
            tri = torch.diag(torch.stack(alpha))
            if beta:
                off = torch.stack(beta[:len(alpha) - 1])
                tri += torch.diag(off, 1) + torch.diag(off, -1)
            eig, vec = torch.linalg.eigh(tri)
            if not torch.isfinite(eig).all() or (eig <= 0).any():
                raise FloatingPointError('Nonpositive Lanczos spectrum; increase work precision or audit the operator')
            values.append(float(self.dimension * (vec[0].square() * eig.log()).sum()))
        values = np.asarray(values)
        return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))

    def moments(self, query, *, final=False):
        if query.dimension != self.dimension or query.device != self.device:
            raise ValueError('Query and prior must use the same latent coordinates and device')
        if self.settings.variance_method == 'exact':
            return self._exact_moments(query)
        count = self.settings.final_draws if final else self.settings.training_draws
        return self._rb_moments(query, count)

    def _exact_moments(self, query):
        values = {name: torch.empty(query.size, dtype=self.dtype, device=self.device)
                  for name in ['global_variance', 'local_variance', 'global_local_covariance']}
        for begin in range(0, query.size, self.settings.rhs_batch_size):
            end = min(begin + self.settings.rhs_batch_size, query.size)
            rows = torch.arange(begin, end, device=self.device)
            h = query.t_matmul(torch.eye(len(rows), dtype=self.dtype, device=self.device), rows)
            hg, hl = h.clone(), h.clone()
            hg[query.global_dimension:] = 0
            hl[:query.global_dimension] = 0
            solved_g, solved_l = self.solve(hg), self.solve(hl)
            values['global_variance'][rows] = (hg * solved_g).sum(0)
            values['local_variance'][rows] = (hl * solved_l).sum(0)
            values['global_local_covariance'][rows] = (hg * solved_l).sum(0)
        total = values['global_variance'] + values['local_variance'] + 2 * values['global_local_covariance']
        if not torch.isfinite(total).all() or (total < -1e-12).any():
            raise FloatingPointError('Invalid coupled marginal variance')
        values.update(latent_gp_variance=total.clamp_min(0), variance_mcse=torch.zeros_like(total),
                      variance_lower=total, variance_upper=total, draws=0)
        return values

    def _rb_design(self, query):
        groups = []
        gamma = torch.zeros(query.size, dtype=self.dtype, device=self.device)
        anchors = query.anchor_local()
        block_ids = torch.where(anchors < 0, -1, anchors // self.settings.rb_block_size)
        for block_id in torch.unique(block_ids).tolist():
            if block_id < 0:
                continue
            rows = torch.nonzero(block_ids == block_id).flatten()
            start = self.prior.global_dimension + block_id * self.settings.rb_block_size
            stop = min(start + self.settings.rb_block_size, self.dimension)
            ids = torch.arange(start, stop, device=self.device)
            if 8 * self.dimension * len(ids) * 5 > self.settings.memory_cap_bytes:
                raise MemoryError('RB block workspace exceeds memory_cap_bytes')
            columns = torch.zeros((self.dimension, len(ids)), dtype=self.dtype, device=self.device)
            columns[ids, torch.arange(len(ids), device=self.device)] = 1
            # Applying the complete precision retains every descendant and
            # observation row, including those outside this selected block.
            pbb = self.precision.matmul(columns)[ids]
            factor = torch.linalg.cholesky((pbb + pbb.T) / 2)
            h = query.matmul(columns, rows)
            coefficient = torch.cholesky_solve(h.T, factor).T
            gamma[rows] = (coefficient * h).sum(1)
            groups.append((rows, ids, coefficient))
        return gamma, groups

    def _rb_moments(self, query, count):
        from scipy.stats import chi2
        gamma, groups = self._rb_design(query)
        sums = torch.zeros((4, query.size), dtype=self.dtype, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(self.settings.seed + 29)
        for begin in range(0, count, self.settings.rhs_batch_size):
            batch = min(self.settings.rhs_batch_size, count - begin)
            rhs = self.precision.forcing(generator, batch)
            sample = self.solve(rhs)
            g, local = query.matmul_parts(sample)
            for rows, ids, coefficient in groups:
                local[rows] -= coefficient @ rhs[ids]
            sums[0] += g.square().sum(1)
            sums[1] += local.square().sum(1)
            sums[2] += (g * local).sum(1)
            sums[3] += (g + local).square().sum(1)
        remaining = sums[3] / count
        total = gamma + remaining
        values = dict(global_variance=sums[0] / count, local_variance=gamma + sums[1] / count,
            global_local_covariance=sums[2] / count, latent_gp_variance=total,
            variance_mcse=remaining * math.sqrt(2 / count),
            variance_lower=gamma + remaining * count / chi2.ppf(.975, count),
            variance_upper=gamma + remaining * count / chi2.ppf(.025, count),
            analytic_conditional_variance=gamma, draws=count)
        if not torch.isfinite(total).all() or (total < 0).any():
            raise FloatingPointError('Nonfinite RB posterior variance')
        return values

    def samples(self, count, *, seed):
        """Coherent, uncorrected zero-mean joint field-coordinate draws."""
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.solve(self.precision.forcing(generator, count))

    def selected_bounds(self, query, rows):
        rows = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        if (rows.ndim != 1 or len(rows) == 0 or len(torch.unique(rows)) != len(rows)
                or (rows < 0).any() or (rows >= query.size).any()):
            raise ValueError('Reference rows must be nonempty, unique and within query support')
        h = query.t_matmul(torch.eye(len(rows), dtype=self.dtype, device=self.device), rows)
        answer = self.solve(h)
        pa = self.precision.matmul(answer)
        residual = h - pa
        lower = (2 * h * answer - answer * pa).sum(0)
        upper = lower + (residual * self.prior.inverse_matmul(residual)).sum(0)
        return dict(variance=(h * answer).sum(0), lower=lower, upper=upper)
