"""Coupled conditional Gaussian variational fitting for three observation models.

Experimental, explicit API. Fixed kernels use exact Gaussian expectations and
adaptive mean/information updates. Dense references use joint-ELBO ascent;
stochastic covariance estimates use a consistent moment-residual search.
Optional nested kernel fitting is a deterministic reference method; GPU
throughput and whole-slide calibration are not asserted.
"""
from dataclasses import asdict, dataclass, field, replace
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .covariance import GaussianCovariance
from .data import fingerprint
from .likelihoods import TobitLikelihood, PoissonLikelihood, LGCPIntegral
from .operators import DenseEvaluation, DensePrior


@dataclass(frozen=True)
class CoupledVISettings:
    backend: str = 'dense'
    variance_method: str = 'exact'
    max_iterations: int = 200
    stationarity_tolerance: float = 1e-5
    initial_step: float = 1.0
    minimum_step: float = 2**-24
    fit_baseline: bool = True
    baseline_prior_sd: float | None = None
    fit_noise: bool = False
    noise_bounds: tuple[float, float] | None = None
    dense_max_dimension: int = 2048
    rhs_batch_size: int = 8
    rb_block_size: int = 64
    training_draws: int = 256
    max_training_draws: int = 4096
    final_draws: int = 2048
    maximum_relative_mcse: float = .05
    maximum_exponential_bias_proxy: float = .01
    reference_relative_tolerance: float = .001
    logdet_method: str = 'slq'
    logdet_probes: int = 16
    lanczos_steps: int = 40
    pcg_rtol: float = 1e-10
    pcg_atol: float = 1e-12
    pcg_max_iterations: int = 4000
    residual_refresh: int = 25
    memory_cap_bytes: int = 8 * 1024**3
    seed: int = 20261019
    failure_policy: str = 'raise'

    def __post_init__(self):
        if self.backend not in ('dense', 'pcg') or self.variance_method not in ('exact', 'rb'):
            raise ValueError('Use dense/pcg backend and exact/rb variance_method')
        if self.logdet_method not in ('dense', 'slq') or self.failure_policy not in ('raise', 'record'):
            raise ValueError('Invalid logdet_method or failure_policy')
        for name in ['max_iterations', 'dense_max_dimension', 'rhs_batch_size', 'rb_block_size',
                     'training_draws', 'max_training_draws', 'final_draws', 'logdet_probes',
                     'lanczos_steps', 'pcg_max_iterations', 'residual_refresh', 'memory_cap_bytes']:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.logdet_probes < 2 or self.max_training_draws < self.training_draws:
            raise ValueError('Use at least two determinant probes and a sufficient max_training_draws')
        for name in ['stationarity_tolerance', 'maximum_relative_mcse', 'maximum_exponential_bias_proxy',
                     'reference_relative_tolerance']:
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f'{name} must be between zero and one')
        if not 0 < self.minimum_step <= self.initial_step <= 1:
            raise ValueError('Require 0 < minimum_step <= initial_step <= 1')
        if not 0 <= self.pcg_rtol < 1 or not 0 <= self.pcg_atol < 1:
            raise ValueError('Invalid PCG tolerances')
        if self.baseline_prior_sd is not None and not 0 < self.baseline_prior_sd < math.inf:
            raise ValueError('baseline_prior_sd must be finite and positive')
        if self.fit_noise and (self.noise_bounds is None or len(self.noise_bounds) != 2
                or not 0 < self.noise_bounds[0] < self.noise_bounds[1] < math.inf):
            raise ValueError('Noise fitting requires explicit positive finite noise_bounds')


class VIConvergenceError(RuntimeError):
    def __init__(self, result):
        super().__init__(f'Coupled VI did not converge: {result.diagnostics}')
        self.result = result


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value


@dataclass
class CoupledVIResult:
    prior: object
    evaluation: object
    likelihood: object
    mean: torch.Tensor
    information: torch.Tensor
    baseline: float
    settings: CoupledVISettings
    converged: bool
    diagnostics: dict
    history: list
    data_metadata: dict
    event_score: torch.Tensor
    event_count: int = 0
    event_log_effort: float = 0.0
    quadrature: object = None
    hyperparameters: dict = field(default_factory=dict)
    hyperparameter_history: list = field(default_factory=list)

    @property
    def posterior_identity(self):
        return fingerprint(_numpy(self.mean), _numpy(self.information), metadata=dict(
            approximation='coupled_conditional_gaussian_vi', prior=self.prior.fingerprint(),
            evaluation=self.evaluation.fingerprint(), observations=self.likelihood.fingerprint(),
            baseline=self.baseline, data=self.data_metadata,
            event_score_hash=fingerprint(_numpy(self.event_score)), event_count=self.event_count,
            event_log_effort=self.event_log_effort))

    def _covariance(self):
        return GaussianCovariance(self.prior, self.evaluation, self.information, self.settings)

    def predict(self, coordinates=None, *, evaluation=None, exposure=None, effort=None,
                threshold=None):
        if evaluation is not None and coordinates is not None:
            raise ValueError('Supply coordinates or evaluation, not both')
        query = evaluation if evaluation is not None else (
            self.evaluation if coordinates is None else self.prior.evaluation(coordinates))
        covariance = self._covariance()
        moments = covariance.moments(query, final=True)
        self._prediction_solver_receipts = covariance.solve_receipts
        global_mean, local_mean = query.matmul_parts(self.mean)
        mean = global_mean + local_mean
        predictor = self.baseline + mean
        variance = moments['latent_gp_variance']
        if self.likelihood.name == 'poisson':
            if exposure is None:
                exposure = self.likelihood.exposure if query.fingerprint() == self.evaluation.fingerprint() else 1.0
            observed = self.likelihood.predict(predictor, variance, exposure=exposure)
        elif self.likelihood.name == 'lgcp':
            if effort is None:
                effort = self.likelihood.effort if query.fingerprint() == self.evaluation.fingerprint() else 1.0
            observed = self.likelihood.predict(predictor, variance, effort=effort)
        else:
            if threshold is None and query.fingerprint() != self.evaluation.fingerprint() and not torch.all(
                    self.likelihood.threshold == self.likelihood.threshold[0]):
                raise ValueError('A new Tobit query with varying thresholds requires threshold=')
            adapter = copy.copy(self.likelihood)
            if threshold is not None:
                from .likelihoods import vector
                adapter.threshold = vector(threshold, 'prediction threshold', size=query.size, device=self.prior.device)
            observed = adapter.predict(predictor, variance)
        maps = dict(moments, latent_gp_mean=mean, latent_predictor_mean=predictor,
                    global_mean=global_mean, local_mean=local_mean, **observed)
        if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all() for v in maps.values()):
            raise FloatingPointError('Nonfinite posterior predictions')
        if getattr(query, 'coordinates', None) is not None:
            maps['coordinates_um'] = query.coordinates.copy()
        return {name: _numpy(value) for name, value in maps.items()}

    def sample_field(self, count, *, coordinates=None, evaluation=None, seed=None):
        if not isinstance(count, int) or count < 1:
            raise ValueError('count must be a positive integer')
        if count * self.prior.dimension * 8 * 8 > self.settings.memory_cap_bytes:
            raise MemoryError('Joint sample workspace exceeds memory_cap_bytes; stream smaller batches')
        query = evaluation if evaluation is not None else (
            self.evaluation if coordinates is None else self.prior.evaluation(coordinates))
        delta = self._covariance().samples(count, seed=self.settings.seed + 101 if seed is None else seed)
        return _numpy(query.matmul(self.mean)[:, None] + query.matmul(delta))

    def regional_count_moments(self, quadrature=None, *, draws=2048, seed=None):
        """LGCP future-count moments from coherent, uncorrected field draws."""
        if self.likelihood.name != 'lgcp':
            raise ValueError('Regional point-process counts require an LGCP fit')
        quadrature = self.quadrature if quadrature is None else quadrature
        if quadrature is None or not isinstance(draws, int) or draws < 2:
            raise ValueError('Supply quadrature and at least two draws')
        query = self.prior.evaluation(quadrature.coordinates)
        weights = torch.tensor((quadrature.areas * quadrature.effort).copy(),
                               dtype=self.prior.dtype, device=self.prior.device)
        covariance = self._covariance()
        generator = torch.Generator(device=self.prior.device).manual_seed(
            self.settings.seed + 103 if seed is None else seed)
        mean = self.baseline + query.matmul(self.mean)
        totals = []
        from .likelihoods import checked_exp
        for start in range(0, draws, self.settings.rhs_batch_size):
            count = min(self.settings.rhs_batch_size, draws - start)
            # No query-specific RB corrections enter this joint sample.
            sample = covariance.solve(covariance.precision.forcing(generator, count))
            positive = weights > 0
            values = checked_exp(mean[positive, None] + query.matmul(sample)[positive])
            totals.extend(_numpy(weights[positive] @ values).tolist())
        totals = np.asarray(totals)
        variance = float(totals.var(ddof=1))
        return dict(integrated_intensity_mean=float(totals.mean()),
            integrated_intensity_variance=variance, observed_count_mean=float(totals.mean()),
            observed_count_variance=float(totals.mean() + variance),
            integrated_mean_mcse=float(np.sqrt(variance / draws)), draws=draws,
            sampling='coherent_uncorrected_joint_gaussian', posterior_identity=self.posterior_identity)

    def audit_variances(self, rows, *, evaluation=None, estimated_variance=None):
        """Selected direct-solve checks with deterministic residual-energy bounds."""
        query = self.evaluation if evaluation is None else evaluation
        covariance = self._covariance()
        reference = covariance.selected_bounds(query, rows)
        self._reference_solver_receipts = covariance.solve_receipts
        values = np.asarray(_numpy(reference['variance']))
        lower, upper = _numpy(reference['lower']), _numpy(reference['upper'])
        width = (upper - lower) / np.maximum(values, np.finfo(float).tiny)
        if not np.isfinite(width).all() or np.max(width) > self.settings.reference_relative_tolerance:
            raise FloatingPointError('Selected reference variance enclosure is too wide; tighten PCG tolerances')
        if estimated_variance is None:
            estimated_variance = self.predict(evaluation=query)['latent_gp_variance']
        estimates = np.asarray(estimated_variance)[np.asarray(rows, int)]
        errors = np.abs(estimates - values) / np.maximum(values, np.finfo(float).tiny)
        return dict(query_rows=list(map(int, rows)), reference_variance=values.tolist(),
            lower=lower.tolist(), upper=upper.tolist(), reference_relative_width=width.tolist(),
            absolute_relative_error=errors.tolist(), p95_absolute_relative_error=float(np.quantile(errors, .95)),
            scope='selected_locations_under_fixed_gaussian_vi_covariance',
            posterior_identity=self.posterior_identity)

    def save(self, output_directory, *, storage=None):
        """Export one fitted feature's native maps, VI state and hashed receipts.

        This is an audited map/state export, not a portable prior checkpoint.
        Retain the declared prior and input data to evaluate new query locations.
        """
        if not self.converged:
            raise ValueError('Refusing to export an unconverged VI fit')
        from ..storage import VIStorageSettings, encode_vi_maps
        from ..posterior_blocks import atomic_savez_maps
        from ..artifacts import sha256_file
        storage = VIStorageSettings() if storage is None else storage
        output = Path(output_directory)
        if output.exists():
            raise FileExistsError(output)
        maps = self.predict()
        encoded = encode_vi_maps(storage, maps)
        selected = np.linspace(0, self.evaluation.size - 1, min(8, self.evaluation.size), dtype=int)
        audit = self.audit_variances(selected, estimated_variance=maps['latent_gp_variance'])
        audit['scope'] = 'selected_working_precision_gaussian_vi_covariance'
        if 'latent_gp_variance' in encoded:
            original = np.asarray(maps['latent_gp_variance'])[selected]
            stored = encoded['latent_gp_variance'][selected].astype(float)
            audit['stored_variance_relative_quantization_error'] = (
                np.abs(stored-original)/np.maximum(original, np.finfo(float).tiny)).tolist()
            reference = np.asarray(audit['reference_variance'])
            audit['stored_variance_relative_error_to_reference'] = (
                np.abs(stored-reference)/np.maximum(reference, np.finfo(float).tiny)).tolist()
        output.mkdir(parents=True, exist_ok=False)
        atomic_savez_maps(output / 'maps.npz', **encoded)
        artifacts = ['maps.npz', 'variance_audit.json', 'solver_receipts.json']
        if storage.retain_state:
            np.savez_compressed(output / 'vi_state.npz', mean=_numpy(self.mean),
                                information=_numpy(self.information), baseline=self.baseline,
                                event_score=_numpy(self.event_score))
            artifacts.append('vi_state.npz')
        (output / 'variance_audit.json').write_text(json.dumps(audit, indent=2, allow_nan=False) + '\n')
        solver_receipts = dict(backend=self.settings.backend,
            prediction=getattr(self, '_prediction_solver_receipts', []),
            selected_references=getattr(self, '_reference_solver_receipts', []))
        (output / 'solver_receipts.json').write_text(json.dumps(solver_receipts, indent=2, allow_nan=False) + '\n')
        metadata = dict(schema_version=1, experimental=True,
            approximation='coupled_conditional_gaussian_vi', likelihood=self.likelihood.name,
            posterior_identity=self.posterior_identity, baseline=self.baseline,
            prior_identity=self.prior.fingerprint(), observation_identity=self.likelihood.fingerprint(),
            evaluation_identity=self.evaluation.fingerprint(),
            posterior_information_hash=fingerprint(_numpy(self.information)),
            latent_mean_hash=fingerprint(_numpy(self.mean)),
            likelihood_noise_sd=getattr(self.likelihood, 'noise_sd', None),
            likelihood_configuration=dict(
                quadrature_order=len(self.likelihood.nodes) if hasattr(self.likelihood, 'nodes') else None,
                measurement_scale=getattr(self.likelihood, 'measurement_scale', None),
                exposure_semantics=getattr(self.likelihood, 'exposure_semantics', None)),
            settings=asdict(self.settings), diagnostics=self.diagnostics,
            vi_history=self.history, data_metadata=self.data_metadata,
            hyperparameters=self.hyperparameters, hyperparameter_history=self.hyperparameter_history,
            storage=dict(asdict(storage), archive_layout='npz-stored',
                         required_moment_pair=storage.field_scale,
                         quantization='IEEE round-to-nearest; overflow rejected; underflow possible'),
            parameter_uncertainty_included=False, prior_checkpoint_included=False,
            artifact_sha256={name: sha256_file(output / name) for name in artifacts})
        (output / 'complete.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
        return output


def _resolve_evaluation(prior, coordinates, evaluation):
    if coordinates is not None and evaluation is not None:
        raise ValueError('Supply coordinates or an explicit evaluation operator')
    if evaluation is not None:
        return evaluation
    if coordinates is None:
        raise ValueError('Supply coordinates or evaluation')
    return prior.evaluation(coordinates)


def _initial_baseline(likelihood, settings, event_count):
    if likelihood.name == 'poisson':
        total, opportunity = float(likelihood.counts.sum()), float(likelihood.exposure.sum())
    elif likelihood.name == 'lgcp':
        total, opportunity = event_count, float((likelihood.areas * likelihood.effort).sum())
    else:
        detected = ~likelihood.is_censored
        if detected.any():
            return float(likelihood.values[detected].mean())
        total, opportunity = 0, 1
    if total == 0:
        if settings.fit_baseline and settings.baseline_prior_sd is None:
            raise ValueError('All-zero/all-censored feature has no finite unregularized baseline optimum; '
                             'set baseline_prior_sd or fix the baseline')
        return 0.0
    if opportunity <= 0:
        raise ValueError('Observed counts/events require positive integrated exposure or effort')
    return math.log(total / opportunity)


@torch.no_grad()
def fit_coupled_vi(prior, likelihood, evaluation, *, settings=None, baseline=None,
                   initial_mean=None, initial_information=None, event_score=None,
                   event_count=0, event_log_effort=0.0, data_metadata=None):
    """Fit mean and covariance information under a full joint Gaussian ELBO.

    Exact references use joint-ELBO backtracking. With RB marginals or SLQ,
    information updates reduce the estimated moment-equation residual; the
    estimated ELBO is a final diagnostic, not an acceptance objective or a
    certificate of exact-ELBO stationarity. Dense references remain required.
    """
    settings = settings or CoupledVISettings()
    likelihood = copy.deepcopy(likelihood)
    if not isinstance(event_count, (int, np.integer)) or event_count < 0 or not math.isfinite(event_log_effort):
        raise ValueError('Event count must be a nonnegative integer and event_log_effort finite')
    if evaluation.size != likelihood.size or evaluation.dimension != prior.dimension:
        raise ValueError('Likelihood, evaluation and prior dimensions disagree')
    if evaluation.device != prior.device or likelihood.expected(
            torch.zeros(evaluation.size, device=prior.device, dtype=prior.dtype),
            torch.zeros(evaluation.size, device=prior.device, dtype=prior.dtype)).value.device != prior.device:
        raise ValueError('Likelihood, evaluation and prior devices must agree')
    if settings.fit_noise and likelihood.name != 'tobit':
        raise ValueError('fit_noise is supported only for the Gaussian measurement noise of Tobit')
    c = _initial_baseline(likelihood, settings, event_count) if baseline is None else float(baseline)
    if not math.isfinite(c):
        raise ValueError('baseline must be finite')
    # Check degeneracy even when the caller supplied a finite initial baseline.
    _initial_baseline(likelihood, settings, event_count)
    m = torch.zeros(prior.dimension, dtype=prior.dtype, device=prior.device)
    if initial_mean is not None:
        m = torch.as_tensor(initial_mean, dtype=prior.dtype, device=prior.device).clone()
    linear = torch.zeros_like(m) if event_score is None else torch.as_tensor(
        event_score, dtype=prior.dtype, device=prior.device).clone()
    if m.shape != (prior.dimension,) or linear.shape != m.shape or not torch.isfinite(m).all() or not torch.isfinite(linear).all():
        raise ValueError('Initial mean and event score must be finite latent vectors')
    zeros = torch.zeros(evaluation.size, dtype=prior.dtype, device=prior.device)
    info = likelihood.expected(c + evaluation.matmul(m), zeros).information
    if initial_information is not None:
        info = torch.as_tensor(initial_information, dtype=prior.dtype, device=prior.device).clone()
    baseline_precision = 0 if settings.baseline_prior_sd is None else settings.baseline_prior_sd**-2
    history = []
    stochastic_fitting = settings.variance_method == 'rb' or (
        settings.backend == 'pcg' and settings.logdet_method == 'slq')
    training_draws = settings.training_draws

    def score_state(covariance, mean, intercept, moments=None, *, evaluate_objective=False):
        moments = covariance.moments(evaluation) if moments is None else moments
        variance = moments['latent_gp_variance']
        expected = likelihood.expected(intercept + evaluation.matmul(mean), variance)
        mean_energy = mean @ prior.precision_matmul(mean)
        location_objective = float(expected.value.sum() + linear @ mean + event_count * intercept
                                   + event_log_effort - .5 * mean_energy
                                   - .5 * baseline_precision * intercept**2)
        if settings.baseline_prior_sd is not None:
            location_objective -= math.log(settings.baseline_prior_sd) + .5 * math.log(2 * math.pi)
        if not math.isfinite(location_objective):
            raise FloatingPointError('Nonfinite expected log likelihood or mean penalty')
        objective = kl = logdet_se = None
        if not stochastic_fitting or evaluate_objective:
            logdet, logdet_se = covariance.logdet_ratio()
            kl = .5 * (mean_energy - covariance.precision.information @ variance + logdet)
            objective = location_objective + .5 * float(covariance.precision.information @ variance) - .5 * logdet
            if not math.isfinite(objective):
                raise FloatingPointError('Nonfinite joint ELBO')
        relative_mcse = moments['variance_mcse'] / variance.clamp_min(torch.finfo(prior.dtype).tiny)
        active = expected.information > 0
        max_mcse = float(relative_mcse[active].max()) if active.any() else 0.0
        # Leading-order log-link bias diagnostic, not an error enclosure.
        bias_proxy = float((moments['variance_mcse'][active].square() / 8).max()) if active.any() else 0.0
        return dict(covariance=covariance, moments=moments, expected=expected,
                    objective=objective, kl=None if kl is None else float(kl), logdet_se=logdet_se,
                    location_objective=location_objective,
                    max_relative_mcse=max_mcse, exponential_bias_proxy=bias_proxy)

    def state_for(information, mean, intercept):
        nonlocal training_draws
        while True:
            draw_settings = replace(settings, training_draws=training_draws)
            covariance = GaussianCovariance(prior, evaluation, information, draw_settings)
            state = score_state(covariance, mean, intercept)
            accurate = state['max_relative_mcse'] <= settings.maximum_relative_mcse
            if likelihood.name in ('poisson', 'lgcp'):
                accurate &= state['exponential_bias_proxy'] <= settings.maximum_exponential_bias_proxy
            if accurate or settings.variance_method == 'exact':
                return state
            if training_draws >= settings.max_training_draws:
                raise FloatingPointError('Variance refresh failed MC precision/bias diagnostics; '
                                         'increase max_training_draws or RB integration')
            training_draws = min(2 * training_draws, settings.max_training_draws)

    state = state_for(info, m, c)
    converged = False
    termination_reason = 'maximum_iterations'
    noise_change = 0.0
    for iteration in range(settings.max_iterations):
        previous_objective = state['objective']
        expected = state['expected']
        score = evaluation.t_matmul(expected.score) + linear
        gradient = score - prior.precision_matmul(m)
        baseline_gradient = float(expected.score.sum()) + event_count - baseline_precision * c
        # The expected-curvature Newton system includes intercept/field coupling.
        curvature = GaussianCovariance(prior, evaluation, expected.information, settings)
        if settings.fit_baseline:
            cross = evaluation.t_matmul(expected.information)
            solved = curvature.solve(torch.stack((gradient, cross), dim=1))
            schur = float(expected.information.sum() + baseline_precision - cross @ solved[:, 1])
            if schur <= 0 or not math.isfinite(schur):
                raise FloatingPointError('Nonpositive baseline Schur complement; audit model identifiability')
            delta_c = (baseline_gradient - float(cross @ solved[:, 0])) / schur
            delta_m = solved[:, 0] - solved[:, 1] * delta_c
        else:
            delta_m, delta_c = curvature.solve(gradient), 0.0
        step = settings.initial_step
        while step >= settings.minimum_step:
            candidate_m, candidate_c = m + step * delta_m, c + step * delta_c
            try:
                candidate = score_state(state['covariance'], candidate_m, candidate_c, state['moments'])
                acceptance_key = 'location_objective' if stochastic_fitting else 'objective'
                if candidate[acceptance_key] >= state[acceptance_key] - 1e-12 * (1 + abs(state[acceptance_key])):
                    m, c, state = candidate_m, candidate_c, candidate
                    break
            except FloatingPointError:
                pass
            step *= .5
        noise_change = 0.0
        if settings.fit_noise:
            from scipy.optimize import minimize_scalar
            old_sigma = likelihood.noise_sd
            def noise_objective(log_sigma):
                likelihood.noise_sd = float(np.exp(log_sigma))
                return -float(likelihood.expected(c + evaluation.matmul(m),
                    state['moments']['latent_gp_variance']).value.sum())
            result = minimize_scalar(noise_objective,
                bounds=tuple(np.log(settings.noise_bounds)), method='bounded',
                options={'xatol': 1e-9})
            likelihood.noise_sd = old_sigma
            proposed_sigma = float(np.exp(result.x))
            if result.success:
                likelihood.noise_sd = proposed_sigma
                candidate = score_state(state['covariance'], m, c, state['moments'])
                acceptance_key = 'location_objective' if stochastic_fitting else 'objective'
                if candidate[acceptance_key] >= state[acceptance_key]:
                    noise_change = abs(math.log(proposed_sigma / old_sigma))
                    state = candidate
                else:
                    likelihood.noise_sd = old_sigma
        site_step = settings.initial_step
        information_accepted = False
        merit_before = merit_after = None
        while site_step >= settings.minimum_step:
            target = state['expected'].information
            # Hold the scale fixed within a search: changing it for each trial
            # can hide increases in the moment-equation residual.
            weights = target.clamp_min(1.)
            if stochastic_fitting:
                merit_before = float(((info - target).square() / weights).mean())
            candidate_info = (1 - site_step) * info + site_step * target
            candidate = None
            try:
                candidate = state_for(candidate_info, m, c)
            except FloatingPointError:
                pass
            if settings.variance_method == 'rb' and state['moments']['draws'] != training_draws:
                # A trial that increases the draw budget changes the sampled
                # equations; refresh the anchor before comparing merits.
                state = state_for(info, m, c)
                site_step = settings.initial_step
                continue
            if candidate is not None:
                if stochastic_fitting:
                    merit_after = float(((candidate_info - candidate['expected'].information).square()
                                         / weights).mean())
                    accepted = merit_after <= (1 - 1e-4 * site_step) * merit_before + 1e-24
                else:
                    accepted = candidate['objective'] >= state['objective'] - 1e-12 * (1 + abs(state['objective']))
                if accepted:
                    info, state = candidate_info, candidate
                    information_accepted = True
                    break
            site_step *= .5
        expected = state['expected']
        likelihood_score = evaluation.t_matmul(expected.score) + linear
        prior_score = prior.precision_matmul(m)
        mean_residual = float(torch.linalg.vector_norm(likelihood_score - prior_score)) / max(
            1., float(torch.linalg.vector_norm(likelihood_score)), float(torch.linalg.vector_norm(prior_score)))
        site_residual = float((info - expected.information).abs().max()) / max(
            1., float(expected.information.abs().max()))
        c_residual = abs(float(expected.score.sum()) + event_count - baseline_precision * c) / max(
            1., float(expected.information.sum()), event_count) if settings.fit_baseline else 0.0
        diagnostics = dict(mean_residual=mean_residual, information_residual=site_residual,
            baseline_residual=c_residual, noise_log_change=noise_change,
            joint_elbo=state['objective'], joint_kl=state['kl'],
            logdet_probe_se=state['logdet_se'],
            maximum_relative_mcse=state['max_relative_mcse'],
            exponential_bias_proxy=state['exponential_bias_proxy'])
        history.append(dict(iteration=iteration + 1, **diagnostics,
            mean_step=float(step), information_step=float(site_step),
            information_update_accepted=information_accepted,
            information_merit_before=merit_before, information_merit_after=merit_after,
            elbo_change=None if stochastic_fitting else state['objective'] - previous_objective,
            variance_draws=state['moments']['draws'],
            mean_solver_receipts=curvature.solve_receipts,
            covariance_solver_receipts=state['covariance'].solve_receipts))
        if max(mean_residual, site_residual, c_residual, noise_change) <= settings.stationarity_tolerance:
            converged = True
            termination_reason = 'stationarity_tolerance'
            break
        if stochastic_fitting and not information_accepted:
            termination_reason = 'information_backtracking_failed'
            break
    if stochastic_fitting:
        # The noisy determinant is an end-of-fit diagnostic; it must not veto
        # an update to the Gaussian moment equations.
        state = score_state(state['covariance'], m, c, state['moments'], evaluate_objective=True)
        diagnostics.update(joint_elbo=state['objective'], joint_kl=state['kl'],
                           logdet_probe_se=state['logdet_se'])
    diagnostics.update(iterations=len(history), converged=converged,
        final_objective_evaluated=True, termination_reason=termination_reason,
        information_update_rule='moment_residual_backtracking' if stochastic_fitting else 'joint_elbo_backtracking',
        stationarity_target='estimated_moment_equations' if settings.variance_method == 'rb' else 'gaussian_vi_moment_equations',
        objective_used_for_information_acceptance=not stochastic_fitting,
        objective_kind='joint_elbo' if settings.variance_method == 'exact' and (
            settings.backend == 'dense' or settings.logdet_method == 'dense') else 'estimated_joint_elbo',
        lanczos_truncation_error_certified=False,
        parameter_uncertainty_included=False, noise_sd=getattr(likelihood, 'noise_sd', None))
    fitted = CoupledVIResult(prior, evaluation, likelihood, m.detach().clone(), info.detach().clone(),
        float(c), settings, converged, diagnostics, history, data_metadata or {}, linear,
        event_count=int(event_count), event_log_effort=float(event_log_effort))
    if not converged and settings.failure_policy == 'raise':
        raise VIConvergenceError(fitted)
    return fitted


def fit_tobit_gp(prior, values, *, is_censored, threshold, noise_sd, coordinates=None,
                  evaluation=None, settings=None, quadrature_order=64, baseline=None,
                  measurement_scale='declared_gaussian'):
    """Fit zero-coded left-censoring with explicit flags and Gaussian noise."""
    operator = _resolve_evaluation(prior, coordinates, evaluation)
    likelihood = TobitLikelihood(values, is_censored, threshold=threshold, noise_sd=noise_sd,
        quadrature_order=quadrature_order, device=prior.device, measurement_scale=measurement_scale)
    return fit_coupled_vi(prior, likelihood, operator, settings=settings, baseline=baseline,
        data_metadata=dict(measurement_scale=measurement_scale, censoring='zero_coded_explicit_flags'))


def fit_poisson_gp(prior, counts, *, coordinates=None, evaluation=None, exposure=1.0,
                   exposure_semantics='unit', settings=None, baseline=None):
    """Fit raw counts with all supplied rows, including acquired zeros."""
    operator = _resolve_evaluation(prior, coordinates, evaluation)
    likelihood = PoissonLikelihood(counts, exposure=exposure,
                                  exposure_semantics=exposure_semantics, device=prior.device)
    return fit_coupled_vi(prior, likelihood, operator, settings=settings, baseline=baseline,
        data_metadata=dict(measurement='raw_integer_counts', exposure_semantics=exposure_semantics))


def fit_lgcp(prior, events, quadrature, *, settings=None, baseline=None):
    """Fit original point events and a physical-area spatial integral."""
    quadrature.validate_events(events)
    operator = prior.evaluation(quadrature.coordinates)
    event_operator = prior.evaluation(events.coordinates)
    event_score = event_operator.t_matmul(torch.ones(len(events.coordinates),
        dtype=prior.dtype, device=prior.device))
    likelihood = LGCPIntegral(quadrature.areas, effort=quadrature.effort, device=prior.device)
    fitted = fit_coupled_vi(prior, likelihood, operator, settings=settings, baseline=baseline,
        event_score=event_score, event_count=len(events.coordinates),
        event_log_effort=float(np.log(events.effort).sum()),
        data_metadata=dict(measurement='original_point_events', events_hash=events.fingerprint(),
            quadrature_hash=quadrature.fingerprint(), coordinate_units='micrometers',
            area_units='square_micrometers', observed_area=events.observed_area,
            feature_name=events.feature_name))
    fitted.quadrature = quadrature
    return fitted


def fit_kernel_parameters(problem_factory, initial_parameters, *, bounds, settings=None,
                          max_iterations=20):
    """Deterministic nested profile-ELBO kernel fitting for reference workloads.

    ``problem_factory(parameters)`` returns ``(prior, likelihood, evaluation,
    keyword_arguments)`` for fit_coupled_vi. Use log kernel parameters if desired.
    Every objective call re-converges VI; a final fit always follows the final
    parameter update. Finite differences avoid incorrect gradients through frozen
    operators. This costs multiple fits per step and is not a GPU gradient method.
    """
    from scipy.optimize import minimize
    settings = settings or CoupledVISettings()
    if settings.variance_method != 'exact' or (settings.backend != 'dense' and settings.logdet_method != 'dense'):
        raise ValueError('Reference kernel learning requires deterministic exact marginals and logdet')
    names = list(initial_parameters)
    if set(bounds) != set(names) or not names:
        raise ValueError('Supply matching named initial parameters and bounds')
    history = []
    def fit_at(values):
        parameters = dict(zip(names, map(float, values)))
        prior, likelihood, operator, kwargs = problem_factory(parameters)
        fitted = fit_coupled_vi(prior, likelihood, operator, settings=settings, **kwargs)
        if not fitted.converged:
            raise VIConvergenceError(fitted)
        history.append(dict(parameters=parameters, joint_elbo=fitted.diagnostics['joint_elbo']))
        return fitted
    result = minimize(lambda values: -fit_at(values).diagnostics['joint_elbo'],
        np.asarray([initial_parameters[k] for k in names], float), method='L-BFGS-B',
        bounds=[bounds[k] for k in names], options={'maxiter': max_iterations, 'ftol': 1e-8,
                                                   'eps': 1e-4, 'gtol': 1e-4})
    fitted = fit_at(result.x)
    fitted.hyperparameters = dict(zip(names, map(float, result.x)))
    fitted.hyperparameter_history = history
    fitted.diagnostics['kernel_optimizer'] = dict(success=bool(result.success), message=str(result.message),
        iterations=int(result.get('nit', 0)), evaluations=int(result.nfev), method='nested_profile_elbo_finite_difference',
        final_parameters_reconverged=True)
    if not result.success:
        fitted.converged = False
        fitted.diagnostics['converged'] = False
        if settings.failure_policy == 'raise':
            raise VIConvergenceError(fitted)
    return fitted
