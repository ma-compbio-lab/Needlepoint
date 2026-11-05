"""Shared/private, posterior-expected conditional prototype latent block model.

Feature-to-prototype memberships are frozen in this opt-in model. Exact first
and second sufficient moments retain peak mass, within-prototype heterogeneity
and inherited GP uncertainty. Shared and private views use the SAME domain
responsibilities, but private block parameters and prototypes are section-local.
There is no zero-filled union, imputation, cross-section spatial edge, or assumed
chemical correspondence between private features.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import time
from typing import Callable

import numpy as np
from scipy import sparse
import torch


@dataclass(frozen=True)
class SharedPrivateSettings:
    n_domains: int = 12
    private_strength: float = 0.5
    shared_shrinkage_fraction: float = 1 / 12
    entropy_temperature: float = 1.0
    graph_strength: float = 2.0
    variance_floor: float = 0.01
    max_iterations: int = 120
    min_iterations: int = 5
    tolerance: float = 1e-4
    objective_tolerance: float = 1e-6
    convergence_patience: int = 3
    damping: float = 0.5
    line_search_steps: int = 16

    def validate(self):
        if any(not math.isfinite(value) for value in asdict(self).values()):
            raise ValueError('Settings must be finite')
        if self.n_domains < 2 or self.max_iterations < self.min_iterations:
            raise ValueError('Invalid domain/iteration count')
        if self.min_iterations < 1:
            raise ValueError('At least one iteration required')
        if min(self.private_strength, self.graph_strength, self.shared_shrinkage_fraction) < 0:
            raise ValueError('View, graph and shrinkage strengths must be nonnegative')
        if min(self.entropy_temperature, self.variance_floor, self.tolerance, self.objective_tolerance) <= 0:
            raise ValueError('Temperatures, variance floor and tolerances must be positive')
        if not 0 < self.damping <= 1 or self.convergence_patience < 1:
            raise ValueError('Invalid damping or convergence patience')
        if self.line_search_steps < 1:
            raise ValueError('At least one line-search step required')


@dataclass
class PrototypeMoments:
    first: torch.Tensor
    second: torch.Tensor
    posterior_variance: torch.Tensor
    mass: torch.Tensor
    raw_peak_mass: torch.Tensor

    @property
    def n_spots(self):
        return self.first.shape[0]

    @property
    def n_prototypes(self):
        return self.first.shape[1]


@dataclass
class SharedPrivateSection:
    name: str
    shared: PrototypeMoments
    private: PrototypeMoments | None
    graph: torch.Tensor | None = None


@dataclass
class SharedPrivateResult:
    probabilities: list[torch.Tensor]
    parameters: dict
    history: list[dict]
    converged: bool
    settings: dict
    elapsed_seconds: float


def reliability_feature_weights(shared_reliability, private_reliability, feature_budget=168.0):
    """Balanced feature budgets, preserving absolute private/shared reliability.

The shared budget sums to feature_budget. A private panel's nominal budget is
the same, but its realized budget scales with mean private reliability divided
by mean shared reliability. Multiplicity within an otherwise identical panel
cannot alone increase its influence. private_strength is applied later.
"""
    shared = np.asarray(shared_reliability, dtype=np.float64)
    private = np.asarray(private_reliability, dtype=np.float64)
    for values in (shared, private):
        if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError('Reliability must be a finite vector in [0,1]')
    if not len(shared) or shared.sum() <= 0 or feature_budget <= 0:
        raise ValueError('A positive reliable shared anchor budget is required')
    common_weights = feature_budget * shared / shared.sum()
    private_weights = (feature_budget / len(private) * private / shared.mean()
                       if len(private) else np.empty(0, dtype=float))
    return common_weights, private_weights


def prototype_moments(mean, variance, memberships, feature_weights) -> PrototypeMoments:
    """Exact expected complete-data Gaussian statistics for fixed memberships."""
    if not isinstance(mean, torch.Tensor):
        mean = torch.as_tensor(mean, dtype=torch.float64)
    variance = torch.as_tensor(variance, dtype=mean.dtype, device=mean.device)
    gamma = torch.as_tensor(memberships, dtype=mean.dtype, device=mean.device)
    weights = torch.as_tensor(feature_weights, dtype=mean.dtype, device=mean.device)
    if mean.ndim != 2 or variance.shape != mean.shape or gamma.ndim != 2:
        raise ValueError('Invalid mean, variance or membership shapes')
    if gamma.shape[0] != mean.shape[1] or weights.shape != (mean.shape[1],):
        raise ValueError('Feature/membership/weight axes differ')
    for value in (mean, variance, gamma, weights):
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite input')
    if (variance < 0).any() or (gamma < 0).any() or (weights < 0).any():
        raise ValueError('Negative variance, membership or weight')
    if not torch.allclose(gamma.sum(1), torch.ones_like(weights), atol=1e-5, rtol=1e-5):
        raise ValueError('Membership rows must sum to one')
    effective = gamma * weights[:, None]
    mass = effective.sum(0)
    if (mass <= 0).any():
        raise ValueError('Remove zero-mass prototype columns explicitly before fitting')
    gp_variance = variance @ effective
    return PrototypeMoments(mean @ effective, mean.square() @ effective + gp_variance,
                            gp_variance, mass, gamma.sum(0))


def gaussian_evidence(view: PrototypeMoments, mean, variance):
    """N x R evidence, exactly summing all weighted expected feature errors."""
    inv = variance.reciprocal()
    first = view.first.to(mean.dtype)
    second = view.second.to(mean.dtype)
    mass = view.mass.to(mean.dtype)
    constant = mass @ ((2 * math.pi * variance).log() + mean.square() * inv)
    return -.5 * (second @ inv - 2 * first @ (mean * inv) + constant[None, :])


def _stats(view, responsibilities):
    # Small block sufficient statistics are accumulated in double precision.
    b = responsibilities.to(torch.float64)
    return (b.mean(0), view.first.to(torch.float64).T @ b / b.shape[0],
            view.second.to(torch.float64).T @ b / b.shape[0])


def fit_block_parameters(sections, probabilities, settings):
    """Exact maximizer of the declared shrinkage-penalized Gaussian block loss.

Shared section means are partially pooled; their block variance is shared.
The penalty is tau * mass_q * (mu_dqr-mu_0qr)^2/(2*variance_qr).
This is an explicit regularizer, not a claimed conjugate marginal likelihood.
Each section contributes a unit total observation weight, regardless of N.
"""
    settings.validate()
    common_mass = sections[0].shared.mass.to(torch.float64)
    if len(sections) < 2:
        raise ValueError('At least two sections required')
    for section in sections:
        if not torch.allclose(section.shared.mass.to(torch.float64), common_mass, rtol=1e-5, atol=1e-7):
            raise ValueError('Shared prototype mass and identity must be identical across sections')
    stats = [_stats(section.shared, b) for section, b in zip(sections, probabilities)]
    domain = torch.stack([s[0] for s in stats])
    first = torch.stack([s[1] for s in stats])
    second = torch.stack([s[2] for s in stats])
    tau = settings.shared_shrinkage_fraction
    denom = (domain + tau).clamp_min(1e-15)
    weighted_mean = first / common_mass[None, :, None]
    pooling_weight = domain / denom
    center = (weighted_mean / denom[:, None, :]).sum(0) / pooling_weight.sum(0).clamp_min(1e-15)[None, :]
    means = (weighted_mean + tau * center[None, :, :]) / denom[:, None, :]
    effective_count = common_mass[None, :, None] * domain[:, None, :]
    sse = second - 2 * means * first + means.square() * effective_count
    shrink_sse = tau * common_mass[:, None] * (means - center[None, :, :]).square().sum(0)
    variance = ((sse.sum(0) + shrink_sse) / effective_count.sum(0).clamp_min(1e-15)).clamp_min(settings.variance_floor)
    private = []
    for section, b in zip(sections, probabilities):
        if section.private is None:
            private.append(None)
            continue
        dn, ss, tt = _stats(section.private, b)
        count = section.private.mass.to(torch.float64)[:, None] * dn[None, :]
        mu = ss / count.clamp_min(1e-15)
        var = (tt / count.clamp_min(1e-15) - mu.square()).clamp_min(settings.variance_floor)
        private.append({'mean': mu, 'variance': var})
    return {'shared_center': center, 'shared_mean': means, 'shared_variance': variance,
            'private': private, 'shrinkage_penalty': .5 * (shrink_sse / variance).sum() / len(sections)}


def _graph_apply(graph, b):
    if graph is None:
        return torch.zeros_like(b)
    return torch.sparse.mm(graph, b)


def objective_terms(sections, probabilities, parameters, settings):
    result = {'shared_log_likelihood': 0., 'private_log_likelihood_unscaled': 0.,
              'entropy': 0., 'spatial_agreement': 0.,
              'shrinkage_penalty': float(parameters['shrinkage_penalty'])}
    for d, (section, b) in enumerate(zip(sections, probabilities)):
        common = gaussian_evidence(section.shared, parameters['shared_mean'][d], parameters['shared_variance'])
        result['shared_log_likelihood'] += float((b * common).sum(1).mean()) / len(sections)
        if section.private is not None:
            params = parameters['private'][d]
            local = gaussian_evidence(section.private, params['mean'], params['variance'])
            result['private_log_likelihood_unscaled'] += float((b * local).sum(1).mean()) / len(sections)
        result['entropy'] += float(-(b * b.clamp_min(1e-30).log()).sum(1).mean()) / len(sections)
        result['spatial_agreement'] += float((b * _graph_apply(section.graph, b)).sum(1).mean()) / (2 * len(sections))
    result['objective'] = (result['shared_log_likelihood'] + settings.private_strength * result['private_log_likelihood_unscaled']
        + settings.entropy_temperature * result['entropy'] + settings.graph_strength * result['spatial_agreement']
        - result['shrinkage_penalty'])
    return result


def fit_shared_private_lbm(sections, initial_probabilities, settings=None, callback: Callable | None = None):
    """Coordinate ascent with objective-checked damped spatial mean-field steps."""
    settings = settings or SharedPrivateSettings()
    settings.validate()
    start = time.monotonic()
    if len(initial_probabilities) != len(sections):
        raise ValueError('One initial domain matrix required per section')
    probabilities = []
    for section, initial in zip(sections, initial_probabilities):
        b = torch.as_tensor(initial, device=section.shared.first.device, dtype=torch.float64).clone()
        if b.shape != (section.shared.n_spots, settings.n_domains) or not torch.isfinite(b).all() or (b < 0).any():
            raise ValueError('Invalid initial domain responsibilities')
        if not torch.allclose(b.sum(1), torch.ones(b.shape[0], device=b.device, dtype=b.dtype), atol=1e-5, rtol=1e-5):
            raise ValueError('Domain responsibility rows must sum to one')
        b /= b.sum(1, keepdim=True)
        if section.private is not None and section.private.n_spots != b.shape[0]:
            raise ValueError('Shared and private views have different row support')
        if section.graph is not None and (section.graph.shape != (len(b), len(b)) or section.graph.dtype != b.dtype):
            raise ValueError('Graph shape/dtype mismatch')
        probabilities.append(b)
    history = []
    stable = 0
    converged = False
    previous = None
    for iteration in range(1, settings.max_iterations + 1):
        parameters = fit_block_parameters(sections, probabilities, settings)
        before = objective_terms(sections, probabilities, parameters, settings)
        proposals = []
        for d, (section, b) in enumerate(zip(sections, probabilities)):
            logits = gaussian_evidence(section.shared, parameters['shared_mean'][d], parameters['shared_variance'])
            if section.private is not None and settings.private_strength:
                params = parameters['private'][d]
                logits += settings.private_strength * gaussian_evidence(section.private, params['mean'], params['variance'])
            logits += settings.graph_strength * _graph_apply(section.graph, b)
            proposals.append(torch.softmax(logits / settings.entropy_temperature, dim=1))
        residual = max(float((new-old).abs().max()) for new, old in zip(proposals, probabilities))
        step = settings.damping
        for trial in range(settings.line_search_steps):
            candidate = [old + step * (new-old) for old, new in zip(probabilities, proposals)]
            after = objective_terms(sections, candidate, parameters, settings)
            if after['objective'] >= before['objective'] - 1e-10 * max(1., abs(before['objective'])):
                break
            step *= .5
        else:
            raise RuntimeError('No ascending spatial mean-field step found')
        delta = max(float((new-old).abs().max()) for new, old in zip(candidate, probabilities))
        probabilities = candidate
        improvement = None if previous is None else after['objective'] - previous
        if improvement is not None and improvement < -1e-7 * max(1., abs(previous)):
            raise RuntimeError('Block coordinate objective decreased')
        relative = math.inf if previous is None else abs(improvement) / max(1., abs(previous))
        row = dict(iteration=iteration, **after, max_probability_delta=delta,
                   fixed_point_residual=residual, relative_objective_change=relative if math.isfinite(relative) else None,
                   damping=step, elapsed_seconds=time.monotonic()-start)
        history.append(row)
        if callback:
            callback(row, probabilities)
        stable = stable + 1 if residual <= settings.tolerance and relative <= settings.objective_tolerance else 0
        previous = after['objective']
        if iteration >= settings.min_iterations and stable >= settings.convergence_patience:
            converged = True
            break
    parameters = fit_block_parameters(sections, probabilities, settings)
    return SharedPrivateResult(probabilities, parameters, history, converged, asdict(settings), time.monotonic()-start)


def symmetric_boundary_graph(adjacency, shared: PrototypeMoments, *, dtype=torch.float64, edge_chunk=200000):
    """Fixed shared-view native edges with symmetric degree normalization.

    The topology is supplied explicitly by the caller. Boundary affinities use
    reliability/mass-weighted standardized shared prototype distances only, so
    private-strength comparisons never change the graph as a second variable.
    """
    adjacency = sparse.csr_matrix(adjacency)
    if adjacency.shape != (shared.n_spots, shared.n_spots) or (adjacency-adjacency.T).nnz:
        raise ValueError('A symmetric within-section adjacency is required')
    upper = sparse.triu(adjacency, k=1).tocoo()
    device = shared.first.device
    mean = shared.first / shared.mass[None, :]
    scale = mean.var(0, unbiased=False).clamp_min(1e-12)
    weight = shared.mass / shared.mass.sum()
    distances = []
    for start in range(0, upper.nnz, edge_chunk):
        u = torch.as_tensor(upper.row[start:start+edge_chunk].astype(np.int64), device=device)
        v = torch.as_tensor(upper.col[start:start+edge_chunk].astype(np.int64), device=device)
        distances.append((((mean[u]-mean[v]).square() / scale) * weight).sum(1).cpu().numpy())
    rough = np.concatenate(distances) if distances else np.empty(0)
    bandwidth = max(float(np.median(rough[rough>0])) if (rough>0).any() else 1., 1e-12)
    affinity = np.exp(-.5 * np.minimum(rough/bandwidth, 100.))
    graph = sparse.coo_matrix((np.r_[affinity,affinity],
                             (np.r_[upper.row,upper.col],np.r_[upper.col,upper.row])),shape=adjacency.shape).tocsr()
    degree = np.asarray(graph.sum(1)).ravel()
    norm = np.divide(1., np.sqrt(degree), out=np.zeros_like(degree), where=degree>0)
    graph = sparse.diags(norm) @ graph @ sparse.diags(norm)
    graph = graph.tocsr(); graph.sort_indices()
    tensor = torch.sparse_csr_tensor(torch.as_tensor(graph.indptr.astype(np.int64),device=device),
        torch.as_tensor(graph.indices.astype(np.int64),device=device),
        torch.as_tensor(graph.data,dtype=dtype,device=device), size=graph.shape)
    return tensor, {'normalization':'symmetric D^-1/2 W D^-1/2', 'boundary_view':'shared-only, fixed for every arm',
                    'undirected_edges':int(upper.nnz), 'median_squared_boundary_distance':bandwidth,
                    'isolated_spots':int((degree==0).sum())}


def initialize_from_shared_moments(sections, n_domains=12, seed=0, max_iterations=30):
    """Full-support, section-balanced GPU K-means; mass/reliability geometry."""
    device=sections[0].shared.first.device
    centers_data=[s.shared.first/s.shared.mass[None,:] for s in sections]
    variance=torch.stack([x.var(0,unbiased=False) for x in centers_data]).mean(0).clamp_min(1e-12)
    gp_variance=torch.stack([(s.shared.posterior_variance/s.shared.mass[None,:]).mean(0) for s in sections]).mean(0)
    reliability=variance/(variance+gp_variance).clamp_min(1e-12)
    geometry=sections[0].shared.mass*reliability
    geometry/=geometry.mean().clamp_min(1e-12)
    scale=(geometry/variance).sqrt()
    x=torch.cat([z*scale for z in centers_data]).to(torch.float64)
    weights=torch.cat([torch.full((len(z),),1/(len(z)*len(sections)),device=device,dtype=torch.float64) for z in centers_data])
    generator=torch.Generator(device=device).manual_seed(seed)
    first=int(torch.multinomial(weights,1,generator=generator))
    centers=[x[first]]; minimum=(x-centers[0]).square().sum(1)
    for _ in range(1,n_domains):
        score=minimum*weights
        ix=int(torch.multinomial(score if float(score.sum())>0 else weights,1,generator=generator))
        centers.append(x[ix]); minimum=torch.minimum(minimum,(x-centers[-1]).square().sum(1))
    centers=torch.stack(centers)
    for iteration in range(max_iterations):
        distances=(x.square().sum(1)[:,None]+centers.square().sum(1)[None,:]-2*x@centers.T).clamp_min(0)
        labels=distances.argmin(1)
        sums=torch.zeros_like(centers); counts=torch.zeros(n_domains,device=device,dtype=x.dtype)
        sums.index_add_(0,labels,x*weights[:,None]); counts.index_add_(0,labels,weights)
        updated=torch.where(counts[:,None]>0,sums/counts[:,None].clamp_min(1e-30),centers)
        change=float((updated-centers).abs().max()); centers=updated
        if change<1e-5:
            break
    distances=(x.square().sum(1)[:,None]+centers.square().sum(1)[None,:]-2*x@centers.T).clamp_min(0)
    temperature=max(float(2*distances.min(1).values.median()/x.shape[1]),1e-3)
    b=torch.softmax(-distances/temperature,dim=1)
    counts=[s.shared.n_spots for s in sections]
    return list(torch.split(b,counts)), {'method':'all-spot section-balanced mass-reliability K-means',
        'seed':seed,'iterations':iteration+1,'temperature':temperature,
        'geometry_weight':geometry.cpu().tolist(),'posterior_mean_reliability':reliability.cpu().tolist(),
        'all_spots_used':True,'all_shared_features_represented':True}
