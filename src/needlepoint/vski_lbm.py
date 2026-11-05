from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import load_npz
from scipy.sparse.csgraph import connected_components
from scipy.special import digamma, gammaln
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from needlepoint.graph import (
    SpatialGraphConfig,
    audit_spatial_graph,
    build_spatial_graph,
)
from needlepoint.artifacts import (
    map_file_fingerprints,
    require_production_gp_artifacts,
)
from needlepoint.lbm_compute import TorchLBMCompute


_RAW_CHUNK_FILE_RE = re.compile(r"peak_features_chunk_(\d+)\.npz$")
_RAW_SPOT_ID_RE = re.compile(r"Spot\s+(\d+)$")


@dataclass
class VSKIMapChunk:
    path: Path
    selected_positions: np.ndarray
    local_columns: np.ndarray
    genes: np.ndarray


class SpatialMapStore(Protocol):
    """Minimal streamed-map interface consumed by :class:`VSKILBM`.

    A block is indexed by local spot rows and local feature positions. This
    avoids assuming that every source stores maps in peak-major chunks: fitted
    V-SKI maps are peak-major, while raw TEMI matrices are spot-major.
    """

    metadata: pd.DataFrame
    spots: pd.DataFrame
    genes: np.ndarray
    uses_posterior_variance: bool

    @property
    def n_spots(self) -> int: ...

    @property
    def n_features(self) -> int: ...

    def iter_blocks(
        self,
        *,
        include_variance: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]: ...

    def load_sampled_peak_maps(self, n_spots: int, random_state: int) -> tuple[np.ndarray, np.ndarray]: ...


class ResidualMapStore(SpatialMapStore, Protocol):
    """Spatial map store that can also stream raw-minus-fitted residuals."""

    def iter_residual_blocks(
        self,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]: ...


@dataclass
class VSKILBMResult:
    A: np.ndarray
    B: np.ndarray
    mu: np.ndarray
    var: np.ndarray
    ubar: np.ndarray
    residual_mu: np.ndarray | None
    residual_var: np.ndarray | None
    residual_ubar: np.ndarray | None
    module_maps: np.ndarray
    history: list[dict[str, float]]
    peak_prior: np.ndarray
    prior_labels: np.ndarray | None
    initial_domain_labels: np.ndarray | None
    final_domain_labels: np.ndarray | None
    final_domain_labeling: str
    peak_logit_margin: np.ndarray
    domain_logit_margin: np.ndarray
    effective_likelihood_power_a: float
    effective_likelihood_power_b: float
    domain_mixture_weights: np.ndarray
    domain_expected_log_weights: np.ndarray
    active_domain_mask: np.ndarray
    final_objective_components: dict[str, float]
    domain_model_history: list[dict[str, Any]]
    graph_audit: dict[str, Any]
    metadata: pd.DataFrame
    spots: pd.DataFrame
    prototype_responsibilities: np.ndarray | None = None
    prototype_transition: np.ndarray | None = None
    prototype_peak_prior: np.ndarray | None = None
    initialization_metadata: dict[str, Any] | None = None
    compute_audit: dict[str, Any] | None = None
    feature_likelihood_weights: np.ndarray | None = None
    graph_module_weights: np.ndarray | None = None
    dataset_names: list[str] | None = None
    dataset_mu: np.ndarray | None = None
    dataset_var: np.ndarray | None = None
    dataset_signature_shrinkage: np.ndarray | None = None


@dataclass(frozen=True)
class BlockLikelihoodConfig:
    """Observation model used by the two LBM coordinate updates.

    ``block-average`` preserves the original approximation: posterior
    uncertainty is averaged into each block variance but omitted from the
    entry-level expected squared error. ``posterior-expected`` evaluates the
    Gaussian block likelihood under q(G)=N(F,U), using F^2+U in the evidence.
    ``posterior-residual`` additionally evaluates a tempered raw observation
    correction on R=Y-F from an aligned :class:`ResidualMapStore`.
    """

    model: str = "block-average"
    residual_weight: float = 0.0
    residual_variance: str = "module"

    def __post_init__(self) -> None:
        if self.model not in {"block-average", "posterior-expected", "posterior-residual"}:
            raise ValueError(
                "likelihood_model must be block-average, posterior-expected, or posterior-residual"
            )
        if self.residual_weight < 0.0:
            raise ValueError("residual_weight must be nonnegative")
        if self.model == "posterior-residual" and self.residual_weight <= 0.0:
            raise ValueError("posterior-residual requires residual_weight > 0")
        if self.model != "posterior-residual" and self.residual_weight != 0.0:
            raise ValueError("residual_weight is only valid with posterior-residual")
        if self.residual_variance not in {"module", "block"}:
            raise ValueError("residual_variance must be module or block")

    @property
    def uses_expected_posterior(self) -> bool:
        return self.model in {"posterior-expected", "posterior-residual"}

    @property
    def uses_residual(self) -> bool:
        return self.model == "posterior-residual"

    def evidence_second_moment(self, second: np.ndarray, uncertainty: np.ndarray) -> np.ndarray:
        if self.uses_expected_posterior:
            return second + uncertainty
        return second

    def stabilize_residual_variance(self, var: np.ndarray, domain_mass: np.ndarray) -> np.ndarray:
        if self.residual_variance == "block":
            return var
        weights = np.asarray(domain_mass, dtype=np.float64)
        weights = weights / max(float(weights.sum()), 1.0e-12)
        module_var = np.sum(var * weights[None, :], axis=1)
        return np.broadcast_to(module_var[:, None], var.shape).copy()


@dataclass(frozen=True)
class LikelihoodPowerConfig:
    """Explicit likelihood-power policy for the two LBM coordinate updates."""

    power_a: float = 1.0
    power_b: float = 1.0
    scaling: str = "canonical"
    coupling: str = "independent"
    shared_power: float | None = None

    def __post_init__(self) -> None:
        if self.scaling not in {"canonical", "legacy-average"}:
            raise ValueError("likelihood_scaling must be canonical or legacy-average")
        if self.coupling not in {"independent", "shared", "normalized-free-energy"}:
            raise ValueError(
                "likelihood_power_coupling must be independent, shared, or "
                "normalized-free-energy"
            )
        if self.power_a <= 0.0 or self.power_b <= 0.0:
            raise ValueError("Likelihood powers must be positive")
        if self.coupling == "shared" and (
            self.shared_power is None or self.shared_power <= 0.0
        ):
            raise ValueError("shared likelihood coupling requires shared_likelihood_power > 0")

    def effective(self, *, n_spots: int, n_features: int) -> tuple[float, float]:
        if self.coupling == "shared":
            eta_a = eta_b = float(self.shared_power)
        else:
            eta_a = float(self.power_a)
            eta_b = float(self.power_b)
        if self.scaling == "legacy-average":
            eta_a /= max(int(n_spots), 1)
            eta_b /= max(int(n_features), 1)
        return eta_a, eta_b

    def a_regularizer_scale(self, eta_a: float, eta_b: float) -> float:
        """Scale A entropy/priors into the B-calibrated common objective."""

        if self.coupling != "normalized-free-energy":
            return 1.0
        return float(eta_b / eta_a)


@dataclass(frozen=True)
class DatasetSignatureConfig:
    """Shared or partially pooled dataset-specific Gaussian block signatures."""

    model: str = "shared"
    shrinkage_mass: float | None = 10_000.0

    def __post_init__(self) -> None:
        if self.model not in {"shared", "dataset-shrinkage"}:
            raise ValueError("dataset_signature_model must be shared or dataset-shrinkage")
        if self.shrinkage_mass is not None and self.shrinkage_mass < 0.0:
            raise ValueError("dataset_signature_shrinkage_mass must be nonnegative")


@dataclass(frozen=True)
class PeakPriorScheduleConfig:
    """Annealing policy for the V-SKI peak-module prior."""

    initial_weight: float = 0.5
    minimum_weight: float = 0.0
    decay_tau: float = 5.0

    def __post_init__(self) -> None:
        if self.initial_weight < 0.0 or self.minimum_weight < 0.0:
            raise ValueError("Peak-prior weights must be nonnegative")
        if self.decay_tau < 0.0:
            raise ValueError("peak_prior_decay_tau must be nonnegative")

    def weight(self, iteration: int) -> float:
        if self.decay_tau == 0.0:
            return float(self.initial_weight + self.minimum_weight)
        t = max(int(iteration) - 1, 0)
        return float(self.initial_weight * np.exp(-t / self.decay_tau) + self.minimum_weight)

    def continued_amplitude_after(self, n_iterations: int) -> float:
        if self.decay_tau == 0.0:
            return float(self.initial_weight)
        return float(self.initial_weight * np.exp(-int(n_iterations) / self.decay_tau))


@dataclass(frozen=True)
class PrototypeTransitionConfig:
    """Map spatial prototype families to LBM feature modules."""

    smoothing: float = 1.0
    damping: float = 0.5
    update_during_fit: bool = True
    identity_smoothing: float = 0.05
    peak_smoothing: float = 0.05
    general_initialization_blend: float = 0.5

    def __post_init__(self) -> None:
        if self.smoothing < 0.0:
            raise ValueError("prototype_transition_smoothing must be nonnegative")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("prototype_transition_damping must be in (0, 1]")
        if not 0.0 <= self.identity_smoothing < 1.0:
            raise ValueError("prototype_transition_identity_smoothing must be in [0, 1)")
        if not 0.0 <= self.peak_smoothing < 1.0:
            raise ValueError("prototype_peak_initialization_smoothing must be in [0, 1)")
        if not 0.0 <= self.general_initialization_blend <= 1.0:
            raise ValueError("prototype_general_initialization_blend must be in [0, 1]")

    def identity(self, n_components: int) -> np.ndarray:
        """Return a softly diagonal prototype-to-module transition."""

        epsilon = float(self.identity_smoothing)
        transition = np.full(
            (n_components, n_components),
            epsilon / max(n_components, 1),
            dtype=np.float64,
        )
        transition[np.arange(n_components), np.arange(n_components)] += 1.0 - epsilon
        return transition

    def initialize(
        self,
        gamma: np.ndarray,
        provisional_A: np.ndarray,
        *,
        explicit_A: bool,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Initialize A and T while keeping prototype and LBM roles explicit.

        When prototype families and LBM modules are one-to-one, Gamma directly
        initializes A and T starts near identity. Otherwise, a map-derived
        provisional A identifies T before Gamma T is blended back into A.
        Explicit externally supplied A always takes precedence.
        """

        gamma = np.asarray(gamma, dtype=np.float64)
        provisional_A = np.asarray(provisional_A, dtype=np.float64)
        n_prototypes = gamma.shape[1]
        n_modules = provisional_A.shape[1]
        if explicit_A:
            transition = self.estimate(gamma, provisional_A)
            return provisional_A, transition, "external_A_with_estimated_transition"
        if n_prototypes == n_modules:
            initialized_A = _smooth_responsibilities(gamma, self.peak_smoothing)
            return initialized_A, self.identity(n_modules), "prototype_gamma_direct"

        transition = self.estimate(gamma, provisional_A)
        induced = self.induced_prior(gamma, transition)
        blend = float(self.general_initialization_blend)
        initialized_A = (1.0 - blend) * provisional_A + blend * induced
        initialized_A /= np.maximum(initialized_A.sum(axis=1, keepdims=True), 1.0e-12)
        return initialized_A, transition, "prototype_transition_blend"

    def estimate(self, gamma: np.ndarray, A: np.ndarray) -> np.ndarray:
        counts = np.asarray(gamma, dtype=np.float64).T @ np.asarray(A, dtype=np.float64)
        counts += float(self.smoothing) / max(A.shape[1], 1)
        return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0e-12)

    def update(self, current: np.ndarray, gamma: np.ndarray, A: np.ndarray) -> np.ndarray:
        if not self.update_during_fit:
            return current
        proposed = self.estimate(gamma, A)
        updated = (1.0 - self.damping) * current + self.damping * proposed
        return updated / np.maximum(updated.sum(axis=1, keepdims=True), 1.0e-12)

    @staticmethod
    def induced_prior(gamma: np.ndarray, transition: np.ndarray) -> np.ndarray:
        prior = np.asarray(gamma, dtype=np.float64) @ np.asarray(transition, dtype=np.float64)
        return prior / np.maximum(prior.sum(axis=1, keepdims=True), 1.0e-12)


@dataclass(frozen=True)
class DomainMixtureState:
    weights: np.ndarray
    expected_log_weights: np.ndarray
    objective: float


@dataclass(frozen=True)
class DomainMixtureConfig:
    """Domain proportion prior and active-component policy."""

    prior: str = "none"
    dirichlet_alpha0: float = 1.0
    weight: float = 1.0
    active_weight_threshold: float = 0.005
    active_min_spots: float = 1.0

    def __post_init__(self) -> None:
        if self.prior not in {"none", "dirichlet"}:
            raise ValueError("domain_mixture_prior must be none or dirichlet")
        if self.dirichlet_alpha0 <= 0.0:
            raise ValueError("domain_dirichlet_alpha0 must be positive")
        if self.weight < 0.0:
            raise ValueError("domain_mixture_prior_weight must be nonnegative")
        if not 0.0 <= self.active_weight_threshold < 1.0:
            raise ValueError("active_domain_weight_threshold must be in [0, 1)")
        if self.active_min_spots < 0.0:
            raise ValueError("active_domain_min_spots must be nonnegative")

    def evaluate(self, B: np.ndarray) -> DomainMixtureState:
        weights, expected_log_weights, objective = _domain_mixture_terms(
            B,
            prior=self.prior,
            alpha0=self.dirichlet_alpha0,
        )
        return DomainMixtureState(
            weights=weights,
            expected_log_weights=expected_log_weights,
            objective=float(objective),
        )

    def logit_adjustment(self, state: DomainMixtureState) -> np.ndarray:
        if self.prior != "dirichlet" or self.weight == 0.0:
            return np.zeros_like(state.expected_log_weights)
        return float(self.weight) * state.expected_log_weights

    def weighted_objective(self, state: DomainMixtureState) -> float:
        return float(self.weight * state.objective)

    def active_mask(self, domain_masses: np.ndarray, n_spots: int) -> np.ndarray:
        mass_threshold = max(
            float(self.active_min_spots),
            float(self.active_weight_threshold) * max(int(n_spots), 1),
        )
        return np.asarray(domain_masses, dtype=np.float64) >= mass_threshold


@dataclass(frozen=True)
class DomainUpdateResult:
    responsibilities: np.ndarray
    method: str
    capacity_factor: float
    stats: dict[str, float]


@dataclass(frozen=True)
class DomainUpdateConfig:
    """Spot-domain assignment update policy."""

    method: str = "auto"
    capacity_factor: float = 0.0
    size_penalty_weight: float = 0.0
    hard_capacity_factor: float = 0.0
    hard_smooth: float = 0.02
    capped_softmax_iter: int = 50
    capped_softmax_step: float = 1.0
    capped_softmax_tol: float = 1.0e-3
    final_labeling: str = "argmax"
    final_capacity_factor: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in {"auto", "softmax", "hardcap", "capped-softmax"}:
            raise ValueError("domain_update must be one of auto, softmax, hardcap, capped-softmax")
        if self.final_labeling not in {"argmax", "hardcap"}:
            raise ValueError("final_domain_labeling must be argmax or hardcap")
        if self.hard_smooth < 0.0:
            raise ValueError("domain_hard_smooth must be nonnegative")

    def resolved_update(self) -> str:
        if self.method != "auto":
            return self.method
        if self.hard_capacity_factor > 0.0 or self.capacity_factor > 0.0:
            return "hardcap"
        return "softmax"

    def resolved_capacity_factor(self) -> float:
        if self.capacity_factor > 0.0:
            return float(self.capacity_factor)
        return float(self.hard_capacity_factor)

    def resolved_final_capacity_factor(self) -> float:
        if self.final_capacity_factor > 0.0:
            return float(self.final_capacity_factor)
        factor = self.resolved_capacity_factor()
        return float(factor if factor > 0.0 else 1.0)

    def apply_size_penalty(self, logits: np.ndarray, domain_weights: np.ndarray) -> np.ndarray:
        if self.size_penalty_weight == 0.0:
            return logits
        adjusted = np.asarray(logits, dtype=np.float64, copy=True)
        adjusted -= float(self.size_penalty_weight) * np.log(np.maximum(domain_weights, 1.0e-12))[None, :]
        return adjusted

    def update(
        self,
        logits: np.ndarray,
        *,
        rng: np.random.Generator,
        temperature: float,
        torch_compute: TorchLBMCompute | None = None,
    ) -> DomainUpdateResult:
        method = self.resolved_update()
        if method == "hardcap":
            labels = _capacity_constrained_labels(
                logits,
                capacity_factor=self.resolved_capacity_factor(),
                rng=rng,
            )
            responsibilities = _smoothed_one_hot(labels, logits.shape[1], smooth=self.hard_smooth)
            return DomainUpdateResult(
                responsibilities=responsibilities,
                method=method,
                capacity_factor=self.resolved_capacity_factor(),
                stats={},
            )
        if method == "capped-softmax":
            capped = (
                torch_compute.capped_softmax
                if torch_compute is not None
                else _capped_softmax
            )
            responsibilities, stats = capped(
                logits,
                capacity_factor=self.resolved_capacity_factor(),
                temperature=max(float(temperature), 1.0e-6),
                step=self.capped_softmax_step,
                max_iter=self.capped_softmax_iter,
                tol=self.capped_softmax_tol,
            )
            return DomainUpdateResult(
                responsibilities=responsibilities,
                method=method,
                capacity_factor=self.resolved_capacity_factor(),
                stats=stats,
            )
        responsibilities = (
            torch_compute.row_softmax(
                logits / max(float(temperature), 1.0e-6),
                output_float32=True,
            )
            if torch_compute is not None
            else _row_softmax(logits / max(float(temperature), 1.0e-6))
        )
        return DomainUpdateResult(
            responsibilities=responsibilities,
            method="softmax",
            capacity_factor=0.0,
            stats={},
        )

    def final_labels(self, B: np.ndarray, *, rng: np.random.Generator) -> np.ndarray | None:
        if self.final_labeling != "hardcap":
            return None
        return _capacity_constrained_labels(
            np.log(np.maximum(B, 1.0e-12)),
            capacity_factor=self.resolved_final_capacity_factor(),
            rng=rng,
        )


@dataclass(frozen=True)
class GraphMessageResult:
    message: np.ndarray
    objective: float
    stats: dict[str, float]


@dataclass(frozen=True)
class GraphRegularizationConfig:
    """Graph policy for spot-domain logits.

    The modes deliberately separate spatial continuity from molecular boundary
    handling. ``signed-roughness`` preserves the previous implementation and
    permits repulsive edges. ``boundary-aware`` attenuates edges exponentially
    at module-map boundaries but never makes an edge repulsive.
    """

    model: str = "signed-roughness"
    beta: float = 1.0
    roughness_weight: float = 0.25
    boundary_scale: float = 1.0
    message_normalization: str = "sum"
    edge_batch_size: int = 200_000
    random_state: int = 0

    def __post_init__(self) -> None:
        if self.model not in {"none", "potts", "signed-roughness", "boundary-aware"}:
            raise ValueError("graph_model must be none, potts, signed-roughness, or boundary-aware")
        if self.beta < 0.0:
            raise ValueError("beta must be nonnegative")
        if self.roughness_weight < 0.0:
            raise ValueError("roughness_weight must be nonnegative")
        if self.boundary_scale < 0.0:
            raise ValueError("graph_boundary_scale must be nonnegative")
        if self.edge_batch_size <= 0:
            raise ValueError("edge_batch_size must be positive")
        if self.message_normalization not in {"sum", "neighbor-mean"}:
            raise ValueError("graph message normalization must be sum or neighbor-mean")

    def message(
        self,
        *,
        module_maps: np.ndarray,
        B: np.ndarray,
        edges_u: np.ndarray,
        edges_v: np.ndarray,
        edge_w: np.ndarray,
        n_peak_modules: int,
        module_weights: np.ndarray | None = None,
    ) -> GraphMessageResult:
        if module_weights is None:
            resolved_module_weights = np.ones(n_peak_modules, dtype=np.float64)
        else:
            resolved_module_weights = np.array(
                module_weights, dtype=np.float64, copy=True
            )
            if resolved_module_weights.shape != (n_peak_modules,):
                raise ValueError("graph module weights must match n_peak_modules")
            if not np.isfinite(resolved_module_weights).all() or np.any(
                resolved_module_weights <= 0.0
            ):
                raise ValueError("graph module weights must be finite and positive")
        resolved_module_weights /= resolved_module_weights.sum()
        message = np.zeros_like(B, dtype=np.float64)
        neighbor_mass = np.zeros(B.shape[0], dtype=np.float64)
        empty_stats = {
            "graph_negative_edge_fraction": 0.0,
            "graph_mean_boundary_attenuation": 0.0,
            "graph_mean_normalized_roughness": 0.0,
            "graph_median_normalized_roughness": 0.0,
            "graph_mean_effective_edge_score": 0.0,
            "graph_message_neighbor_normalized": float(
                self.message_normalization == "neighbor-mean"
            ),
        }
        disabled = self.model == "none" or (
            self.model in {"potts", "boundary-aware"} and self.beta == 0.0
        ) or (
            self.model == "signed-roughness"
            and self.beta == 0.0
            and self.roughness_weight == 0.0
        )
        if edges_u.size == 0 or disabled:
            return GraphMessageResult(message=message, objective=0.0, stats=empty_stats)

        uses_roughness = self.model in {"signed-roughness", "boundary-aware"}
        scales = None
        if uses_roughness:
            scales = _edge_roughness_scales(
                module_maps,
                edges_u,
                edges_v,
                random_state=self.random_state,
            )

        objective = 0.0
        negative_edges = 0
        attenuation_sum = 0.0
        roughness_sum = 0.0
        roughness_values: list[np.ndarray] = []
        edge_score_sum = 0.0
        n_edges = 0
        for start in range(0, edges_u.size, self.edge_batch_size):
            stop = min(start + self.edge_batch_size, edges_u.size)
            u = edges_u[start:stop]
            v = edges_v[start:stop]
            weights = edge_w[start:stop].astype(np.float64, copy=False)
            if uses_roughness:
                dz = module_maps[u].astype(np.float64, copy=False) - module_maps[v].astype(
                    np.float64, copy=False
                )
                normalized_roughness = (
                    ((dz * dz) / scales[None, :]) @ resolved_module_weights
                )
                roughness_sum += float(normalized_roughness.sum())
                if sum(value.size for value in roughness_values) < 200_000:
                    remaining = 200_000 - sum(value.size for value in roughness_values)
                    roughness_values.append(normalized_roughness[:remaining].copy())
            else:
                normalized_roughness = np.zeros(weights.size, dtype=np.float64)

            if self.model == "signed-roughness":
                compatibility = self.beta - 0.5 * self.roughness_weight * normalized_roughness
                attenuation = np.ones_like(compatibility)
            elif self.model == "boundary-aware":
                attenuation = np.exp(
                    np.clip(-self.boundary_scale * normalized_roughness, -80.0, 0.0)
                )
                compatibility = self.beta * attenuation
            else:  # pure Potts
                compatibility = np.full(weights.size, self.beta, dtype=np.float64)
                attenuation = np.ones_like(compatibility)

            edge_score = weights * compatibility
            np.add.at(message, u, edge_score[:, None] * B[v])
            np.add.at(message, v, edge_score[:, None] * B[u])
            if self.message_normalization == "neighbor-mean":
                affinity = weights * attenuation
                np.add.at(neighbor_mass, u, affinity)
                np.add.at(neighbor_mass, v, affinity)
            objective += float(np.sum(edge_score[:, None] * B[u] * B[v]))
            negative_edges += int(np.count_nonzero(edge_score < 0.0))
            attenuation_sum += float(attenuation.sum())
            edge_score_sum += float(edge_score.sum())
            n_edges += int(edge_score.size)

        if self.message_normalization == "neighbor-mean":
            message /= np.maximum(neighbor_mass[:, None], 1.0e-12)
            objective = float(0.5 * np.sum(B * message))

        roughness_sample = (
            np.concatenate(roughness_values) if roughness_values else np.zeros(1, dtype=np.float64)
        )
        stats = {
            "graph_negative_edge_fraction": float(negative_edges / max(n_edges, 1)),
            "graph_mean_boundary_attenuation": float(attenuation_sum / max(n_edges, 1)),
            "graph_mean_normalized_roughness": float(roughness_sum / max(n_edges, 1)),
            "graph_median_normalized_roughness": float(np.median(roughness_sample)),
            "graph_mean_effective_edge_score": float(edge_score_sum / max(n_edges, 1)),
            "graph_message_neighbor_normalized": float(
                self.message_normalization == "neighbor-mean"
            ),
        }
        return GraphMessageResult(message=message, objective=objective, stats=stats)


class VSKIMapStore:
    """Chunked reader for V-SKI posterior maps.

    The store keeps V-SKI fitted maps on disk and streams only the requested
    spot rows and peak columns. This is the central scalability constraint for
    TEMI: downstream LBM updates operate on small summaries, while the
    N-by-P posterior maps are read chunk by chunk.
    """

    def __init__(
        self,
        vski_dir: Path,
        *,
        max_spots: int | None = None,
        max_peaks: int | None = None,
        spot_indices: np.ndarray | None = None,
        peak_positions: np.ndarray | None = None,
        selected_models_file: Path | None = None,
        map_standardization: str = "none",
        use_posterior_variance: bool = True,
        require_production_fit: bool = False,
        allow_nonproduction_fit: bool = False,
        allow_legacy_full_coverage: bool = False,
    ) -> None:
        self.vski_dir = Path(vski_dir)
        from needlepoint.posterior_blocks import PosteriorBlocks
        self._posterior_blocks = PosteriorBlocks()
        if not self.vski_dir.exists():
            raise FileNotFoundError(self.vski_dir)
        self.spot_file = self.vski_dir / "vski_gp_map_spots.csv.gz"
        self.metadata_file = self.vski_dir / "vski_gp_model_metadata.csv"
        self.selected_models_file = (
            None if selected_models_file is None else Path(selected_models_file)
        )
        if not self.spot_file.exists():
            raise FileNotFoundError(self.spot_file)
        if not self.metadata_file.exists():
            raise FileNotFoundError(self.metadata_file)

        spots = pd.read_csv(self.spot_file)
        metadata = (
            pd.read_csv(self.metadata_file)
            if self.selected_models_file is None
            else pd.read_csv(self.selected_models_file)
        )
        if self.selected_models_file is not None:
            if not self.selected_models_file.is_file():
                raise FileNotFoundError(self.selected_models_file)
            if "selected" in metadata:
                selected_values = metadata["selected"]
                selected_mask = (
                    selected_values
                    if pd.api.types.is_bool_dtype(selected_values)
                    else selected_values.astype(str).str.lower().isin({"1", "true", "yes"})
                )
                metadata = metadata.loc[selected_mask].copy()
            required_selected = {"model_chunk_path", "map_column", "feature_position", "gene"}
            missing_selected = sorted(required_selected.difference(metadata.columns))
            if missing_selected:
                raise ValueError(
                    f"Selected-model table lacks columns: {missing_selected}"
                )
        if "feature_position" in metadata.columns:
            feature_position = pd.to_numeric(
                metadata["feature_position"], errors="raise"
            ).to_numpy(dtype=np.int64)
            expected = np.arange(metadata.shape[0], dtype=np.int64)
            np.testing.assert_array_equal(
                np.sort(feature_position),
                expected,
                err_msg=(
                    "V-SKI metadata feature_position must be unique and exhaustive "
                    "over 0..n_features-1"
                ),
            )
            # Metadata tables may be sorted for presentation (for example by
            # peak_index) after shard merge.  Model arrays and prototype
            # signatures are defined by feature_position, so restore that
            # canonical order before applying peak_positions or max_peaks.
            metadata = metadata.sort_values(
                "feature_position", kind="stable"
            ).reset_index(drop=True)
        if "gene" not in metadata.columns:
            raise ValueError(f"{self.metadata_file} must include a gene column")
        source_dirs = (
            {Path(path).parent for path in metadata["model_chunk_path"].astype(str)}
            if self.selected_models_file is not None
            else {self.vski_dir}
        )
        self.gp_artifact_audits = require_production_gp_artifacts(
            source_dirs,
            allow_nonproduction=(
                bool(allow_nonproduction_fit) or not bool(require_production_fit)
            ),
            allow_legacy_full_coverage=bool(allow_legacy_full_coverage),
        )

        if spot_indices is not None:
            rows = np.asarray(spot_indices, dtype=np.int64)
        elif max_spots is not None and int(max_spots) > 0:
            rows = np.arange(min(int(max_spots), spots.shape[0]), dtype=np.int64)
        else:
            rows = np.arange(spots.shape[0], dtype=np.int64)
        if rows.ndim != 1 or rows.size == 0:
            raise ValueError("At least one spot row is required")
        if rows.min() < 0 or rows.max() >= spots.shape[0]:
            raise ValueError("spot_indices contain rows outside the V-SKI spot table")
        self.spot_rows = rows
        self.spots = spots.iloc[rows].reset_index(drop=True).copy()

        n_features_total = metadata.shape[0]
        if peak_positions is not None:
            features = np.asarray(peak_positions, dtype=np.int64)
        elif max_peaks is not None and int(max_peaks) > 0:
            features = np.arange(min(int(max_peaks), n_features_total), dtype=np.int64)
        else:
            features = np.arange(n_features_total, dtype=np.int64)
        if features.ndim != 1 or features.size == 0:
            raise ValueError("At least one peak is required")
        if features.min() < 0 or features.max() >= n_features_total:
            raise ValueError("peak_positions contain columns outside V-SKI metadata")
        self.feature_positions = features
        self.metadata = metadata.iloc[features].reset_index(drop=True).copy()
        self.genes = self.metadata["gene"].astype(str).to_numpy()
        self.map_standardization = str(map_standardization)
        if self.map_standardization not in {"none", "zscore"}:
            raise ValueError("map_standardization must be one of: none, zscore")

        if self.selected_models_file is not None:
            reference_columns = [
                column for column in ["spot_index", "x", "y"] if column in spots
            ]
            reference_spots = spots.iloc[self.spot_rows][reference_columns].reset_index(drop=True)
            source_dirs = sorted(
                {Path(path).parent for path in self.metadata["model_chunk_path"].astype(str)}
            )
            for source_dir in source_dirs:
                source_spot_file = source_dir / "vski_gp_map_spots.csv.gz"
                if not source_spot_file.is_file():
                    raise FileNotFoundError(source_spot_file)
                source_spots = pd.read_csv(source_spot_file).iloc[self.spot_rows]
                if not np.array_equal(
                    source_spots[reference_columns].to_numpy(),
                    reference_spots[reference_columns].to_numpy(),
                ):
                    raise ValueError(
                        f"Selected-model spot coordinates/order differ in {source_dir}"
                    )

        self.chunks = self._build_chunk_index()
        if not self.chunks:
            raise FileNotFoundError(f"No V-SKI map chunks found in {self.vski_dir}")
        self.feature_mean_, self.feature_std_ = self._compute_feature_standardization()
        self.uses_posterior_variance = bool(use_posterior_variance)

    @property
    def n_spots(self) -> int:
        return int(self.spot_rows.size)

    @property
    def n_features(self) -> int:
        return int(self.feature_positions.size)

    def _chunk_paths(self) -> list[Path]:
        index_path = self.vski_dir / "vski_gp_map_chunks.csv"
        if index_path.exists():
            table = pd.read_csv(index_path).sort_values("chunk_id")
            return [self.vski_dir / str(name) for name in table["file"]]
        return sorted(self.vski_dir.glob("vski_gp_maps_chunk_*.npz"))

    def _build_chunk_index(self) -> list[VSKIMapChunk]:
        if self.selected_models_file is not None:
            chunks: list[VSKIMapChunk] = []
            for source_path, rows in self.metadata.groupby("model_chunk_path", sort=False):
                path = Path(source_path)
                if not path.is_file():
                    raise FileNotFoundError(path)
                selected_positions = rows.index.to_numpy(dtype=np.int64)
                local_columns = rows["map_column"].to_numpy(dtype=np.int64)
                genes = rows["gene"].astype(str).to_numpy()
                with np.load(path, allow_pickle=True) as npz:
                    if local_columns.max(initial=-1) >= self._posterior_blocks.array(path, "posterior_mean").shape[1]:
                        raise ValueError(f"Selected map_column is outside {path}")
                    source_genes = np.asarray(npz["gene"]).astype(str)[local_columns]
                if not np.array_equal(source_genes, genes):
                    raise ValueError(f"Selected feature IDs do not match map columns in {path}")
                chunks.append(
                    VSKIMapChunk(
                        path=path,
                        selected_positions=selected_positions,
                        local_columns=local_columns,
                        genes=genes,
                    )
                )
            covered = (
                np.sort(np.concatenate([chunk.selected_positions for chunk in chunks]))
                if chunks
                else np.empty(0, dtype=np.int64)
            )
            if not np.array_equal(covered, np.arange(self.n_features, dtype=np.int64)):
                raise ValueError("Selected-model chunks do not cover every selected feature")
            return chunks
        selected_lookup = {int(global_pos): selected_pos for selected_pos, global_pos in enumerate(self.feature_positions)}
        chunk_column_lookup: dict[tuple[str, int], int] = {}
        if {"map_chunk_file", "map_column"}.issubset(self.metadata.columns):
            for selected_pos, row in self.metadata.iterrows():
                if pd.isna(row["map_chunk_file"]) or pd.isna(row["map_column"]):
                    continue
                chunk_column_lookup[(Path(str(row["map_chunk_file"])).name, int(row["map_column"]))] = int(selected_pos)
        chunks: list[VSKIMapChunk] = []
        for path in self._chunk_paths():
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as npz:
                if "feature_position" in npz:
                    feature_position = np.asarray(npz["feature_position"], dtype=np.int64)
                else:
                    genes = np.asarray(npz["gene"]).astype(str)
                    metadata_lookup = self.metadata.reset_index().set_index("gene")["index"].to_dict()
                    feature_position = np.array([metadata_lookup.get(g, -1) for g in genes], dtype=np.int64)
                genes = np.asarray(npz["gene"]).astype(str)
            keep_local: list[int] = []
            keep_selected: list[int] = []
            keep_genes: list[str] = []
            for local_col, global_pos in enumerate(feature_position):
                selected_pos = None
                if chunk_column_lookup:
                    selected_pos = chunk_column_lookup.get((path.name, int(local_col)))
                if selected_pos is None:
                    selected_pos = selected_lookup.get(int(global_pos))
                if selected_pos is None:
                    continue
                keep_local.append(local_col)
                keep_selected.append(selected_pos)
                keep_genes.append(str(genes[local_col]))
            if keep_local:
                chunks.append(
                    VSKIMapChunk(
                        path=path,
                        selected_positions=np.asarray(keep_selected, dtype=np.int64),
                        local_columns=np.asarray(keep_local, dtype=np.int64),
                        genes=np.asarray(keep_genes, dtype=str),
                    )
                )
        covered = np.sort(np.concatenate([chunk.selected_positions for chunk in chunks])) if chunks else np.array([], dtype=np.int64)
        expected = np.arange(self.n_features, dtype=np.int64)
        if not np.array_equal(covered, expected):
            missing = np.setdiff1d(expected, covered)
            raise ValueError(f"V-SKI map chunks do not cover selected peaks; missing positions {missing[:10].tolist()}")
        return chunks

    def iter_arrays(self, *, include_variance: bool = True) -> Iterator[tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]:
        for chunk in self.chunks:
            with np.load(chunk.path, allow_pickle=True) as npz:
                mean = np.asarray(npz["posterior_mean"][np.ix_(self.spot_rows, chunk.local_columns)], dtype=np.float32)
                var = None
                if include_variance and self.uses_posterior_variance:
                    var = np.asarray(npz["posterior_variance"][np.ix_(self.spot_rows, chunk.local_columns)], dtype=np.float32)
            if self.map_standardization == "zscore":
                center = self.feature_mean_[chunk.selected_positions].astype(np.float32, copy=False)
                scale = self.feature_std_[chunk.selected_positions].astype(np.float32, copy=False)
                mean = (mean - center[None, :]) / scale[None, :]
                if var is not None:
                    var = var / (scale[None, :] * scale[None, :])
            if not np.isfinite(mean).all():
                raise ValueError(f"Non-finite GP posterior mean values in {chunk.path}")
            if var is not None:
                if not np.isfinite(var).all():
                    raise ValueError(f"Non-finite GP posterior variance values in {chunk.path}")
                if (var < 0.0).any():
                    raise ValueError(f"Negative GP posterior variance values in {chunk.path}")
            yield mean, var, chunk.selected_positions, chunk.genes

    def iter_blocks(
        self,
        *,
        include_variance: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]:
        """Read only the active row block; never create a full FP32 disk cache."""
        for chunk in self.chunks:
            for start in range(0, self.n_spots, 8192):
                rows = np.arange(start, min(start + 8192, self.n_spots), dtype=np.int64)
                mean = self._posterior_blocks.read(chunk.path, 'posterior_mean',
                    self.spot_rows[rows], chunk.local_columns)
                var = None
                if include_variance and self.uses_posterior_variance:
                    var = self._posterior_blocks.read(chunk.path, 'posterior_variance',
                        self.spot_rows[rows], chunk.local_columns)
                if self.map_standardization == 'zscore':
                    center = self.feature_mean_[chunk.selected_positions].astype(np.float32)
                    scale = self.feature_std_[chunk.selected_positions].astype(np.float32)
                    mean = (mean - center) / scale
                    if var is not None:
                        var = var / (scale * scale)
                if not np.isfinite(mean).all():
                    raise ValueError(f'Non-finite GP posterior mean values in {chunk.path}')
                if var is not None and (not np.isfinite(var).all() or (var < 0).any()):
                    raise ValueError(f'Invalid GP posterior variance in {chunk.path}')
                yield rows, mean, var, chunk.selected_positions, chunk.genes

    def _compute_feature_standardization(self) -> tuple[np.ndarray, np.ndarray]:
        if self.map_standardization == "none":
            return (
                np.zeros(self.n_features, dtype=np.float32),
                np.ones(self.n_features, dtype=np.float32),
            )
        sums = np.zeros(self.n_features, dtype=np.float64)
        sums_sq = np.zeros(self.n_features, dtype=np.float64)
        for chunk in self.chunks:
            for start in range(0, self.n_spots, 8192):
                mean = self._posterior_blocks.read(chunk.path, 'posterior_mean',
                    self.spot_rows[start:start+8192], chunk.local_columns, dtype=np.float64)
                sums[chunk.selected_positions] += mean.sum(axis=0)
                sums_sq[chunk.selected_positions] += np.square(mean).sum(axis=0)
        n = max(self.n_spots, 1)
        feature_mean = sums / n
        feature_var = np.maximum(sums_sq / n - feature_mean * feature_mean, 0.0)
        feature_std = np.sqrt(feature_var)
        feature_std[feature_std < 1.0e-8] = 1.0
        return feature_mean.astype(np.float32), feature_std.astype(np.float32)

    def load_sampled_peak_maps(self, n_spots: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
        return _sample_store_peak_maps(self, n_spots=n_spots, random_state=random_state)


@dataclass(frozen=True)
class RawTEMIMapChunk:
    """One source spot chunk and the selected rows it contributes."""

    matrix_path: Path
    spot_ids_path: Path
    source_rows: np.ndarray
    local_rows: np.ndarray


@dataclass(frozen=True)
class RawTEMIFeatureChunk:
    """Raw on-disk spot chunk before selecting rows and peak columns."""

    chunk_index: int
    matrix_path: Path
    spot_ids_path: Path


class CachedSpatialMapStore:
    """Disk-backed aligned cache for repeated LBM fits over the same map store."""

    def __init__(
        self,
        source: SpatialMapStore,
        cache_dir: Path,
        *,
        feature_block_size: int = 64,
    ) -> None:
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.feature_block_size = int(feature_block_size)
        if self.feature_block_size <= 0:
            raise ValueError("feature_block_size must be positive")
        self.metadata = source.metadata.copy()
        self.spots = source.spots.copy()
        self.genes = np.asarray(source.genes).copy()
        self.uses_posterior_variance = bool(source.uses_posterior_variance)
        self.gp_artifact_audits = getattr(source, "gp_artifact_audits", [])
        self.spot_rows = getattr(
            source,
            "spot_rows",
            np.arange(source.n_spots, dtype=np.int64),
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mean_path = self.cache_dir / "lbm_map_mean.npy"
        self.variance_path = self.cache_dir / "lbm_map_variance.npy"
        self.manifest_path = self.cache_dir / "lbm_map_cache_manifest.json"
        self._prepare()

    @property
    def n_spots(self) -> int:
        return int(self.spots.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.metadata.shape[0])

    def _signature(self) -> dict[str, Any]:
        coordinates = np.ascontiguousarray(
            self.spots[["x", "y"]].to_numpy(dtype=np.float64)
        )
        metadata_columns = [
            column
            for column in ["gene", "feature_position", "model_chunk_path", "map_column"]
            if column in self.metadata
        ]
        metadata_text = self.metadata[metadata_columns].astype(str).to_csv(index=False)
        source_map_files = []
        source_chunks = getattr(self.source, "chunks", None)
        if source_chunks is None and hasattr(self.source, "vski"):
            source_chunks = getattr(self.source.vski, "chunks", None)
        if source_chunks:
            source_map_files = map_file_fingerprints(
                [chunk.path for chunk in source_chunks]
            )
        return {
            "format_version": 2,
            "n_spots": self.n_spots,
            "n_features": self.n_features,
            "uses_posterior_variance": self.uses_posterior_variance,
            "coordinate_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata_text.encode()).hexdigest(),
            "source_type": type(self.source).__name__,
            "map_standardization": getattr(self.source, "map_standardization", None),
            "source_map_files": source_map_files,
        }

    def _prepare(self) -> None:
        expected = self._signature()
        if self.mean_path.is_file() and self.variance_path.is_file() and self.manifest_path.is_file():
            observed = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if observed != expected:
                raise ValueError(
                    f"LBM map cache metadata mismatch in {self.cache_dir}; choose another cache directory"
                )
            self._mean = np.load(self.mean_path, mmap_mode="r")
            self._variance = np.load(self.variance_path, mmap_mode="r")
            if self._mean.shape != (self.n_spots, self.n_features):
                raise ValueError("Cached LBM mean map has an unexpected shape")
            return

        token = f"partial-{os.getpid()}"
        mean_temporary = self.cache_dir / f".{self.mean_path.stem}.{token}.npy"
        variance_temporary = self.cache_dir / f".{self.variance_path.stem}.{token}.npy"
        mean = np.lib.format.open_memmap(
            mean_temporary,
            mode="w+",
            dtype=np.float32,
            shape=(self.n_spots, self.n_features),
        )
        variance = np.lib.format.open_memmap(
            variance_temporary,
            mode="w+",
            dtype=np.float32,
            shape=(self.n_spots, self.n_features),
        )
        counts = np.zeros(self.n_features, dtype=np.int64)
        row_sums = np.zeros(self.n_features, dtype=np.uint64)
        row_square_sums = np.zeros(self.n_features, dtype=np.uint64)
        row_xors = np.zeros(self.n_features, dtype=np.uint64)
        try:
            for rows, block_mean, block_variance, positions, _ in self.source.iter_blocks(
                include_variance=True
            ):
                rows = np.asarray(rows, dtype=np.int64)
                positions = np.asarray(positions, dtype=np.int64)
                block_mean = np.asarray(block_mean, dtype=np.float32)
                if rows.ndim != 1 or positions.ndim != 1:
                    raise ValueError("Source map rows and feature positions must be one-dimensional")
                if np.unique(rows).size != rows.size or np.unique(positions).size != positions.size:
                    raise ValueError("Source map block contains duplicate rows or feature positions")
                if rows.min(initial=0) < 0 or rows.max(initial=-1) >= self.n_spots:
                    raise ValueError("Source map block contains out-of-range rows")
                if positions.min(initial=0) < 0 or positions.max(initial=-1) >= self.n_features:
                    raise ValueError("Source map block contains out-of-range feature positions")
                expected_shape = (rows.size, positions.size)
                if block_mean.shape != expected_shape or not np.isfinite(block_mean).all():
                    raise ValueError("Source map mean block has invalid shape or non-finite values")
                if block_variance is not None:
                    block_variance = np.asarray(block_variance, dtype=np.float32)
                    if (
                        block_variance.shape != expected_shape
                        or not np.isfinite(block_variance).all()
                        or (block_variance < 0.0).any()
                    ):
                        raise ValueError("Source map variance block is invalid")
                mean[np.ix_(rows, positions)] = block_mean
                variance[np.ix_(rows, positions)] = (
                    0.0 if block_variance is None else block_variance
                )
                rows_u = rows.astype(np.uint64, copy=False)
                counts[positions] += rows.size
                row_sums[positions] += rows_u.sum(dtype=np.uint64)
                row_square_sums[positions] += np.square(rows_u).sum(dtype=np.uint64)
                row_xors[positions] ^= np.bitwise_xor.reduce(rows_u, initial=np.uint64(0))
            expected_rows = np.arange(self.n_spots, dtype=np.uint64)
            expected_count = self.n_spots
            expected_sum = expected_rows.sum(dtype=np.uint64)
            expected_square_sum = np.square(expected_rows).sum(dtype=np.uint64)
            expected_xor = np.bitwise_xor.reduce(expected_rows, initial=np.uint64(0))
            complete = (
                (counts == expected_count)
                & (row_sums == expected_sum)
                & (row_square_sums == expected_square_sum)
                & (row_xors == expected_xor)
            )
            if not complete.all():
                raise ValueError(
                    "Source map store did not cover every row exactly once for feature "
                    f"positions {np.flatnonzero(~complete)[:10].tolist()}"
                )
            mean.flush()
            variance.flush()
            if self.manifest_path.exists():
                os.replace(
                    self.manifest_path,
                    self.manifest_path.with_name(self.manifest_path.name + ".previous"),
                )
            os.replace(mean_temporary, self.mean_path)
            os.replace(variance_temporary, self.variance_path)
            temporary_manifest = self.manifest_path.with_name(
                f".{self.manifest_path.name}.{token}"
            )
            temporary_manifest.write_text(json.dumps(expected, indent=2), encoding="utf-8")
            os.replace(temporary_manifest, self.manifest_path)
        finally:
            for path in (mean_temporary, variance_temporary):
                if path.exists():
                    path.unlink()
        self._mean = np.load(self.mean_path, mmap_mode="r")
        self._variance = np.load(self.variance_path, mmap_mode="r")

    def iter_blocks(
        self,
        *,
        include_variance: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]:
        rows = np.arange(self.n_spots, dtype=np.int64)
        for start in range(0, self.n_features, self.feature_block_size):
            stop = min(start + self.feature_block_size, self.n_features)
            positions = np.arange(start, stop, dtype=np.int64)
            block_mean = np.asarray(self._mean[:, start:stop], dtype=np.float32)
            block_variance = (
                np.asarray(self._variance[:, start:stop], dtype=np.float32)
                if include_variance and self.uses_posterior_variance
                else None
            )
            yield rows, block_mean, block_variance, positions, self.genes[start:stop]

    def load_sampled_peak_maps(
        self,
        n_spots: int,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        count = min(int(n_spots), self.n_spots)
        rows = np.sort(rng.choice(self.n_spots, size=count, replace=False))
        return np.asarray(self._mean[rows], dtype=np.float32).T, rows


class RawTEMIMapStore:
    """Stream normalized raw TEMI peak intensities without a dense global matrix.

    The raw feature chunks are spot-major. Each iteration loads one sparse spot
    chunk, applies the same panel-wise TIC normalization used for V-SKI input,
    then emits small dense feature blocks. Raw input has no posterior variance,
    so callers receive ``None`` for the uncertainty block.
    """

    def __init__(
        self,
        feature_dir: Path,
        coords_file: Path,
        peak_metadata: Path,
        *,
        selected_peak_indices: np.ndarray | None = None,
        normalization_peak_indices: np.ndarray | None = None,
        max_spots: int | None = None,
        max_peaks: int | None = None,
        spot_indices: np.ndarray | None = None,
        map_standardization: str = "none",
        target_sum: float = 1.0e4,
        transform: str = "none",
        feature_block_size: int = 64,
    ) -> None:
        self.feature_dir = Path(feature_dir)
        if (self.feature_dir / "chunks").exists():
            self.feature_dir = self.feature_dir / "chunks"
        if not self.feature_dir.exists():
            raise FileNotFoundError(self.feature_dir)
        self.coords_file = Path(coords_file)
        self.peak_metadata_file = Path(peak_metadata)
        if feature_block_size < 1:
            raise ValueError("feature_block_size must be positive")
        self.target_sum = float(target_sum)
        self.transform = str(transform)
        self.feature_block_size = int(feature_block_size)
        self.map_standardization = str(map_standardization)
        if self.map_standardization not in {"none", "zscore"}:
            raise ValueError("map_standardization must be one of: none, zscore")
        if self.transform not in {"none", "sqrt", "log1p"}:
            raise ValueError("transform must be one of: none, sqrt, log1p")

        coords = _load_raw_temi_coordinates(self.coords_file)
        metadata = _load_raw_temi_peak_metadata(self.peak_metadata_file)
        if "peak_index" not in metadata.columns:
            raise ValueError(f"{self.peak_metadata_file} must include peak_index")
        metadata = metadata.drop_duplicates("peak_index").set_index("peak_index", drop=False)

        if spot_indices is not None:
            rows = np.asarray(spot_indices, dtype=np.int64)
        elif max_spots is not None and int(max_spots) > 0:
            rows = np.arange(min(int(max_spots), coords.shape[0]), dtype=np.int64)
        else:
            rows = np.arange(coords.shape[0], dtype=np.int64)
        if rows.ndim != 1 or rows.size == 0:
            raise ValueError("At least one raw TEMI spot is required")
        rows = np.unique(rows)
        if rows.min() < 0 or rows.max() >= coords.shape[0]:
            raise ValueError("spot_indices contain rows outside the coordinate table")
        self.spot_rows = rows
        self.spots = coords.iloc[rows].reset_index(drop=True).copy()
        if "map_row" not in self.spots.columns:
            self.spots.insert(0, "map_row", rows)

        if selected_peak_indices is None:
            peak_ids = metadata.index.to_numpy(dtype=np.int64)
        else:
            peak_ids = np.asarray(selected_peak_indices, dtype=np.int64)
        if peak_ids.ndim != 1 or peak_ids.size == 0:
            raise ValueError("At least one raw TEMI peak is required")
        if max_peaks is not None and int(max_peaks) > 0:
            peak_ids = peak_ids[: min(int(max_peaks), peak_ids.size)]
        if len(np.unique(peak_ids)) != peak_ids.size:
            raise ValueError("selected_peak_indices must be unique")
        missing = np.setdiff1d(peak_ids, metadata.index.to_numpy(dtype=np.int64))
        if missing.size:
            raise ValueError(f"peak metadata is missing requested peak indices, e.g. {missing[:5].tolist()}")
        self.peak_indices = peak_ids
        if normalization_peak_indices is None:
            normalization_peak_ids = peak_ids
        else:
            normalization_peak_ids = np.asarray(normalization_peak_indices, dtype=np.int64)
            if normalization_peak_ids.ndim != 1 or normalization_peak_ids.size == 0:
                raise ValueError("normalization_peak_indices must be a nonempty vector")
            if np.unique(normalization_peak_ids).size != normalization_peak_ids.size:
                raise ValueError("normalization_peak_indices must be unique")
            absent = np.setdiff1d(peak_ids, normalization_peak_ids)
            if absent.size:
                raise ValueError("Selected peaks must be included in normalization_peak_indices")
        self.normalization_peak_indices = normalization_peak_ids
        normalization_lookup = {
            int(peak_index): position
            for position, peak_index in enumerate(self.normalization_peak_indices)
        }
        self.selected_normalization_positions = np.asarray(
            [normalization_lookup[int(peak_index)] for peak_index in self.peak_indices],
            dtype=np.int64,
        )
        self.metadata = metadata.loc[peak_ids].reset_index(drop=True).copy()
        if "gene" not in self.metadata.columns:
            self.metadata.insert(0, "gene", [f"peak_{idx}" for idx in peak_ids])
        else:
            self.metadata["gene"] = self.metadata["gene"].astype(str)
        self.genes = self.metadata["gene"].to_numpy(dtype=str)
        self.uses_posterior_variance = False

        self.chunks = self._build_chunk_index()
        if not self.chunks:
            raise FileNotFoundError(f"No raw TEMI chunks overlap selected spots in {self.feature_dir}")
        self.feature_mean_, self.feature_std_ = self._compute_feature_standardization()

    @property
    def n_spots(self) -> int:
        return int(self.spot_rows.size)

    @property
    def n_features(self) -> int:
        return int(self.peak_indices.size)

    def _build_chunk_index(self) -> list[RawTEMIMapChunk]:
        chunks: list[RawTEMIMapChunk] = []
        for chunk in _discover_raw_temi_chunks(self.feature_dir):
            source_rows = _parse_raw_spot_indices(chunk.spot_ids_path)
            local_rows = np.searchsorted(self.spot_rows, source_rows)
            keep = (local_rows < self.n_spots) & (self.spot_rows[np.minimum(local_rows, self.n_spots - 1)] == source_rows)
            if not keep.any():
                continue
            source_keep = np.flatnonzero(keep)
            local_keep = local_rows[keep]
            order = np.argsort(local_keep, kind="stable")
            chunks.append(
                RawTEMIMapChunk(
                    matrix_path=chunk.matrix_path,
                    spot_ids_path=chunk.spot_ids_path,
                    source_rows=source_keep[order].astype(np.int64, copy=False),
                    local_rows=local_keep[order].astype(np.int64, copy=False),
                )
            )
        return chunks

    def _load_chunk_matrix(self, chunk: RawTEMIMapChunk) -> np.ndarray | sparse.csr_matrix:
        matrix, _, _, _ = self._load_chunk_matrix_with_diagnostics(chunk)
        return matrix

    def _load_chunk_matrix_with_diagnostics(
        self, chunk: RawTEMIMapChunk
    ) -> tuple[sparse.csr_matrix, np.ndarray, int, float]:
        matrix = load_npz(chunk.matrix_path).tocsr()[chunk.source_rows]
        if self.normalization_peak_indices.max(initial=-1) >= matrix.shape[1]:
            raise ValueError(f"{chunk.matrix_path} has fewer columns than requested peak indices")
        normalization_panel = matrix[:, self.normalization_peak_indices].tocsr().astype(
            np.float32, copy=True
        )
        negative = normalization_panel.data < 0.0
        negative_count = int(np.count_nonzero(negative))
        clipped_negative_mass = (
            float(-normalization_panel.data[negative].sum()) if negative_count else 0.0
        )
        if negative_count:
            normalization_panel.data[negative] = 0.0
            normalization_panel.eliminate_zeros()
        normalization_totals = np.asarray(
            normalization_panel.sum(axis=1), dtype=np.float64
        ).ravel()
        normalized = _normalize_raw_temi_csr(
            normalization_panel,
            target_sum=self.target_sum,
            transform=self.transform,
        )
        selected = normalized[:, self.selected_normalization_positions].tocsr()
        return selected, normalization_totals, negative_count, clipped_negative_mass

    def iter_blocks(
        self,
        *,
        include_variance: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]:
        """Yield selected raw map blocks in spot-major input order."""

        del include_variance  # Raw intensities intentionally have no GP uncertainty.
        for chunk in self.chunks:
            matrix = self._load_chunk_matrix(chunk)
            for start in range(0, self.n_features, self.feature_block_size):
                stop = min(start + self.feature_block_size, self.n_features)
                positions = np.arange(start, stop, dtype=np.int64)
                mean = matrix[:, start:stop].toarray().astype(np.float32, copy=False)
                if self.map_standardization == "zscore":
                    center = self.feature_mean_[positions]
                    scale = self.feature_std_[positions]
                    mean = (mean - center[None, :]) / scale[None, :]
                yield chunk.local_rows, mean, None, positions, self.genes[positions]

    def iter_sparse_spot_chunks(
        self,
    ) -> Iterator[tuple[np.ndarray, sparse.csr_matrix, np.ndarray, int, float]]:
        """Yield normalized selected-peak matrices without densifying them.

        This is the spot-chunk counterpart of :meth:`iter_blocks`.  It is
        useful for sparse sufficient statistics that span thousands of raw
        TEMI peaks, where materializing every spot-by-feature block would add
        substantial memory traffic and floating-point work.  Rows are local
        to this store and every selected spot is emitted exactly once.
        """

        if self.map_standardization != "none":
            raise ValueError(
                "Sparse raw chunks are only available with map_standardization='none'"
            )
        for chunk in self.chunks:
            matrix, totals, negative_count, clipped_mass = (
                self._load_chunk_matrix_with_diagnostics(chunk)
            )
            yield (
                chunk.local_rows.copy(),
                matrix,
                totals,
                negative_count,
                clipped_mass,
            )

    def _compute_feature_standardization(self) -> tuple[np.ndarray, np.ndarray]:
        if self.map_standardization == "none":
            return np.zeros(self.n_features, dtype=np.float32), np.ones(self.n_features, dtype=np.float32)
        sums = np.zeros(self.n_features, dtype=np.float64)
        sums_sq = np.zeros(self.n_features, dtype=np.float64)
        for chunk in self.chunks:
            matrix = self._load_chunk_matrix(chunk)
            sums += np.asarray(matrix.sum(axis=0), dtype=np.float64).ravel()
            sums_sq += np.asarray(matrix.multiply(matrix).sum(axis=0), dtype=np.float64).ravel()
        n = max(self.n_spots, 1)
        means = sums / n
        variances = np.maximum(sums_sq / n - means * means, 0.0)
        stds = np.sqrt(variances)
        stds[stds < 1.0e-8] = 1.0
        return means.astype(np.float32), stds.astype(np.float32)

    def load_sampled_peak_maps(self, n_spots: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
        return _sample_store_peak_maps(self, n_spots=n_spots, random_state=random_state)


class PairedVSKIRawMapStore:
    """Align fitted V-SKI maps with normalized raw intensities on disk.

    The raw panel is converted once to the exact standardized scale used to
    train V-SKI, then to the optional downstream map z-score using the *same*
    affine transform as the fitted map. A disk-backed N-by-P array avoids
    materializing the full raw panel in RAM during repeated LBM iterations.
    """

    def __init__(
        self,
        vski_dir: Path,
        *,
        raw_feature_dir: Path,
        coords_file: Path,
        peak_metadata: Path,
        raw_cache_file: Path,
        max_spots: int | None = None,
        max_peaks: int | None = None,
        spot_indices: np.ndarray | None = None,
        peak_positions: np.ndarray | None = None,
        selected_models_file: Path | None = None,
        map_standardization: str = "none",
        use_posterior_variance: bool = True,
        require_production_fit: bool = False,
        allow_nonproduction_fit: bool = False,
        allow_legacy_full_coverage: bool = False,
        target_sum: float = 1.0e4,
        transform: str = "none",
        feature_block_size: int = 64,
    ) -> None:
        if selected_models_file is not None:
            raise ValueError("Paired posterior-residual input does not support selected_models_file")
        self.vski = VSKIMapStore(
            vski_dir,
            max_spots=max_spots,
            max_peaks=max_peaks,
            spot_indices=spot_indices,
            peak_positions=peak_positions,
            map_standardization=map_standardization,
            use_posterior_variance=use_posterior_variance,
            require_production_fit=require_production_fit,
            allow_nonproduction_fit=allow_nonproduction_fit,
            allow_legacy_full_coverage=allow_legacy_full_coverage,
        )
        self.metadata = self.vski.metadata
        self.spots = self.vski.spots
        self.genes = self.vski.genes
        self.uses_posterior_variance = self.vski.uses_posterior_variance
        self.raw_cache_file = Path(raw_cache_file)
        if "peak_index" not in self.metadata.columns:
            raise ValueError("V-SKI metadata must include peak_index for paired raw residuals")
        if not {"standardization_mean", "standardization_std"}.issubset(self.metadata.columns):
            raise ValueError(
                "V-SKI metadata must include training standardization_mean and standardization_std"
            )

        peak_indices = self.metadata["peak_index"].to_numpy(dtype=np.int64)
        full_metadata = pd.read_csv(Path(vski_dir) / "vski_gp_model_metadata.csv")
        if "peak_index" not in full_metadata.columns:
            raise ValueError("Full V-SKI metadata must include peak_index for raw normalization")
        normalization_peak_indices = full_metadata["peak_index"].to_numpy(dtype=np.int64)
        self.raw = RawTEMIMapStore(
            raw_feature_dir,
            coords_file,
            peak_metadata,
            selected_peak_indices=peak_indices,
            normalization_peak_indices=normalization_peak_indices,
            spot_indices=self.vski.spot_rows,
            map_standardization="none",
            target_sum=target_sum,
            transform=transform,
            feature_block_size=feature_block_size,
        )
        if self.raw.n_spots != self.n_spots or self.raw.n_features != self.n_features:
            raise ValueError("Raw and V-SKI map dimensions do not match")
        if not np.allclose(
            self.raw.spots[["x", "y"]].to_numpy(dtype=np.float64),
            self.spots[["x", "y"]].to_numpy(dtype=np.float64),
            rtol=0.0,
            atol=1.0e-6,
        ):
            raise ValueError("Raw and V-SKI spot coordinates are not aligned")

        self._cache_spec = {
            "shape": [self.n_spots, self.n_features],
            "peak_indices": peak_indices.tolist(),
            "normalization_peak_indices": normalization_peak_indices.tolist(),
            "spot_row_first": int(self.vski.spot_rows[0]),
            "spot_row_last": int(self.vski.spot_rows[-1]),
            "spot_rows_sha256": hashlib.sha256(
                self.vski.spot_rows.astype(np.int64, copy=False).tobytes()
            ).hexdigest(),
            "target_sum": float(target_sum),
            "transform": str(transform),
            "map_standardization": str(map_standardization),
            "vski_dir": str(Path(vski_dir).resolve()),
        }
        self._prepare_raw_cache()

    @property
    def n_spots(self) -> int:
        return self.vski.n_spots

    @property
    def n_features(self) -> int:
        return self.vski.n_features

    @property
    def cache_metadata_file(self) -> Path:
        return self.raw_cache_file.with_suffix(self.raw_cache_file.suffix + ".json")

    def _cache_is_valid(self) -> bool:
        if not self.raw_cache_file.exists() or not self.cache_metadata_file.exists():
            return False
        try:
            observed = json.loads(self.cache_metadata_file.read_text(encoding="utf-8"))
            if observed != self._cache_spec:
                return False
            cached = np.load(self.raw_cache_file, mmap_mode="r")
            return cached.shape == (self.n_spots, self.n_features) and cached.dtype == np.float32
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _prepare_raw_cache(self) -> None:
        if self._cache_is_valid():
            return
        self.raw_cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.raw_cache_file.with_suffix(self.raw_cache_file.suffix + ".partial")
        temporary_metadata = self.cache_metadata_file.with_suffix(
            self.cache_metadata_file.suffix + ".partial"
        )
        cache = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(self.n_spots, self.n_features),
        )
        training_mean = self.metadata["standardization_mean"].to_numpy(dtype=np.float32)
        training_std = self.metadata["standardization_std"].to_numpy(dtype=np.float32)
        training_std[training_std < 1.0e-8] = 1.0
        downstream_center = self.vski.feature_mean_.astype(np.float32, copy=False)
        downstream_scale = self.vski.feature_std_.astype(np.float32, copy=False)
        for rows, raw_values, _, positions, _ in self.raw.iter_blocks(include_variance=False):
            values = (raw_values - training_mean[positions][None, :]) / training_std[positions][None, :]
            values = (values - downstream_center[positions][None, :]) / downstream_scale[positions][None, :]
            cache[np.ix_(rows, positions)] = values.astype(np.float32, copy=False)
        cache.flush()
        del cache
        temporary_metadata.write_text(json.dumps(self._cache_spec, indent=2), encoding="utf-8")
        os.replace(temporary, self.raw_cache_file)
        os.replace(temporary_metadata, self.cache_metadata_file)

    def iter_blocks(
        self,
        *,
        include_variance: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]:
        yield from self.vski.iter_blocks(include_variance=include_variance)

    def iter_residual_blocks(
        self,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]:
        raw = np.load(self.raw_cache_file, mmap_mode="r")
        for rows, fitted, variance, positions, genes in self.vski.iter_blocks(include_variance=True):
            residual = np.asarray(raw[np.ix_(rows, positions)], dtype=np.float32) - fitted
            yield rows, residual, variance, positions, genes

    def load_sampled_peak_maps(self, n_spots: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
        return self.vski.load_sampled_peak_maps(n_spots, random_state)


def _sample_store_peak_maps(store: SpatialMapStore, *, n_spots: int, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect a bounded map sample for K-means prior initialization."""

    rng = np.random.default_rng(random_state)
    if n_spots <= 0 or n_spots >= store.n_spots:
        rows = np.arange(store.n_spots, dtype=np.int64)
    else:
        rows = np.sort(rng.choice(store.n_spots, size=int(n_spots), replace=False).astype(np.int64))
    row_to_sample = np.full(store.n_spots, -1, dtype=np.int64)
    row_to_sample[rows] = np.arange(rows.size, dtype=np.int64)
    maps = np.zeros((store.n_features, rows.size), dtype=np.float32)
    for block_rows, mean, _, positions, _ in store.iter_blocks(include_variance=False):
        sample_cols = row_to_sample[block_rows]
        keep = sample_cols >= 0
        if keep.any():
            maps[np.ix_(positions, sample_cols[keep])] = mean[keep].T
    return maps, rows


def _store_spot_weights(store: SpatialMapStore) -> np.ndarray:
    """Return optional observation weights while preserving legacy behavior."""

    values = getattr(store, "spot_weights", None)
    if values is None:
        return np.ones(store.n_spots, dtype=np.float64)
    weights = np.asarray(values, dtype=np.float64)
    if weights.shape != (store.n_spots,):
        raise ValueError(
            f"spot_weights must have shape ({store.n_spots},), got {weights.shape}"
        )
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("spot_weights must be finite and strictly positive")
    return weights


def _metadata_gp_signal_reliability(metadata: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Estimate per-feature latent-signal reliability from fitted GP metadata."""

    candidates = [
        (
            ("joint_global_outputscale", "joint_local_outputscale"),
            "joint_noise_variance",
            "(joint_global_outputscale + joint_local_outputscale) / "
            "(signal + joint_noise_variance)",
        ),
        (
            ("learned_outputscale",),
            "learned_noise_variance",
            "learned_outputscale / (learned_outputscale + learned_noise_variance)",
        ),
    ]
    for signal_columns, noise_column, description in candidates:
        if noise_column not in metadata or not set(signal_columns).issubset(metadata.columns):
            continue
        signal = sum(
            metadata[column].to_numpy(dtype=np.float64) for column in signal_columns
        )
        noise = metadata[noise_column].to_numpy(dtype=np.float64)
        denominator = signal + noise
        valid = (
            np.isfinite(signal)
            & np.isfinite(noise)
            & np.isfinite(denominator)
            & (signal >= 0.0)
            & (noise >= 0.0)
            & (denominator > 0.0)
        )
        reliability = np.ones(metadata.shape[0], dtype=np.float64)
        reliability[valid] = signal[valid] / denominator[valid]
        reliability = np.clip(reliability, 1.0e-6, 1.0)
        return reliability, description
    return np.ones(metadata.shape[0], dtype=np.float64), "unavailable; neutral reliability"


def _feature_gp_signal_reliability(store: SpatialMapStore) -> tuple[np.ndarray, list[str]]:
    """Return aligned feature reliability, averaging joint datasets equally."""

    children = getattr(store, "children", None)
    if children:
        child_values = []
        sources = []
        for index, child in enumerate(children):
            values, child_sources = _feature_gp_signal_reliability(child)
            if values.shape != (store.n_features,):
                raise ValueError("Joint child GP reliability does not match the feature axis")
            child_values.append(values)
            sources.extend(f"dataset_{index}: {source}" for source in child_sources)
        return np.mean(np.vstack(child_values), axis=0), sources

    metadata = getattr(store, "metadata", None)
    if metadata is None or len(metadata) != store.n_features:
        return (
            np.ones(store.n_features, dtype=np.float64),
            ["GP metadata unavailable; neutral reliability"],
        )
    values, source = _metadata_gp_signal_reliability(metadata)
    return values, [source]


def _feature_likelihood_weights(
    store: SpatialMapStore,
    *,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return mean-one fractional-likelihood weights for aligned features."""

    reliability, sources = _feature_gp_signal_reliability(store)
    if mode == "uniform":
        raw = np.ones(store.n_features, dtype=np.float64)
    elif mode == "gp-reliability":
        raw = reliability.astype(np.float64, copy=True)
    else:
        raise ValueError(f"Unsupported feature likelihood weighting: {mode}")
    weights = raw / max(float(np.mean(raw)), 1.0e-12)
    return weights, {
        "mode": mode,
        "normalization": "arithmetic mean one; preserves total feature likelihood mass",
        "sources": sources,
        "gp_signal_reliability": reliability.tolist(),
        "weights_mean_one": weights.tolist(),
    }


def _dataset_slices(store: SpatialMapStore) -> tuple[list[str], list[slice]]:
    offsets = getattr(store, "offsets", None)
    specs = getattr(store, "dataset_specs", None)
    if offsets is None or specs is None:
        return ["shared"], [slice(0, store.n_spots)]
    resolved_offsets = np.asarray(offsets, dtype=np.int64)
    if resolved_offsets.size != len(specs) + 1:
        raise ValueError("Joint dataset offsets do not match dataset specifications")
    return (
        [str(spec.name) for spec in specs],
        [
            slice(int(resolved_offsets[index]), int(resolved_offsets[index + 1]))
            for index in range(len(specs))
        ],
    )


def _domain_initialization_module_weights(
    store: SpatialMapStore,
    A: np.ndarray,
    module_maps: np.ndarray,
    posterior_variance_sums: np.ndarray,
    *,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Weight initial module-map geometry by mass and fitted-GP reliability.

    K-means receives module-map columns multiplied by the square root of the
    returned weights, so each module's squared-distance contribution is linear
    in its soft peak mass and, when requested, its reliability.
    """

    A = np.asarray(A, dtype=np.float64)
    module_maps = np.asarray(module_maps, dtype=np.float64)
    posterior_variance_sums = np.asarray(posterior_variance_sums, dtype=np.float64)
    module_mass = np.maximum(A.sum(axis=0), 1.0e-12)
    feature_reliability, reliability_sources = _feature_gp_signal_reliability(store)
    module_gp_reliability = (A.T @ feature_reliability) / module_mass

    spot_weights = _store_spot_weights(store)
    normalized_spot_weights = spot_weights / spot_weights.sum()
    map_mean = normalized_spot_weights @ module_maps
    signal_variance = normalized_spot_weights @ np.square(module_maps - map_mean[None, :])
    if np.any(posterior_variance_sums > 0.0):
        mean_posterior_variance = normalized_spot_weights @ (
            posterior_variance_sums / module_mass[None, :]
        )
        posterior_reliability = signal_variance / np.maximum(
            signal_variance + mean_posterior_variance,
            1.0e-12,
        )
        posterior_source = (
            "spatial variance of the module posterior mean / "
            "(mean variance + average per-peak posterior variance)"
        )
    else:
        mean_posterior_variance = np.zeros_like(module_mass)
        posterior_reliability = np.ones_like(module_mass)
        posterior_source = "posterior variance unavailable or disabled; neutral reliability"

    module_gp_reliability = np.clip(module_gp_reliability, 1.0e-6, 1.0)
    posterior_reliability = np.clip(posterior_reliability, 1.0e-6, 1.0)
    combined_reliability = module_gp_reliability * posterior_reliability
    if mode == "uniform":
        raw_weight = np.ones_like(module_mass)
    elif mode == "mass":
        raw_weight = module_mass.copy()
    elif mode == "mass-reliability":
        raw_weight = module_mass * combined_reliability
    else:
        raise ValueError(f"Unsupported domain initialization module weighting: {mode}")
    geometry_weight = raw_weight / max(float(np.mean(raw_weight)), 1.0e-12)
    metadata = {
        "mode": mode,
        "distance_semantics": (
            "standardized module-map columns are multiplied by sqrt(geometry_weight); "
            "K-means squared distance is therefore linear in geometry_weight"
        ),
        "mass": module_mass.tolist(),
        "gp_signal_reliability": module_gp_reliability.tolist(),
        "gp_signal_reliability_sources": reliability_sources,
        "posterior_reliability": posterior_reliability.tolist(),
        "posterior_reliability_source": posterior_source,
        "mean_posterior_variance": mean_posterior_variance.tolist(),
        "signal_variance": signal_variance.tolist(),
        "combined_reliability": combined_reliability.tolist(),
        "raw_weight": raw_weight.tolist(),
        "geometry_weight_mean_one": geometry_weight.tolist(),
    }
    return geometry_weight.astype(np.float64), metadata


def _weighted_domain_masses(store: SpatialMapStore, B: np.ndarray) -> np.ndarray:
    """Effective domain masses used by weighted sufficient statistics."""

    return _store_spot_weights(store) @ np.asarray(B, dtype=np.float64)


def _discover_raw_temi_chunks(feature_dir: Path) -> list[RawTEMIFeatureChunk]:
    chunks: list[RawTEMIFeatureChunk] = []
    for matrix_path in feature_dir.glob("peak_features_chunk_*.npz"):
        match = _RAW_CHUNK_FILE_RE.match(matrix_path.name)
        if match is None:
            continue
        chunk_index = int(match.group(1))
        spot_ids_path = feature_dir / f"peak_features_chunk_{chunk_index}_spot_ids.npy"
        if not spot_ids_path.exists():
            raise FileNotFoundError(f"Missing spot-id file for {matrix_path}: {spot_ids_path}")
        chunks.append(RawTEMIFeatureChunk(chunk_index, matrix_path, spot_ids_path))
    chunks.sort(key=lambda item: item.chunk_index)
    if not chunks:
        raise FileNotFoundError(f"No peak_features_chunk_*.npz files found in {feature_dir}")
    return chunks


def _parse_raw_spot_indices(spot_ids_path: Path) -> np.ndarray:
    raw = np.load(spot_ids_path, allow_pickle=True)
    indices = np.empty(raw.shape[0], dtype=np.int64)
    for idx, value in enumerate(raw):
        match = _RAW_SPOT_ID_RE.match(str(value))
        if match is None:
            raise ValueError(f"Could not parse spot id {value!r} from {spot_ids_path}")
        indices[idx] = int(match.group(1)) - 1
    return indices


def _load_raw_temi_coordinates(coords_file: Path) -> pd.DataFrame:
    if not coords_file.exists():
        raise FileNotFoundError(coords_file)
    coords = pd.read_csv(coords_file, sep=";", comment="#")
    required = {"Spot index", "x", "y"}
    missing = required.difference(coords.columns)
    if missing:
        raise ValueError(f"Coordinate file is missing columns: {sorted(missing)}")
    coords = coords.rename(columns={"Spot index": "spot_index"})
    expected = np.arange(coords.shape[0], dtype=np.int64)
    observed = coords["spot_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed, expected):
        raise ValueError("TEMI coordinate rows are not in contiguous spot-index order")
    return coords


def _load_raw_temi_peak_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _normalize_raw_temi_csr(matrix: sparse.spmatrix, *, target_sum: float, transform: str) -> sparse.csr_matrix:
    matrix = matrix.tocsr(copy=True).astype(np.float32)
    row_sums = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32, copy=False)
    if matrix.nnz:
        row_counts = np.diff(matrix.indptr)
        row_ids = np.repeat(np.arange(matrix.shape[0], dtype=np.int32), row_counts)
        scale = np.ones(matrix.shape[0], dtype=np.float32)
        positive = row_sums > 0
        scale[positive] = np.float32(target_sum) / row_sums[positive]
        matrix.data *= scale[row_ids]
        if transform == "sqrt":
            np.sqrt(matrix.data, out=matrix.data)
        elif transform == "log1p":
            np.log1p(matrix.data, out=matrix.data)
        elif transform != "none":
            raise ValueError(f"Unsupported transform: {transform}")
    return matrix


class VSKILBM:
    """Latent block model over fitted V-SKI posterior maps.

    This model jointly updates peak-module responsibilities and spot-domain
    responsibilities from fitted GP posterior means. Posterior variance enters
    the block variance estimate, so uncertain fitted regions contribute less
    decisive likelihood evidence. Spatial coherence is added with a sparse kNN
    graph over spots and does not require dense GP covariance operations.
    """

    def __init__(
        self,
        *,
        n_peak_modules: int = 20,
        n_domains: int = 12,
        graph_neighbors: int = 16,
        graph_construction: str = "auto",
        graph_radius: float | None = None,
        graph_distance_bandwidth: float | None = None,
        lambda_z: float = 0.5,
        lambda_z_min: float = 0.0,
        peak_prior_decay_tau: float = 5.0,
        likelihood_power_a: float = 1.0,
        likelihood_power_b: float = 1.0,
        likelihood_scaling: str = "canonical",
        likelihood_power_coupling: str = "independent",
        shared_likelihood_power: float | None = None,
        evidence_aggregation: str = "sum",
        likelihood_model: str = "block-average",
        residual_weight: float = 0.0,
        residual_variance: str = "module",
        graph_model: str = "signed-roughness",
        beta: float = 1.0,
        roughness_weight: float = 0.25,
        graph_boundary_scale: float = 1.0,
        graph_message_normalization: str = "sum",
        domain_prior_weight: float = 0.0,
        domain_mixture_prior: str = "none",
        domain_dirichlet_alpha0: float = 1.0,
        domain_mixture_prior_weight: float = 1.0,
        active_domain_weight_threshold: float = 0.005,
        active_domain_min_spots: float = 1.0,
        domain_update: str = "auto",
        domain_capacity_factor: float = 0.0,
        domain_size_penalty_weight: float = 0.0,
        domain_hard_capacity_factor: float = 0.0,
        domain_hard_smooth: float = 0.02,
        capped_softmax_iter: int = 50,
        capped_softmax_step: float = 1.0,
        capped_softmax_tol: float = 1.0e-3,
        final_domain_labeling: str = "argmax",
        final_domain_capacity_factor: float = 0.0,
        max_iter: int = 30,
        tol: float = 1.0e-4,
        variance_floor: float = 1.0e-4,
        min_peak_module_mass: float = 1.0,
        min_domain_mass: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_max: float = 0.6,
        a_temperature: float = 1.0,
        b_temperature: float = 1.0,
        edge_batch_size: int = 200_000,
        sample_spots_for_init: int = 60_000,
        peak_init_n_init: int = 20,
        factor_feature_prior_weight: float = 0.0,
        factor_location_prior_weight: float = 0.0,
        factor_feature_prior_temperature: float = 1.0,
        factor_location_prior_temperature: float = 1.0,
        prototype_transition_smoothing: float = 1.0,
        prototype_transition_damping: float = 0.5,
        update_prototype_transition: bool = True,
        prototype_transition_identity_smoothing: float = 0.05,
        prototype_peak_initialization_smoothing: float = 0.05,
        prototype_general_initialization_blend: float = 0.5,
        domain_initialization: str = "soft-kmeans-module-maps",
        domain_initialization_temperature: float | None = None,
        domain_initialization_n_init: int = 20,
        domain_initialization_module_weighting: str = "mass-reliability",
        feature_likelihood_weighting: str = "uniform",
        graph_module_weighting: str = "uniform",
        dataset_signature_model: str = "shared",
        dataset_signature_shrinkage_mass: float | None = 10_000.0,
        compute_backend: str = "numpy",
        compute_device: str = "cuda",
        compute_dtype: str = "float32",
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        self.K = int(n_peak_modules)
        self.R = int(n_domains)
        self.graph_neighbors = int(graph_neighbors)
        self.graph_build_config = SpatialGraphConfig(
            construction=str(graph_construction),
            n_neighbors=int(graph_neighbors),
            radius=graph_radius,
            distance_bandwidth=graph_distance_bandwidth,
            weighted=True,
            row_normalized=False,
        )
        self.peak_prior_schedule = PeakPriorScheduleConfig(
            initial_weight=float(lambda_z),
            minimum_weight=float(lambda_z_min),
            decay_tau=float(peak_prior_decay_tau),
        )
        self.likelihood_config = LikelihoodPowerConfig(
            power_a=float(likelihood_power_a),
            power_b=float(likelihood_power_b),
            scaling=str(likelihood_scaling),
            coupling=str(likelihood_power_coupling),
            shared_power=(
                None if shared_likelihood_power is None else float(shared_likelihood_power)
            ),
        )
        if evidence_aggregation not in {"sum", "mean"}:
            raise ValueError("evidence_aggregation must be sum or mean")
        self.evidence_aggregation = str(evidence_aggregation)
        self.dataset_signature_config = DatasetSignatureConfig(
            model=str(dataset_signature_model),
            shrinkage_mass=(
                None
                if dataset_signature_shrinkage_mass is None
                else float(dataset_signature_shrinkage_mass)
            ),
        )
        self.block_likelihood_config = BlockLikelihoodConfig(
            model=str(likelihood_model),
            residual_weight=float(residual_weight),
            residual_variance=str(residual_variance),
        )
        self.graph_config = GraphRegularizationConfig(
            model=str(graph_model),
            beta=float(beta),
            roughness_weight=float(roughness_weight),
            boundary_scale=float(graph_boundary_scale),
            message_normalization=str(graph_message_normalization),
            edge_batch_size=int(edge_batch_size),
            random_state=int(random_state),
        )
        self.lambda_z = self.peak_prior_schedule.initial_weight
        self.lambda_z_min = self.peak_prior_schedule.minimum_weight
        self.peak_prior_decay_tau = self.peak_prior_schedule.decay_tau
        self.likelihood_power_a = self.likelihood_config.power_a
        self.likelihood_power_b = self.likelihood_config.power_b
        self.likelihood_scaling = self.likelihood_config.scaling
        self.likelihood_power_coupling = self.likelihood_config.coupling
        self.shared_likelihood_power = self.likelihood_config.shared_power
        self.likelihood_model = self.block_likelihood_config.model
        self.residual_weight = self.block_likelihood_config.residual_weight
        self.residual_variance = self.block_likelihood_config.residual_variance
        self.graph_model = self.graph_config.model
        self.beta = self.graph_config.beta
        self.roughness_weight = self.graph_config.roughness_weight
        self.graph_boundary_scale = self.graph_config.boundary_scale
        self.graph_construction = self.graph_build_config.construction
        self.graph_radius = self.graph_build_config.radius
        self.graph_distance_bandwidth = self.graph_build_config.distance_bandwidth
        if (
            self.graph_model != "none"
            and self.graph_build_config.resolved_construction(
                graph_model=self.graph_model
            )
            == "none"
        ):
            raise ValueError(
                "graph_construction=none cannot be combined with graph_model != none"
            )
        self.domain_prior_weight = float(domain_prior_weight)
        self.domain_mixture_config = DomainMixtureConfig(
            prior=str(domain_mixture_prior),
            dirichlet_alpha0=float(domain_dirichlet_alpha0),
            weight=float(domain_mixture_prior_weight),
            active_weight_threshold=float(active_domain_weight_threshold),
            active_min_spots=float(active_domain_min_spots),
        )
        self.domain_mixture_prior = self.domain_mixture_config.prior
        self.domain_dirichlet_alpha0 = self.domain_mixture_config.dirichlet_alpha0
        self.domain_mixture_prior_weight = self.domain_mixture_config.weight
        self.active_domain_weight_threshold = self.domain_mixture_config.active_weight_threshold
        self.active_domain_min_spots = self.domain_mixture_config.active_min_spots
        if self.domain_mixture_prior == "dirichlet" and self.domain_prior_weight != 0.0:
            raise ValueError("domain_prior_weight cannot be combined with the Dirichlet mixture prior")
        self.domain_update_config = DomainUpdateConfig(
            method=str(domain_update),
            capacity_factor=float(domain_capacity_factor),
            size_penalty_weight=float(domain_size_penalty_weight),
            hard_capacity_factor=float(domain_hard_capacity_factor),
            hard_smooth=float(domain_hard_smooth),
            capped_softmax_iter=int(capped_softmax_iter),
            capped_softmax_step=float(capped_softmax_step),
            capped_softmax_tol=float(capped_softmax_tol),
            final_labeling=str(final_domain_labeling),
            final_capacity_factor=float(final_domain_capacity_factor),
        )
        self.domain_update = self.domain_update_config.method
        self.domain_capacity_factor = self.domain_update_config.capacity_factor
        self.domain_size_penalty_weight = self.domain_update_config.size_penalty_weight
        self.domain_hard_capacity_factor = self.domain_update_config.hard_capacity_factor
        self.domain_hard_smooth = self.domain_update_config.hard_smooth
        self.capped_softmax_iter = self.domain_update_config.capped_softmax_iter
        self.capped_softmax_step = self.domain_update_config.capped_softmax_step
        self.capped_softmax_tol = self.domain_update_config.capped_softmax_tol
        self.final_domain_labeling = self.domain_update_config.final_labeling
        self.final_domain_capacity_factor = self.domain_update_config.final_capacity_factor
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.variance_floor = float(variance_floor)
        self.min_peak_module_mass = float(min_peak_module_mass)
        self.min_domain_mass = float(min_domain_mass)
        if self.min_peak_module_mass <= 0.0 or self.min_domain_mass <= 0.0:
            raise ValueError("Minimum effective masses must be positive")
        self.epsilon_min = float(epsilon_min)
        self.epsilon_max = float(epsilon_max)
        self.a_temperature = float(a_temperature)
        self.b_temperature = float(b_temperature)
        if self.a_temperature <= 0.0 or self.b_temperature <= 0.0:
            raise ValueError("Softmax temperatures must be positive")
        self.edge_batch_size = self.graph_config.edge_batch_size
        self.sample_spots_for_init = int(sample_spots_for_init)
        self.peak_init_n_init = int(peak_init_n_init)
        if self.peak_init_n_init <= 0:
            raise ValueError("peak_init_n_init must be positive")
        self.factor_feature_prior_weight = float(factor_feature_prior_weight)
        self.factor_location_prior_weight = float(factor_location_prior_weight)
        self.factor_feature_prior_temperature = float(factor_feature_prior_temperature)
        self.factor_location_prior_temperature = float(factor_location_prior_temperature)
        if self.factor_feature_prior_weight < 0.0 or self.factor_location_prior_weight < 0.0:
            raise ValueError("Factor-space prior weights must be nonnegative")
        if self.factor_feature_prior_temperature <= 0.0 or self.factor_location_prior_temperature <= 0.0:
            raise ValueError("Factor-space prior temperatures must be positive")
        self.prototype_transition_config = PrototypeTransitionConfig(
            smoothing=float(prototype_transition_smoothing),
            damping=float(prototype_transition_damping),
            update_during_fit=bool(update_prototype_transition),
            identity_smoothing=float(prototype_transition_identity_smoothing),
            peak_smoothing=float(prototype_peak_initialization_smoothing),
            general_initialization_blend=float(prototype_general_initialization_blend),
        )
        if domain_initialization not in {
            "soft-kmeans-module-maps",
            "hard-kmeans-module-maps",
        }:
            raise ValueError(
                "domain_initialization must be soft-kmeans-module-maps or "
                "hard-kmeans-module-maps"
            )
        if domain_initialization_temperature is not None and domain_initialization_temperature <= 0.0:
            raise ValueError("domain_initialization_temperature must be positive or None")
        if domain_initialization_n_init <= 0:
            raise ValueError("domain_initialization_n_init must be positive")
        if domain_initialization_module_weighting not in {
            "uniform",
            "mass",
            "mass-reliability",
        }:
            raise ValueError(
                "domain_initialization_module_weighting must be uniform, mass, "
                "or mass-reliability"
            )
        self.domain_initialization = str(domain_initialization)
        self.domain_initialization_temperature = (
            None
            if domain_initialization_temperature is None
            else float(domain_initialization_temperature)
        )
        self.domain_initialization_n_init = int(domain_initialization_n_init)
        self.domain_initialization_module_weighting = str(
            domain_initialization_module_weighting
        )
        if feature_likelihood_weighting not in {"uniform", "gp-reliability"}:
            raise ValueError(
                "feature_likelihood_weighting must be uniform or gp-reliability"
            )
        if graph_module_weighting not in {"uniform", "initialization"}:
            raise ValueError(
                "graph_module_weighting must be uniform or initialization"
            )
        self.feature_likelihood_weighting = str(feature_likelihood_weighting)
        self.graph_module_weighting = str(graph_module_weighting)
        self.dataset_signature_model = self.dataset_signature_config.model
        self.dataset_signature_shrinkage_mass = (
            self.dataset_signature_config.shrinkage_mass
        )
        if compute_backend not in {"numpy", "torch"}:
            raise ValueError("compute_backend must be numpy or torch")
        if compute_dtype not in {"float32", "float64"}:
            raise ValueError("compute_dtype must be float32 or float64")
        self.compute_backend = str(compute_backend)
        self.compute_device = str(compute_device)
        self.compute_dtype = str(compute_dtype)
        self._torch_compute = (
            TorchLBMCompute(device=self.compute_device, dtype=self.compute_dtype)
            if self.compute_backend == "torch"
            else None
        )
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

    def fit(
        self,
        store: SpatialMapStore,
        *,
        peak_prior_labels: np.ndarray | None = None,
        peak_prior_distance: np.ndarray | None = None,
        initial_domain_labels: np.ndarray | None = None,
        initial_peak_responsibilities: np.ndarray | None = None,
        fixed_peak_prior_responsibilities: np.ndarray | None = None,
        initial_domain_responsibilities: np.ndarray | None = None,
        factor_feature_embedding: np.ndarray | None = None,
        factor_location_embedding: np.ndarray | None = None,
        prototype_responsibilities: np.ndarray | None = None,
        precomputed_graph: sparse.spmatrix | None = None,
        persistent_graph_module_weights: np.ndarray | None = None,
    ) -> VSKILBMResult:
        if self.K > store.n_features:
            raise ValueError("n_peak_modules cannot exceed selected peak count")
        if self.R > store.n_spots:
            raise ValueError("n_domains cannot exceed selected spot count")
        if self.block_likelihood_config.uses_residual and not hasattr(store, "iter_residual_blocks"):
            raise ValueError("posterior-residual requires a paired V-SKI/raw residual map store")
        if (
            self.dataset_signature_model == "dataset-shrinkage"
            and self.block_likelihood_config.uses_residual
        ):
            raise ValueError(
                "dataset-shrinkage is not yet compatible with posterior-residual likelihoods"
            )

        factor_feature_embedding = _validate_optional_embedding(
            factor_feature_embedding,
            expected_rows=store.n_features,
            name="factor_feature_embedding",
        )
        factor_location_embedding = _validate_optional_embedding(
            factor_location_embedding,
            expected_rows=store.n_spots,
            name="factor_location_embedding",
        )

        rng = np.random.default_rng(self.random_state)
        prior_labels_aligned = _align_labels_to_count(peak_prior_labels, self.K) if peak_prior_labels is not None else None
        if fixed_peak_prior_responsibilities is not None:
            peak_prior = _validate_responsibilities(
                fixed_peak_prior_responsibilities,
                expected_shape=(store.n_features, self.K),
                name="fixed_peak_prior_responsibilities",
            )
            prior_labels_aligned = peak_prior.argmax(axis=1).astype(np.int64)
        elif initial_peak_responsibilities is not None and prior_labels_aligned is None:
            peak_prior = np.full(
                (store.n_features, self.K),
                1.0 / self.K,
                dtype=np.float32,
            )
        else:
            peak_prior = self._build_peak_prior(store, prior_labels_aligned, peak_prior_distance)
        has_explicit_initial_A = initial_peak_responsibilities is not None
        if initial_peak_responsibilities is None:
            A = peak_prior.copy()
        else:
            A = _validate_responsibilities(
                initial_peak_responsibilities,
                expected_shape=(store.n_features, self.K),
                name="initial_peak_responsibilities",
            )
        prototype_gamma = None
        prototype_transition = None
        peak_initialization_mode = (
            "external_peak_responsibilities"
            if has_explicit_initial_A
            else "sampled_map_kmeans"
        )
        if prototype_responsibilities is not None:
            prototype_gamma = _validate_prototype_responsibilities(
                prototype_responsibilities,
                expected_rows=store.n_features,
            )
            A, prototype_transition, peak_initialization_mode = (
                self.prototype_transition_config.initialize(
                    prototype_gamma,
                    A,
                    explicit_A=has_explicit_initial_A,
                )
            )
            peak_prior = self.prototype_transition_config.induced_prior(
                prototype_gamma,
                prototype_transition,
            )
            prior_labels_aligned = peak_prior.argmax(axis=1).astype(np.int64)
        if initial_domain_responsibilities is None:
            B, initial_domain_aligned, domain_initialization_metadata = self._initialize_domains(
                store,
                A,
                initial_domain_labels,
            )
        else:
            B = _validate_responsibilities(
                initial_domain_responsibilities,
                expected_shape=(store.n_spots, self.R),
                name="initial_domain_responsibilities",
            )
            initial_domain_aligned = B.argmax(axis=1).astype(np.int64)
            domain_initialization_metadata = {
                "mode": "external_domain_responsibilities",
                "temperature": None,
            }

        initialization_metadata = {
            "peak_mode": peak_initialization_mode,
            "prototype_count": (
                None if prototype_gamma is None else int(prototype_gamma.shape[1])
            ),
            "peak_module_count": int(self.K),
            "domain": domain_initialization_metadata,
            "fixed_peak_prior": bool(fixed_peak_prior_responsibilities is not None),
        }
        feature_likelihood_weights, feature_weighting_metadata = (
            _feature_likelihood_weights(
                store,
                mode=self.feature_likelihood_weighting,
            )
        )
        initialization_metadata["feature_likelihood_weighting"] = (
            feature_weighting_metadata
        )
        if persistent_graph_module_weights is not None:
            graph_module_weights = np.asarray(
                persistent_graph_module_weights, dtype=np.float64
            )
            if graph_module_weights.shape != (self.K,):
                raise ValueError(
                    "persistent_graph_module_weights must match n_peak_modules"
                )
            if not np.isfinite(graph_module_weights).all() or np.any(
                graph_module_weights <= 0.0
            ):
                raise ValueError(
                    "persistent_graph_module_weights must be finite and positive"
                )
            graph_module_weights /= np.mean(graph_module_weights)
            graph_weight_source = "external frozen continuation weights"
        elif self.graph_module_weighting == "initialization":
            module_weighting = domain_initialization_metadata.get("module_weighting")
            if module_weighting is None:
                _, _, posterior_variance_sums, initial_module_maps = (
                    self._spot_module_stats(store, A)
                )
                graph_module_weights, module_weighting = (
                    _domain_initialization_module_weights(
                        store,
                        A,
                        initial_module_maps,
                        posterior_variance_sums,
                        mode=self.domain_initialization_module_weighting,
                    )
                )
            else:
                graph_module_weights = np.asarray(
                    module_weighting["geometry_weight_mean_one"], dtype=np.float64
                )
            graph_weight_source = "domain initialization mass/reliability geometry"
        else:
            graph_module_weights = np.ones(self.K, dtype=np.float64)
            graph_weight_source = "uniform"
        initialization_metadata["persistent_graph_module_weighting"] = {
            "mode": self.graph_module_weighting,
            "weights_mean_one": graph_module_weights.tolist(),
            "source": graph_weight_source,
        }

        if precomputed_graph is None:
            graph, graph_audit = self._spatial_graph(store)
        else:
            graph = sparse.csr_matrix(precomputed_graph, dtype=np.float64)
            expected_shape = (store.n_spots, store.n_spots)
            if graph.shape != expected_shape:
                raise ValueError(
                    f"precomputed_graph must have shape {expected_shape}, got {graph.shape}"
                )
            if not np.isfinite(graph.data).all() or np.any(graph.data < 0.0):
                raise ValueError("precomputed_graph must contain finite nonnegative weights")
            effective_graph_construction = (
                self.graph_build_config.resolved_construction(
                    graph_model=self.graph_model
                )
            )
            graph_distance_bandwidth = self.graph_build_config.distance_bandwidth
            if (
                effective_graph_construction == "radius-knn"
                and graph_distance_bandwidth is None
            ):
                graph_distance_bandwidth = float(self.graph_build_config.radius) / 2.0
            graph_audit = audit_spatial_graph(
                graph,
                store.spots[["x", "y"]].to_numpy(dtype=np.float32),
                requested_construction=self.graph_build_config.construction,
                effective_construction=effective_graph_construction,
                distance_bandwidth=graph_distance_bandwidth,
            )
            graph_audit.update(
                {
                    "n_neighbors": int(self.graph_build_config.n_neighbors),
                    "maximum_allowed_edge_distance": (
                        None
                        if self.graph_build_config.radius is None
                        else float(self.graph_build_config.radius)
                    ),
                    "fixed_coordinate_distance_support": bool(
                        effective_graph_construction == "radius-knn"
                    ),
                }
            )
            realized_95th = graph_audit[
                "realized_edge_distance_95th_percentile"
            ]
            graph_audit["radius_coverage_95th_fraction"] = (
                None
                if self.graph_build_config.radius is None
                or realized_95th is None
                else float(realized_95th / self.graph_build_config.radius)
            )
        edges_u, edges_v, edge_w = _undirected_edges(graph)

        history: list[dict[str, float]] = []
        previous_A = A.copy()
        previous_B = B.copy()
        last_mu = np.zeros((self.K, self.R), dtype=np.float32)
        last_var = np.ones((self.K, self.R), dtype=np.float32)
        last_ubar = np.zeros((self.K, self.R), dtype=np.float32)
        last_residual_mu: np.ndarray | None = None
        last_residual_var: np.ndarray | None = None
        last_residual_ubar: np.ndarray | None = None
        module_maps = np.zeros((store.n_spots, self.K), dtype=np.float32)
        effective_eta_a, effective_eta_b = self._effective_likelihood_powers(store)
        observation_evidence_scale = (
            1.0 / max(float(_store_spot_weights(store).sum()), 1.0e-12)
            if self.evidence_aggregation == "mean"
            else 1.0
        )
        feature_evidence_scale = (
            1.0 / max(float(feature_likelihood_weights.sum()), 1.0e-12)
            if self.evidence_aggregation == "mean"
            else 1.0
        )
        initialization_metadata["evidence_aggregation"] = {
            "mode": self.evidence_aggregation,
            "observation_scale": float(observation_evidence_scale),
            "feature_scale": float(feature_evidence_scale),
        }
        a_objective_scale = self.likelihood_config.a_regularizer_scale(
            effective_eta_a, effective_eta_b
        )
        peak_logit_margin = np.zeros(store.n_features, dtype=np.float64)
        domain_logit_margin = np.zeros(store.n_spots, dtype=np.float64)
        domain_expected_log_weights = np.zeros(self.R, dtype=np.float64)
        domain_mixture_weights = np.full(self.R, 1.0 / self.R, dtype=np.float64)
        prototype_transition_change = 0.0
        dataset_names, dataset_slices = _dataset_slices(store)
        if self.dataset_signature_model == "dataset-shrinkage" and len(dataset_slices) < 2:
            raise ValueError("dataset-shrinkage requires a multi-dataset map store")
        last_dataset_mu: np.ndarray | None = None
        last_dataset_var: np.ndarray | None = None
        last_dataset_shrinkage: np.ndarray | None = None

        for iteration in range(1, self.max_iter + 1):
            if self.dataset_signature_model == "dataset-shrinkage":
                stats_by_dataset, masses_by_dataset, _, _ = (
                    self._dataset_feature_domain_stats(store, B)
                )
                stats = {
                    key: np.sum([item[key] for item in stats_by_dataset], axis=0)
                    for key in ("FtB", "F2tB", "UtB")
                }
                likelihood_domain_masses = masses_by_dataset.sum(axis=0)
            else:
                stats = self._feature_domain_stats(store, B)
                likelihood_domain_masses = _weighted_domain_masses(store, B)
                stats_by_dataset = []
                masses_by_dataset = np.empty((0, self.R), dtype=np.float64)
            A_effective = A * feature_likelihood_weights[:, None]
            last_mu, last_var, last_ubar = _block_parameters(
                A_effective,
                stats["FtB"],
                stats["F2tB"],
                stats["UtB"],
                likelihood_domain_masses,
                self.variance_floor,
                self.min_peak_module_mass,
                self.min_domain_mass,
            )
            if self.dataset_signature_model == "dataset-shrinkage":
                last_dataset_mu, last_dataset_var, last_dataset_shrinkage = (
                    self._dataset_block_parameters(
                        A_effective,
                        last_mu,
                        last_var,
                        stats_by_dataset,
                        masses_by_dataset,
                    )
                )
                fitted_peak_evidence = self._dataset_peak_evidence(
                    stats_by_dataset,
                    masses_by_dataset,
                    last_dataset_mu,
                    last_dataset_var,
                )
            else:
                fitted_peak_evidence = _peak_evidence(
                    stats["FtB"],
                    self.block_likelihood_config.evidence_second_moment(
                        stats["F2tB"], stats["UtB"]
                    ),
                    likelihood_domain_masses,
                    last_mu,
                    last_var,
                )
            residual_stats = None
            residual_peak_evidence = np.zeros_like(fitted_peak_evidence)
            if self.block_likelihood_config.uses_residual:
                residual_stats = self._residual_feature_domain_stats(store, B)
                last_residual_mu, last_residual_var, last_residual_ubar = _block_parameters(
                    A_effective,
                    residual_stats["FtB"],
                    residual_stats["F2tB"],
                    residual_stats["UtB"],
                    likelihood_domain_masses,
                    self.variance_floor,
                    self.min_peak_module_mass,
                    self.min_domain_mass,
                )
                last_residual_var = self.block_likelihood_config.stabilize_residual_variance(
                    last_residual_var,
                    likelihood_domain_masses,
                )
                residual_peak_evidence = _peak_evidence(
                    residual_stats["FtB"],
                    residual_stats["F2tB"] + residual_stats["UtB"],
                    likelihood_domain_masses,
                    last_residual_mu,
                    last_residual_var,
                )
            peak_evidence = (
                fitted_peak_evidence
                + self.block_likelihood_config.residual_weight * residual_peak_evidence
            )
            peak_evidence *= feature_likelihood_weights[:, None]
            fitted_peak_evidence *= feature_likelihood_weights[:, None]
            residual_peak_evidence *= feature_likelihood_weights[:, None]
            peak_evidence *= observation_evidence_scale
            fitted_peak_evidence *= observation_evidence_scale
            residual_peak_evidence *= observation_evidence_scale
            peak_prior_weight = self._peak_prior_weight(iteration)
            peak_data_logits = effective_eta_a * peak_evidence
            peak_prior_logits = peak_prior_weight * np.log(np.maximum(peak_prior, 1.0e-12))
            factor_feature_prior_logits = _embedding_prior_logits(
                factor_feature_embedding,
                A,
                weight=self.factor_feature_prior_weight,
                temperature=self.factor_feature_prior_temperature,
            )
            logits_A = peak_data_logits + peak_prior_logits + factor_feature_prior_logits
            peak_logit_margin = self._top_two_margin(logits_A)
            A = self._row_softmax(
                logits_A / max(self.a_temperature, 1.0e-6)
            )
            A_effective = A * feature_likelihood_weights[:, None]
            if prototype_gamma is not None and prototype_transition is not None:
                previous_transition = prototype_transition
                prototype_transition = self.prototype_transition_config.update(
                    prototype_transition,
                    prototype_gamma,
                    A,
                )
                prototype_transition_change = float(
                    np.max(np.abs(prototype_transition - previous_transition))
                )
                peak_prior = self.prototype_transition_config.induced_prior(
                    prototype_gamma,
                    prototype_transition,
                )

            last_mu, last_var, last_ubar = _block_parameters(
                A_effective,
                stats["FtB"],
                stats["F2tB"],
                stats["UtB"],
                likelihood_domain_masses,
                self.variance_floor,
                self.min_peak_module_mass,
                self.min_domain_mass,
            )
            if residual_stats is not None:
                last_residual_mu, last_residual_var, last_residual_ubar = _block_parameters(
                    A_effective,
                    residual_stats["FtB"],
                    residual_stats["F2tB"],
                    residual_stats["UtB"],
                    likelihood_domain_masses,
                    self.variance_floor,
                    self.min_peak_module_mass,
                    self.min_domain_mass,
                )
                last_residual_var = self.block_likelihood_config.stabilize_residual_variance(
                    last_residual_var,
                    likelihood_domain_masses,
                )
            if self.dataset_signature_model == "dataset-shrinkage":
                last_dataset_mu, last_dataset_var, last_dataset_shrinkage = (
                    self._dataset_block_parameters(
                        A_effective,
                        last_mu,
                        last_var,
                        stats_by_dataset,
                        masses_by_dataset,
                    )
                )
            FA, F2A, UA, module_maps = self._spot_module_stats(store, A_effective)
            fitted_second = self.block_likelihood_config.evidence_second_moment(F2A, UA)
            if self.dataset_signature_model == "dataset-shrinkage":
                fitted_spot_evidence = self._dataset_spot_evidence(
                    FA,
                    fitted_second,
                    A_effective.sum(axis=0),
                    last_dataset_mu,
                    last_dataset_var,
                    dataset_slices,
                )
            else:
                fitted_spot_evidence = self._spot_evidence(
                    FA,
                    fitted_second,
                    A_effective.sum(axis=0),
                    last_mu,
                    last_var,
                )
            residual_spot_evidence = np.zeros_like(fitted_spot_evidence)
            if self.block_likelihood_config.uses_residual:
                RA, R2A, RUA = self._residual_spot_module_stats(store, A_effective)
                residual_spot_evidence = self._spot_evidence(
                    RA,
                    R2A + RUA,
                    A_effective.sum(axis=0),
                    last_residual_mu,
                    last_residual_var,
                )
            spot_evidence = (
                fitted_spot_evidence
                + self.block_likelihood_config.residual_weight * residual_spot_evidence
            )
            spot_evidence *= feature_evidence_scale
            fitted_spot_evidence *= feature_evidence_scale
            residual_spot_evidence *= feature_evidence_scale
            graph_result = self._graph_message(
                module_maps,
                B,
                edges_u,
                edges_v,
                edge_w,
                module_weights=graph_module_weights,
            )
            graph_message = graph_result.message
            graph_objective = graph_result.objective
            rho = np.maximum(B.mean(axis=0), 1.0e-8)
            rho = rho / rho.sum()
            domain_mixture_state = self.domain_mixture_config.evaluate(B)
            domain_mixture_weights = domain_mixture_state.weights
            domain_expected_log_weights = domain_mixture_state.expected_log_weights
            domain_data_logits = effective_eta_b * spot_evidence
            logits_B = domain_data_logits + graph_message
            logits_B += self.domain_mixture_config.logit_adjustment(domain_mixture_state)[None, :]
            if self.domain_prior_weight != 0.0:
                logits_B += self.domain_prior_weight * np.log(rho)[None, :]
            factor_location_prior_logits = _embedding_prior_logits(
                factor_location_embedding,
                B,
                weight=self.factor_location_prior_weight,
                temperature=self.factor_location_prior_temperature,
            )
            logits_B += factor_location_prior_logits
            # Optional size penalty: large domains get lower logits, while
            # underused domains become easier to populate.
            logits_B = self.domain_update_config.apply_size_penalty(logits_B, rho)
            domain_logit_margin = self._top_two_margin(logits_B)
            update_result = self.domain_update_config.update(
                logits_B,
                rng=rng,
                temperature=self.b_temperature,
                torch_compute=self._torch_compute,
            )
            B = update_result.responsibilities
            domain_update = update_result.method

            peak_masses = A.sum(axis=0)
            domain_masses = B.sum(axis=0)
            domain_mixture_state = self.domain_mixture_config.evaluate(B)
            domain_mixture_weights = domain_mixture_state.weights
            domain_expected_log_weights = domain_mixture_state.expected_log_weights
            domain_mixture_objective = domain_mixture_state.objective
            active_domain_mask = self._active_domain_mask(domain_masses, store.n_spots)
            data_objective = float(np.sum(A * peak_evidence))
            fitted_data_objective = float(np.sum(A * fitted_peak_evidence))
            residual_data_objective = float(np.sum(A * residual_peak_evidence))
            peak_powered_data_objective = float(effective_eta_a * data_objective)
            domain_powered_data_objective = float(effective_eta_b * data_objective)
            prior_objective = float(peak_prior_weight * np.sum(A * np.log(np.maximum(peak_prior, 1.0e-12))))
            factor_feature_prior_objective = float(np.sum(A * factor_feature_prior_logits))
            factor_location_prior_objective = float(np.sum(B * factor_location_prior_logits))
            entropy_A = _entropy(A)
            entropy_B = _entropy(B)
            delta_A = float(np.mean(np.abs(A - previous_A)))
            delta_B = float(np.mean(np.abs(B - previous_B)))
            objective = (
                domain_powered_data_objective
                + a_objective_scale
                * (
                    prior_objective
                    + factor_feature_prior_objective
                    + self.a_temperature * entropy_A
                )
                + factor_location_prior_objective
                + graph_objective
                + self.domain_mixture_config.weighted_objective(domain_mixture_state)
                + self.b_temperature * entropy_B
            )
            row_entropy_B = _row_entropy(B)
            record = {
                "iteration": int(iteration),
                "objective": objective,
                "data_objective": data_objective,
                "fitted_data_objective": fitted_data_objective,
                "residual_data_objective": residual_data_objective,
                "weighted_residual_data_objective": float(
                    self.block_likelihood_config.residual_weight * residual_data_objective
                ),
                "peak_powered_data_objective": peak_powered_data_objective,
                "domain_powered_data_objective": domain_powered_data_objective,
                "peak_prior_objective": prior_objective,
                "peak_prior_weight": float(peak_prior_weight),
                "prototype_transition_change": float(prototype_transition_change),
                "prototype_prior_median_max_probability": float(
                    np.median(peak_prior.max(axis=1))
                ),
                "prototype_prior_median_normalized_entropy": float(
                    np.median(_normalized_row_entropy(peak_prior))
                ),
                "factor_feature_prior_objective": factor_feature_prior_objective,
                "factor_location_prior_objective": factor_location_prior_objective,
                "effective_likelihood_power_a": float(effective_eta_a),
                "effective_likelihood_power_b": float(effective_eta_b),
                "a_objective_regularizer_scale": float(a_objective_scale),
                "graph_objective": float(graph_objective),
                "domain_mixture_objective": float(domain_mixture_objective),
                "weighted_domain_mixture_objective": float(
                    self.domain_mixture_config.weighted_objective(domain_mixture_state)
                ),
                "entropy_A": entropy_A,
                "entropy_B": entropy_B,
                "delta_A_mean_abs": delta_A,
                "delta_B_mean_abs": delta_B,
                "min_peak_module_size": float(peak_masses.min()),
                "min_domain_size": float(domain_masses.min()),
                "max_domain_size": float(domain_masses.max()),
                "peak_modules_below_min_mass": float(np.sum(peak_masses < self.min_peak_module_mass)),
                "domains_below_min_mass": float(np.sum(domain_masses < self.min_domain_mass)),
                "domain_size_cv": float(np.std(domain_masses) / max(np.mean(domain_masses), 1.0e-8)),
                "effective_domain_count": float(np.exp(_entropy(domain_masses[None, :] / max(B.sum(), 1.0e-8)))),
                "active_domain_count": float(active_domain_mask.sum()),
                "min_active_domain_weight": float(
                    domain_mixture_weights[active_domain_mask].min() if active_domain_mask.any() else 0.0
                ),
                "max_domain_mixture_weight": float(domain_mixture_weights.max()),
                "min_domain_expected_log_weight": float(domain_expected_log_weights.min()),
                "hard_min_domain_size": float(np.bincount(B.argmax(axis=1), minlength=self.R).min()),
                "hard_max_domain_size": float(np.bincount(B.argmax(axis=1), minlength=self.R).max()),
                "median_peak_logit_margin": float(np.median(peak_logit_margin)),
                "median_domain_data_logit_margin": float(
                    np.median(self._top_two_margin(domain_data_logits))
                ),
                "median_domain_graph_abs": float(np.median(np.abs(graph_message))),
                "median_domain_logit_margin": float(np.median(domain_logit_margin)),
                "median_domain_max_probability": float(np.median(B.max(axis=1))),
                "median_domain_normalized_entropy": float(np.median(_normalized_row_entropy(B))),
                "median_domain_effective_candidates": float(np.median(np.exp(row_entropy_B))),
                "domain_update": domain_update,
                "domain_capacity_factor": float(update_result.capacity_factor),
            }
            record.update(graph_result.stats)
            record.update(update_result.stats)
            history.append(record)
            if self.verbose:
                print(
                    f"iter {iteration:03d} obj={objective:.3f} "
                    f"dA={delta_A:.5f} dB={delta_B:.5f}"
                )
            if max(delta_A, delta_B) < self.tol:
                break
            previous_A = A.copy()
            previous_B = B.copy()

        if self.dataset_signature_model == "dataset-shrinkage":
            stats_by_dataset, masses_by_dataset, dataset_names, dataset_slices = (
                self._dataset_feature_domain_stats(store, B)
            )
            stats = {
                key: np.sum([item[key] for item in stats_by_dataset], axis=0)
                for key in ("FtB", "F2tB", "UtB")
            }
            likelihood_domain_masses = masses_by_dataset.sum(axis=0)
        else:
            stats = self._feature_domain_stats(store, B)
            likelihood_domain_masses = _weighted_domain_masses(store, B)
            stats_by_dataset = []
            masses_by_dataset = np.empty((0, self.R), dtype=np.float64)
        A_effective = A * feature_likelihood_weights[:, None]
        last_mu, last_var, last_ubar = _block_parameters(
            A_effective,
            stats["FtB"],
            stats["F2tB"],
            stats["UtB"],
            likelihood_domain_masses,
            self.variance_floor,
            self.min_peak_module_mass,
            self.min_domain_mass,
        )
        _, _, _, module_maps = self._spot_module_stats(store, A_effective)
        if self.dataset_signature_model == "dataset-shrinkage":
            last_dataset_mu, last_dataset_var, last_dataset_shrinkage = (
                self._dataset_block_parameters(
                    A_effective,
                    last_mu,
                    last_var,
                    stats_by_dataset,
                    masses_by_dataset,
                )
            )
            final_fitted_peak_evidence = self._dataset_peak_evidence(
                stats_by_dataset,
                masses_by_dataset,
                last_dataset_mu,
                last_dataset_var,
            )
        else:
            final_fitted_peak_evidence = _peak_evidence(
                stats["FtB"],
                self.block_likelihood_config.evidence_second_moment(
                    stats["F2tB"], stats["UtB"]
                ),
                likelihood_domain_masses,
                last_mu,
                last_var,
            )
        final_residual_peak_evidence = np.zeros_like(final_fitted_peak_evidence)
        if self.block_likelihood_config.uses_residual:
            residual_stats = self._residual_feature_domain_stats(store, B)
            last_residual_mu, last_residual_var, last_residual_ubar = _block_parameters(
                A_effective,
                residual_stats["FtB"],
                residual_stats["F2tB"],
                residual_stats["UtB"],
                likelihood_domain_masses,
                self.variance_floor,
                self.min_peak_module_mass,
                self.min_domain_mass,
            )
            last_residual_var = self.block_likelihood_config.stabilize_residual_variance(
                last_residual_var,
                likelihood_domain_masses,
            )
            final_residual_peak_evidence = _peak_evidence(
                residual_stats["FtB"],
                residual_stats["F2tB"] + residual_stats["UtB"],
                likelihood_domain_masses,
                last_residual_mu,
                last_residual_var,
            )
        final_peak_evidence = (
            final_fitted_peak_evidence
            + self.block_likelihood_config.residual_weight * final_residual_peak_evidence
        )
        final_peak_evidence *= feature_likelihood_weights[:, None]
        final_fitted_peak_evidence *= feature_likelihood_weights[:, None]
        final_residual_peak_evidence *= feature_likelihood_weights[:, None]
        final_peak_evidence *= observation_evidence_scale
        final_fitted_peak_evidence *= observation_evidence_scale
        final_residual_peak_evidence *= observation_evidence_scale
        final_data_objective = float(np.sum(A * final_peak_evidence))
        final_fitted_data_objective = float(np.sum(A * final_fitted_peak_evidence))
        final_residual_data_objective = float(np.sum(A * final_residual_peak_evidence))
        final_graph_result = self._graph_message(
            module_maps,
            B,
            edges_u,
            edges_v,
            edge_w,
            module_weights=graph_module_weights,
        )
        final_graph_objective = final_graph_result.objective
        final_mixture_state = self.domain_mixture_config.evaluate(B)
        domain_mixture_weights = final_mixture_state.weights
        domain_expected_log_weights = final_mixture_state.expected_log_weights
        final_mixture_objective = final_mixture_state.objective
        active_domain_mask = self._active_domain_mask(B.sum(axis=0), store.n_spots)
        final_prior_weight = self._peak_prior_weight(max(len(history), 1))
        final_peak_prior_objective = float(
            final_prior_weight * np.sum(A * np.log(np.maximum(peak_prior, 1.0e-12)))
        )
        final_factor_feature_logits = _embedding_prior_logits(
            factor_feature_embedding,
            A,
            weight=self.factor_feature_prior_weight,
            temperature=self.factor_feature_prior_temperature,
        )
        final_factor_location_logits = _embedding_prior_logits(
            factor_location_embedding,
            B,
            weight=self.factor_location_prior_weight,
            temperature=self.factor_location_prior_temperature,
        )
        final_objective_components = {
            "data_objective": final_data_objective,
            "fitted_data_objective": final_fitted_data_objective,
            "residual_data_objective": final_residual_data_objective,
            "weighted_residual_data_objective": float(
                self.block_likelihood_config.residual_weight * final_residual_data_objective
            ),
            "domain_powered_data_objective": float(effective_eta_b * final_data_objective),
            "peak_prior_objective": final_peak_prior_objective,
            "peak_prior_cross_entropy": float(np.sum(A * np.log(np.maximum(peak_prior, 1.0e-12)))),
            "factor_feature_prior_objective": float(np.sum(A * final_factor_feature_logits)),
            "factor_location_prior_objective": float(np.sum(B * final_factor_location_logits)),
            "graph_objective": float(final_graph_objective),
            "domain_mixture_objective": float(final_mixture_objective),
            "weighted_domain_mixture_objective": float(
                self.domain_mixture_config.weighted_objective(final_mixture_state)
            ),
            "entropy_A": float(_entropy(A)),
            "entropy_B": float(_entropy(B)),
            "active_domain_count": float(active_domain_mask.sum()),
            "a_objective_regularizer_scale": float(a_objective_scale),
        }
        final_objective_components.update(final_graph_result.stats)
        final_objective_components["objective"] = float(
            final_objective_components["domain_powered_data_objective"]
            + a_objective_scale
            * (
                final_objective_components["peak_prior_objective"]
                + final_objective_components["factor_feature_prior_objective"]
                + self.a_temperature * final_objective_components["entropy_A"]
            )
            + final_objective_components["factor_location_prior_objective"]
            + final_objective_components["graph_objective"]
            + final_objective_components["weighted_domain_mixture_objective"]
            + self.b_temperature * final_objective_components["entropy_B"]
        )
        final_domain_labels = self.domain_update_config.final_labels(B, rng=rng)
        if prototype_gamma is not None:
            prior_labels_aligned = peak_prior.argmax(axis=1).astype(np.int64)
        return VSKILBMResult(
            A=A.astype(np.float32),
            B=B.astype(np.float32),
            mu=last_mu.astype(np.float32),
            var=last_var.astype(np.float32),
            ubar=last_ubar.astype(np.float32),
            residual_mu=(
                None if last_residual_mu is None else last_residual_mu.astype(np.float32)
            ),
            residual_var=(
                None if last_residual_var is None else last_residual_var.astype(np.float32)
            ),
            residual_ubar=(
                None if last_residual_ubar is None else last_residual_ubar.astype(np.float32)
            ),
            module_maps=module_maps.astype(np.float32),
            history=history,
            peak_prior=peak_prior.astype(np.float32),
            prior_labels=prior_labels_aligned,
            initial_domain_labels=initial_domain_aligned,
            final_domain_labels=final_domain_labels,
            final_domain_labeling=self.final_domain_labeling,
            peak_logit_margin=peak_logit_margin.astype(np.float32),
            domain_logit_margin=domain_logit_margin.astype(np.float32),
            effective_likelihood_power_a=float(effective_eta_a),
            effective_likelihood_power_b=float(effective_eta_b),
            domain_mixture_weights=domain_mixture_weights.astype(np.float32),
            domain_expected_log_weights=domain_expected_log_weights.astype(np.float32),
            active_domain_mask=active_domain_mask,
            final_objective_components=final_objective_components,
            domain_model_history=[],
            graph_audit=graph_audit,
            metadata=store.metadata.copy(),
            spots=store.spots.copy(),
            prototype_responsibilities=(
                None if prototype_gamma is None else prototype_gamma.astype(np.float32)
            ),
            prototype_transition=(
                None if prototype_transition is None else prototype_transition.astype(np.float32)
            ),
            prototype_peak_prior=(
                None if prototype_gamma is None else peak_prior.astype(np.float32)
            ),
            initialization_metadata=initialization_metadata,
            feature_likelihood_weights=feature_likelihood_weights.astype(np.float32),
            graph_module_weights=graph_module_weights.astype(np.float32),
            dataset_names=(
                dataset_names if self.dataset_signature_model == "dataset-shrinkage" else None
            ),
            dataset_mu=(
                None if last_dataset_mu is None else last_dataset_mu.astype(np.float32)
            ),
            dataset_var=(
                None if last_dataset_var is None else last_dataset_var.astype(np.float32)
            ),
            dataset_signature_shrinkage=(
                None
                if last_dataset_shrinkage is None
                else last_dataset_shrinkage.astype(np.float32)
            ),
            compute_audit=(
                {
                    "backend": "numpy",
                    "device": "cpu",
                    "dtype": "float64",
                    "peak_device_memory_bytes": 0,
                    "gpu_accelerated": False,
                    "hardcap_update_backend": "numpy",
                    "graph_message_backend": "numpy",
                }
                if self._torch_compute is None
                else self._torch_compute.audit()
            ),
        )

    def _effective_likelihood_powers(self, store: SpatialMapStore) -> tuple[float, float]:
        """Return explicit powers multiplying the two likelihood updates."""

        return self.likelihood_config.effective(n_spots=store.n_spots, n_features=store.n_features)

    def _peak_prior_weight(self, iteration: int) -> float:
        """Anneal the V-SKI prior while retaining an optional nonzero floor."""

        return self.peak_prior_schedule.weight(iteration)

    def _active_domain_mask(self, domain_masses: np.ndarray, n_spots: int) -> np.ndarray:
        """Mark components with enough posterior mass to be scientifically active."""

        return self.domain_mixture_config.active_mask(domain_masses, n_spots)

    def _spatial_graph(
        self,
        store: SpatialMapStore,
    ) -> tuple[sparse.csr_matrix, dict[str, Any]]:
        return build_spatial_graph(
            store.spots[["x", "y"]].to_numpy(dtype=np.float32),
            config=self.graph_build_config,
            graph_model=self.graph_model,
        )

    def _build_peak_prior(
        self,
        store: SpatialMapStore,
        labels: np.ndarray | None,
        distance: np.ndarray | None,
    ) -> np.ndarray:
        if labels is None:
            maps, _ = store.load_sampled_peak_maps(self.sample_spots_for_init, self.random_state)
            labels = KMeans(
                n_clusters=self.K,
                n_init=self.peak_init_n_init,
                random_state=self.random_state,
            ).fit_predict(_zscore_rows(maps))
            confidence = np.full(store.n_features, 0.5, dtype=np.float32)
        else:
            labels = np.asarray(labels, dtype=np.int64)
            if labels.size != store.n_features:
                raise ValueError("peak_prior_labels length must match selected peak count")
            if distance is None:
                confidence = np.full(store.n_features, 0.75, dtype=np.float32)
            else:
                dist = np.asarray(distance, dtype=np.float64)
                if dist.size != store.n_features:
                    raise ValueError("peak_prior_distance length must match selected peak count")
                lo, hi = np.percentile(dist[np.isfinite(dist)], [5, 95]) if np.isfinite(dist).any() else (0.0, 1.0)
                scaled = (np.clip(dist, lo, hi) - lo) / max(hi - lo, 1.0e-8)
                confidence = (1.0 - scaled).astype(np.float32)
        eps = self.epsilon_min + (1.0 - confidence) * (self.epsilon_max - self.epsilon_min)
        prior = np.full((store.n_features, self.K), 0.0, dtype=np.float32)
        prior += (eps / self.K)[:, None]
        prior[np.arange(store.n_features), labels] += 1.0 - eps
        prior /= prior.sum(axis=1, keepdims=True)
        return prior

    def _initialize_domains(
        self,
        store: SpatialMapStore,
        A: np.ndarray,
        initial_labels: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if initial_labels is not None:
            labels = np.asarray(initial_labels, dtype=np.int64)
            if labels.size != store.n_spots:
                raise ValueError("initial_domain_labels length must match selected spot count")
            unique = np.unique(labels)
            if unique.size != self.R:
                raise ValueError(
                    "Explicit initial domain labels must contain exactly "
                    f"n_domains={self.R} unique labels; found {unique.size}"
                )
            mapping = {old: new for new, old in enumerate(sorted(unique))}
            aligned = np.array([mapping[x] for x in labels], dtype=np.int64)
            return (
                _smoothed_one_hot(aligned, self.R, smooth=0.05),
                aligned,
                {
                    "mode": "supplied_labels_smoothed_one_hot",
                    "temperature": None,
                    "assigned_probability": float(0.95 + 0.05 / self.R),
                    "other_probability": float(0.05 / self.R),
                },
            )

        _, _, posterior_variance_sums, module_maps = self._spot_module_stats(store, A)
        standardized_maps = _weighted_zscore_columns(
            module_maps,
            _store_spot_weights(store),
        )
        geometry_weights, weighting_metadata = _domain_initialization_module_weights(
            store,
            A,
            module_maps,
            posterior_variance_sums,
            mode=self.domain_initialization_module_weighting,
        )
        initialization_geometry = standardized_maps * np.sqrt(geometry_weights)[None, :]
        kmeans = KMeans(
            n_clusters=self.R,
            n_init=self.domain_initialization_n_init,
            random_state=self.random_state,
        )
        spot_weights = _store_spot_weights(store)
        if np.allclose(spot_weights, 1.0):
            kmeans.fit(initialization_geometry)
        else:
            kmeans.fit(initialization_geometry, sample_weight=spot_weights)
        labels = kmeans.labels_.astype(np.int64, copy=False)
        if self.domain_initialization == "hard-kmeans-module-maps":
            return (
                _smoothed_one_hot(labels, self.R, smooth=0.05),
                labels,
                {
                    "mode": "hard_kmeans_on_initial_module_maps",
                    "temperature": None,
                    "module_weighting": weighting_metadata,
                },
            )

        squared_distances = np.square(
            kmeans.transform(initialization_geometry),
            dtype=np.float64,
        )
        temperature = self.domain_initialization_temperature
        if temperature is None:
            assigned_distances = squared_distances[np.arange(store.n_spots), labels]
            per_dimension = assigned_distances / max(initialization_geometry.shape[1], 1)
            positive = per_dimension[np.isfinite(per_dimension) & (per_dimension > 0.0)]
            robust_variance = float(np.median(positive)) if positive.size else 1.0
            temperature = max(2.0 * robust_variance, 1.0e-6)
        responsibilities = _row_softmax(-squared_distances / float(temperature))
        return (
            responsibilities.astype(np.float32),
            labels,
            {
                "mode": "soft_kmeans_on_initial_module_maps",
                "temperature": float(temperature),
                "temperature_policy": (
                    "explicit"
                    if self.domain_initialization_temperature is not None
                    else "twice_median_within_cluster_variance_per_module"
                ),
                "module_weighting": weighting_metadata,
            },
        )

    def _resolved_domain_update(self) -> str:
        return self.domain_update_config.resolved_update()

    def _resolved_domain_capacity_factor(self) -> float:
        return self.domain_update_config.resolved_capacity_factor()

    def _resolved_final_domain_capacity_factor(self) -> float:
        return self.domain_update_config.resolved_final_capacity_factor()

    def _feature_domain_stats(self, store: SpatialMapStore, B: np.ndarray) -> dict[str, np.ndarray]:
        return self._feature_domain_stats_from_blocks(store.iter_blocks(include_variance=True), store, B)

    def _residual_feature_domain_stats(
        self,
        store: SpatialMapStore,
        B: np.ndarray,
    ) -> dict[str, np.ndarray]:
        blocks = getattr(store, "iter_residual_blocks")()
        return self._feature_domain_stats_from_blocks(blocks, store, B)

    def _feature_domain_stats_from_blocks(
        self,
        blocks: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]],
        store: SpatialMapStore,
        B: np.ndarray,
        spot_weights: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        resolved_spot_weights = (
            _store_spot_weights(store)
            if spot_weights is None
            else np.asarray(spot_weights, dtype=np.float64)
        )
        if resolved_spot_weights.shape != (B.shape[0],):
            raise ValueError("spot_weights must match the rows of B")
        if self._torch_compute is not None:
            return self._torch_compute.feature_domain_stats(
                blocks,
                n_features=store.n_features,
                n_domains=self.R,
                B=B,
                spot_weights=resolved_spot_weights,
            )
        FtB = np.zeros((store.n_features, self.R), dtype=np.float64)
        F2tB = np.zeros_like(FtB)
        UtB = np.zeros_like(FtB)
        for rows, mean, var, positions, _ in blocks:
            mean64 = mean.astype(np.float64, copy=False)
            B_block = B[rows] * resolved_spot_weights[rows, None]
            FtB[positions] += mean64.T @ B_block
            F2tB[positions] += (mean64 * mean64).T @ B_block
            if var is not None:
                UtB[positions] += var.astype(np.float64, copy=False).T @ B_block
        return {"FtB": FtB, "F2tB": F2tB, "UtB": UtB}

    def _dataset_feature_domain_stats(
        self,
        store: SpatialMapStore,
        B: np.ndarray,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray, list[str], list[slice]]:
        """Compute per-dataset sufficient statistics on a common feature axis."""

        names, slices = _dataset_slices(store)
        if len(slices) == 1:
            stats = self._feature_domain_stats(store, B)
            return [stats], np.asarray([_weighted_domain_masses(store, B)]), names, slices
        children = getattr(store, "children", None)
        if children is None or len(children) != len(slices):
            raise ValueError("dataset-shrinkage requires aligned child map stores")
        all_weights = _store_spot_weights(store)
        stats_by_dataset: list[dict[str, np.ndarray]] = []
        masses = []
        for child, rows in zip(children, slices):
            local_B = B[rows]
            local_weights = all_weights[rows]
            stats_by_dataset.append(
                self._feature_domain_stats_from_blocks(
                    child.iter_blocks(include_variance=True),
                    child,
                    local_B,
                    spot_weights=local_weights,
                )
            )
            masses.append(local_weights @ local_B)
        return stats_by_dataset, np.asarray(masses), names, slices

    def _dataset_block_parameters(
        self,
        A_effective: np.ndarray,
        shared_mu: np.ndarray,
        shared_var: np.ndarray,
        stats_by_dataset: list[dict[str, np.ndarray]],
        masses_by_dataset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate child signatures and shrink them toward the pooled blocks."""

        dataset_mu = []
        dataset_var = []
        shrinkage = []
        for stats, masses in zip(stats_by_dataset, masses_by_dataset):
            raw_mu, raw_var, _ = _block_parameters(
                A_effective,
                stats["FtB"],
                stats["F2tB"],
                stats["UtB"],
                masses,
                self.variance_floor,
                self.min_peak_module_mass,
                self.min_domain_mass,
            )
            tau = (
                float(np.sum(masses)) / max(self.R, 1)
                if self.dataset_signature_shrinkage_mass is None
                else float(self.dataset_signature_shrinkage_mass)
            )
            weight = masses / np.maximum(masses + tau, 1.0e-12)
            weight_2d = weight[None, :]
            child_mu = weight_2d * raw_mu + (1.0 - weight_2d) * shared_mu
            child_var = (
                weight_2d * raw_var
                + (1.0 - weight_2d) * shared_var
                + weight_2d
                * (1.0 - weight_2d)
                * np.square(raw_mu - shared_mu)
            )
            dataset_mu.append(child_mu)
            dataset_var.append(np.maximum(child_var, self.variance_floor))
            shrinkage.append(weight)
        return np.asarray(dataset_mu), np.asarray(dataset_var), np.asarray(shrinkage)

    def _dataset_peak_evidence(
        self,
        stats_by_dataset: list[dict[str, np.ndarray]],
        masses_by_dataset: np.ndarray,
        dataset_mu: np.ndarray,
        dataset_var: np.ndarray,
    ) -> np.ndarray:
        evidence = np.zeros((stats_by_dataset[0]["FtB"].shape[0], self.K), dtype=np.float64)
        for index, (stats, masses) in enumerate(zip(stats_by_dataset, masses_by_dataset)):
            evidence += _peak_evidence(
                stats["FtB"],
                self.block_likelihood_config.evidence_second_moment(
                    stats["F2tB"], stats["UtB"]
                ),
                masses,
                dataset_mu[index],
                dataset_var[index],
            )
        return evidence

    def _dataset_spot_evidence(
        self,
        FA: np.ndarray,
        F2A: np.ndarray,
        nA: np.ndarray,
        dataset_mu: np.ndarray,
        dataset_var: np.ndarray,
        slices: list[slice],
    ) -> np.ndarray:
        evidence = np.empty((FA.shape[0], self.R), dtype=np.float64)
        for index, rows in enumerate(slices):
            evidence[rows] = self._spot_evidence(
                FA[rows], F2A[rows], nA, dataset_mu[index], dataset_var[index]
            )
        return evidence

    def _spot_module_stats(
        self,
        store: SpatialMapStore,
        A: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self._spot_module_stats_from_blocks(
            store.iter_blocks(include_variance=self.block_likelihood_config.uses_expected_posterior),
            store,
            A,
            compute_module_maps=True,
        )

    def _residual_spot_module_stats(
        self,
        store: SpatialMapStore,
        A: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        FA, F2A, UA, _ = self._spot_module_stats_from_blocks(
            getattr(store, "iter_residual_blocks")(),
            store,
            A,
            compute_module_maps=False,
        )
        return FA, F2A, UA

    def _spot_module_stats_from_blocks(
        self,
        blocks: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]],
        store: SpatialMapStore,
        A: np.ndarray,
        *,
        compute_module_maps: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._torch_compute is not None:
            return self._torch_compute.spot_module_stats(
                blocks,
                n_spots=store.n_spots,
                n_modules=self.K,
                A=A,
                compute_module_maps=compute_module_maps,
            )
        FA = np.zeros((store.n_spots, self.K), dtype=np.float64)
        F2A = np.zeros_like(FA)
        UA = np.zeros_like(FA)
        nA = np.maximum(A.sum(axis=0), 1.0e-8)
        A_norm = A / nA[None, :]
        module_maps = np.zeros_like(FA)
        for rows, mean, var, positions, _ in blocks:
            mean64 = mean.astype(np.float64, copy=False)
            A_block = A[positions]
            FA[rows] += mean64 @ A_block
            F2A[rows] += (mean64 * mean64) @ A_block
            if var is not None:
                UA[rows] += var.astype(np.float64, copy=False) @ A_block
            if compute_module_maps:
                module_maps[rows] += mean64 @ A_norm[positions]
        return FA, F2A, UA, module_maps.astype(np.float32)

    def _spot_evidence(
        self,
        FA: np.ndarray,
        F2A: np.ndarray,
        nA: np.ndarray,
        mu: np.ndarray,
        var: np.ndarray,
    ) -> np.ndarray:
        if self._torch_compute is not None:
            return self._torch_compute.spot_evidence(FA, F2A, nA, mu, var)
        return _spot_evidence(FA, F2A, nA, mu, var)

    def _row_softmax(self, logits: np.ndarray) -> np.ndarray:
        if self._torch_compute is not None:
            return self._torch_compute.row_softmax(logits)
        return _row_softmax(logits)

    def _top_two_margin(self, logits: np.ndarray) -> np.ndarray:
        if self._torch_compute is not None:
            return self._torch_compute.top_two_margin(logits)
        return _top_two_margin(logits)

    def _graph_message(
        self,
        module_maps: np.ndarray,
        B: np.ndarray,
        edges_u: np.ndarray,
        edges_v: np.ndarray,
        edge_w: np.ndarray,
        module_weights: np.ndarray | None = None,
    ) -> GraphMessageResult:
        return self.graph_config.message(
            module_maps=module_maps,
            B=B,
            edges_u=edges_u,
            edges_v=edges_v,
            edge_w=edge_w,
            n_peak_modules=self.K,
            module_weights=module_weights,
        )


@dataclass
class _DomainProposal:
    operation: str
    initial_B: np.ndarray
    screening_score: float
    details: dict[str, Any]


@dataclass(frozen=True)
class _SplitProposalDescriptor:
    parent_domain: int
    centers: np.ndarray
    distance_scale: float
    screening_score: float
    details: dict[str, Any]


@dataclass(frozen=True)
class _MergeProposalDescriptor:
    left_domain: int
    right_domain: int
    screening_score: float
    details: dict[str, Any]


class AdaptiveVSKILBM:
    """Split/merge model selection around a fitted :class:`VSKILBM`.

    The inner LBM retains soft responsibilities. Candidate splits are seeded
    only from spatially coherent two-way structure in the fitted peak-module
    maps, and merges are restricted to graph-adjacent domains with similar
    Gaussian block signatures. Every accepted topology change must improve a
    common pseudo-ELBO containing the tempered likelihood, graph term,
    variational Dirichlet mixture term, assignment entropies, and an explicit
    per-active-domain complexity penalty.
    """

    def __init__(
        self,
        base_model: VSKILBM,
        *,
        min_domains: int = 4,
        max_domains: int = 20,
        max_rounds: int = 4,
        proposal_max_iter: int = 3,
        final_polish_iter: int = 5,
        split_candidates: int = 1,
        merge_candidates: int = 1,
        split_min_spots: int = 1_000,
        split_sample_spots: int = 20_000,
        split_min_relative_sse_gain: float = 0.05,
        split_max_edge_cut_fraction: float = 0.35,
        split_min_lcc_fraction: float = 0.25,
        split_temperature: float = 0.5,
        max_median_normalized_entropy: float = 0.95,
        min_signature_symmetric_kl: float = 1.0e-3,
        active_domain_penalty: float | None = None,
        min_score_improvement: float = 0.0,
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        self.base_model = base_model
        self.min_domains = int(min_domains)
        self.max_domains = int(max_domains)
        self.max_rounds = int(max_rounds)
        self.proposal_max_iter = int(proposal_max_iter)
        self.final_polish_iter = int(final_polish_iter)
        self.split_candidates = int(split_candidates)
        self.merge_candidates = int(merge_candidates)
        self.split_min_spots = int(split_min_spots)
        self.split_sample_spots = int(split_sample_spots)
        self.split_min_relative_sse_gain = float(split_min_relative_sse_gain)
        self.split_max_edge_cut_fraction = float(split_max_edge_cut_fraction)
        self.split_min_lcc_fraction = float(split_min_lcc_fraction)
        self.split_temperature = float(split_temperature)
        self.max_median_normalized_entropy = float(max_median_normalized_entropy)
        self.min_signature_symmetric_kl = float(min_signature_symmetric_kl)
        self.active_domain_penalty = None if active_domain_penalty is None else float(active_domain_penalty)
        self.min_score_improvement = float(min_score_improvement)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        if self.min_domains < 1 or self.max_domains < self.min_domains:
            raise ValueError("Require 1 <= min_domains <= max_domains")
        if not self.min_domains <= base_model.R <= self.max_domains:
            raise ValueError("The initial n_domains must lie between min_domains and max_domains")
        if self.max_rounds < 0 or self.proposal_max_iter < 1 or self.final_polish_iter < 0:
            raise ValueError("Invalid adaptive-domain iteration counts")
        if self.split_candidates < 0 or self.merge_candidates < 0:
            raise ValueError("Candidate counts must be nonnegative")
        if self.split_min_spots < 2 or self.split_sample_spots < 2:
            raise ValueError("Split spot thresholds must be at least two")
        if not 0.0 <= self.split_min_relative_sse_gain < 1.0:
            raise ValueError("split_min_relative_sse_gain must be in [0, 1)")
        if not 0.0 <= self.split_max_edge_cut_fraction <= 1.0:
            raise ValueError("split_max_edge_cut_fraction must be in [0, 1]")
        if not 0.0 <= self.split_min_lcc_fraction <= 1.0:
            raise ValueError("split_min_lcc_fraction must be in [0, 1]")
        if self.split_temperature <= 0.0:
            raise ValueError("split_temperature must be positive")
        if not 0.0 < self.max_median_normalized_entropy <= 1.0:
            raise ValueError("max_median_normalized_entropy must be in (0, 1]")
        if self.min_signature_symmetric_kl < 0.0:
            raise ValueError("min_signature_symmetric_kl must be nonnegative")
        if base_model.domain_mixture_prior != "dirichlet":
            raise ValueError("Adaptive domain selection requires domain_mixture_prior='dirichlet'")
        if base_model._resolved_domain_update() != "softmax":
            raise ValueError("Adaptive domain selection requires unconstrained softmax domain updates")
        if base_model.domain_size_penalty_weight != 0.0:
            raise ValueError("Adaptive domain selection cannot be combined with domain-size penalties")
        if base_model.final_domain_labeling != "argmax":
            raise ValueError("Adaptive domain selection requires final_domain_labeling='argmax'")

    def fit(
        self,
        store: SpatialMapStore,
        *,
        peak_prior_labels: np.ndarray | None = None,
        peak_prior_distance: np.ndarray | None = None,
        initial_domain_labels: np.ndarray | None = None,
        initial_peak_responsibilities: np.ndarray | None = None,
        initial_domain_responsibilities: np.ndarray | None = None,
        prototype_responsibilities: np.ndarray | None = None,
        precomputed_graph: sparse.spmatrix | None = None,
    ) -> VSKILBMResult:
        if precomputed_graph is None:
            graph, _ = self.base_model._spatial_graph(store)
        else:
            graph = sparse.csr_matrix(precomputed_graph, dtype=np.float64)
            expected_shape = (store.n_spots, store.n_spots)
            if graph.shape != expected_shape:
                raise ValueError(
                    f"precomputed_graph must have shape {expected_shape}, got {graph.shape}"
                )
        edges_u, edges_v, edge_w = _undirected_edges(graph)
        penalty = self._resolved_active_domain_penalty(store)
        common_peak_prior_weight = float(self.base_model.lambda_z_min)
        proposal_peak_prior_amplitude = self._continued_peak_prior_amplitude()
        rng = np.random.default_rng(self.random_state)

        current = self.base_model.fit(
            store,
            peak_prior_labels=peak_prior_labels,
            peak_prior_distance=peak_prior_distance,
            initial_domain_labels=initial_domain_labels,
            initial_peak_responsibilities=initial_peak_responsibilities,
            initial_domain_responsibilities=initial_domain_responsibilities,
            prototype_responsibilities=prototype_responsibilities,
            precomputed_graph=graph,
        )
        original_initial_labels = current.initial_domain_labels
        original_initialization_metadata = copy.deepcopy(current.initialization_metadata)
        current_score = self._selection_score(current, penalty, common_peak_prior_weight)
        model_history: list[dict[str, Any]] = [
            self._history_record(
                round_index=0,
                operation="initial_fit",
                accepted=True,
                result=current,
                score=current_score,
                score_delta=0.0,
                penalty=penalty,
                details={},
            )
        ]
        any_accepted = False

        for round_index in range(1, self.max_rounds + 1):
            proposals = self._candidate_proposals(current, graph, edges_u, edges_v, rng)
            if not proposals:
                model_history.append(
                    self._history_record(
                        round_index=round_index,
                        operation="no_valid_proposal",
                        accepted=False,
                        result=current,
                        score=current_score,
                        score_delta=0.0,
                        penalty=penalty,
                        details={},
                    )
                )
                break

            # Candidate results carry full N x R responsibilities and N x K
            # module maps. Retaining every valid candidate can exhaust memory
            # on full slides, so preserve only the best result while recording
            # scalar diagnostics for all evaluated proposals.
            best_evaluated: tuple[float, VSKILBMResult, _DomainProposal] | None = None
            for proposal in proposals:
                candidate_model = self._proposal_model(
                    proposal.initial_B.shape[1], proposal_peak_prior_amplitude
                )
                candidate = candidate_model.fit(
                    store,
                    peak_prior_labels=peak_prior_labels,
                    peak_prior_distance=peak_prior_distance,
                    initial_peak_responsibilities=current.A,
                    initial_domain_responsibilities=proposal.initial_B,
                    prototype_responsibilities=prototype_responsibilities,
                    precomputed_graph=graph,
                )
                candidate_score = self._selection_score(candidate, penalty, common_peak_prior_weight)
                delta = candidate_score - current_score
                valid_change = self._has_expected_active_count_change(current, candidate, proposal.operation)
                identifiable, identification_reasons = self._is_identifiable(candidate)
                details = dict(proposal.details)
                details["screening_score"] = float(proposal.screening_score)
                details["expected_active_count_change"] = bool(valid_change)
                details["identifiable"] = bool(identifiable)
                details["identification_reasons"] = identification_reasons
                model_history.append(
                    self._history_record(
                        round_index=round_index,
                        operation=f"evaluate_{proposal.operation}",
                        accepted=False,
                        result=candidate,
                        score=candidate_score,
                        score_delta=delta,
                        penalty=penalty,
                        details=details,
                    )
                )
                if valid_change and identifiable:
                    if best_evaluated is None or candidate_score > best_evaluated[0]:
                        best_evaluated = (candidate_score, candidate, proposal)
                    else:
                        del candidate
                else:
                    del candidate

            if best_evaluated is None:
                break
            best_score, best_result, best_proposal = best_evaluated
            improvement = best_score - current_score
            if improvement <= self.min_score_improvement:
                del best_result
                break
            current = best_result
            current_score = best_score
            any_accepted = True
            accepted_details = dict(best_proposal.details)
            accepted_details["screening_score"] = float(best_proposal.screening_score)
            model_history.append(
                self._history_record(
                    round_index=round_index,
                    operation=f"accept_{best_proposal.operation}",
                    accepted=True,
                    result=current,
                    score=current_score,
                    score_delta=improvement,
                    penalty=penalty,
                    details=accepted_details,
                )
            )
            if self.verbose:
                print(
                    f"adaptive round {round_index}: accepted {best_proposal.operation}; "
                    f"R={current.B.shape[1]} delta={improvement:.3f}"
                )

        if self.final_polish_iter > 0 and any_accepted:
            polish_model = self._proposal_model(current.B.shape[1], proposal_peak_prior_amplitude)
            polish_model.max_iter = self.final_polish_iter
            polished = polish_model.fit(
                store,
                peak_prior_labels=peak_prior_labels,
                peak_prior_distance=peak_prior_distance,
                initial_peak_responsibilities=current.A,
                initial_domain_responsibilities=current.B,
                prototype_responsibilities=prototype_responsibilities,
                precomputed_graph=graph,
            )
            polished_score = self._selection_score(polished, penalty, common_peak_prior_weight)
            improvement = polished_score - current_score
            polished_identifiable, polish_reasons = self._is_identifiable(polished)
            same_active_count = int(polished.active_domain_mask.sum()) == int(current.active_domain_mask.sum())
            accept_polish = improvement >= -1.0e-6 and polished_identifiable and same_active_count
            if accept_polish:
                current = polished
                current_score = polished_score
            model_history.append(
                self._history_record(
                    round_index=self.max_rounds + 1,
                    operation="final_polish",
                    accepted=bool(accept_polish),
                    result=polished,
                    score=polished_score,
                    score_delta=improvement,
                    penalty=penalty,
                    details={
                        "iterations": int(self.final_polish_iter),
                        "identifiable": bool(polished_identifiable),
                        "identification_reasons": polish_reasons,
                        "same_active_count": bool(same_active_count),
                    },
                )
            )

        current.initial_domain_labels = original_initial_labels
        current.initialization_metadata = {
            **(original_initialization_metadata or {}),
            "adaptive_continuation": {
                "initial_components": int(self.base_model.R),
                "final_components": int(current.B.shape[1]),
                "accepted_topology_changes": int(
                    sum(
                        bool(row.get("accepted"))
                        and str(row.get("operation", "")).startswith("accept_")
                        for row in model_history
                    )
                ),
                "final_polish_used_current_responsibilities": bool(any_accepted),
            },
        }
        current.domain_model_history = model_history
        current.final_objective_components["model_selection_score"] = float(current_score)
        current.final_objective_components["active_domain_penalty_per_component"] = float(penalty)
        return current

    def _resolved_active_domain_penalty(self, store: SpatialMapStore) -> float:
        if self.active_domain_penalty is not None:
            return self.active_domain_penalty
        # One extra domain adds one mean and variance per peak module plus one
        # mixture-weight degree of freedom. This is a pseudo-BIC penalty because
        # the graph-regularized objective is not an iid likelihood.
        added_parameters = 2 * self.base_model.K + 1
        return float(0.5 * added_parameters * np.log(max(store.n_spots, 2)))

    def _selection_score(
        self,
        result: VSKILBMResult,
        penalty: float,
        common_peak_prior_weight: float,
    ) -> float:
        components = result.final_objective_components
        a_scale = float(components.get("a_objective_regularizer_scale", 1.0))
        return float(
            components["domain_powered_data_objective"]
            + a_scale
            * (
                common_peak_prior_weight * components["peak_prior_cross_entropy"]
                + components["factor_feature_prior_objective"]
                + self.base_model.a_temperature * components["entropy_A"]
            )
            + components["factor_location_prior_objective"]
            + components["graph_objective"]
            + components["weighted_domain_mixture_objective"]
            + self.base_model.b_temperature * components["entropy_B"]
            - penalty * int(result.active_domain_mask.sum())
        )

    def _continued_peak_prior_amplitude(self) -> float:
        return self.base_model.peak_prior_schedule.continued_amplitude_after(self.base_model.max_iter)

    def _proposal_model(self, n_domains: int, peak_prior_amplitude: float) -> VSKILBM:
        model = copy.copy(self.base_model)
        model.R = int(n_domains)
        model.max_iter = self.proposal_max_iter
        model.peak_prior_schedule = PeakPriorScheduleConfig(
            initial_weight=float(peak_prior_amplitude),
            minimum_weight=float(model.lambda_z_min),
            decay_tau=float(model.peak_prior_decay_tau),
        )
        model.lambda_z = model.peak_prior_schedule.initial_weight
        model.random_state = self.random_state
        model.graph_config = GraphRegularizationConfig(
            model=model.graph_model,
            beta=model.beta,
            roughness_weight=model.roughness_weight,
            boundary_scale=model.graph_boundary_scale,
            edge_batch_size=model.edge_batch_size,
            random_state=self.random_state,
        )
        return model

    def _is_identifiable(self, result: VSKILBMResult) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        hard_sizes = np.bincount(result.B.argmax(axis=1), minlength=result.B.shape[1])
        active = result.active_domain_mask
        required_hard_size = max(int(np.ceil(self.base_model.active_domain_min_spots)), 1)
        if np.any(hard_sizes[active] < required_hard_size):
            reasons.append("an active posterior component lacks enough hard assignments")
        median_entropy = float(np.median(_normalized_row_entropy(result.B)))
        if median_entropy > self.max_median_normalized_entropy:
            reasons.append("median normalized domain entropy exceeds the configured threshold")
        active_indices = np.flatnonzero(active)
        signature_mu, signature_var, _ = _identifiability_signature_arrays(result)
        min_kl = np.inf
        for left_pos, left in enumerate(active_indices):
            for right in active_indices[left_pos + 1 :]:
                min_kl = min(
                    min_kl,
                    _symmetric_diagonal_gaussian_kl(
                        signature_mu[:, left],
                        signature_var[:, left],
                        signature_mu[:, right],
                        signature_var[:, right],
                    ),
                )
        if np.isfinite(min_kl) and min_kl < self.min_signature_symmetric_kl:
            reasons.append("at least two active Gaussian domain signatures are effectively duplicated")
        return not reasons, reasons

    def _candidate_proposals(
        self,
        result: VSKILBMResult,
        graph: sparse.csr_matrix,
        edges_u: np.ndarray,
        edges_v: np.ndarray,
        rng: np.random.Generator,
    ) -> list[_DomainProposal]:
        proposals: list[_DomainProposal] = []
        graph_free = (
            self.base_model.graph_build_config.resolved_construction(
                graph_model=self.base_model.graph_model
            )
            == "none"
        )
        n_active = int(result.active_domain_mask.sum())
        if n_active < result.B.shape[1] and n_active >= self.min_domains:
            keep = np.flatnonzero(result.active_domain_mask)
            pruned = result.B[:, keep]
            pruned /= np.maximum(pruned.sum(axis=1, keepdims=True), 1.0e-12)
            proposals.append(
                _DomainProposal(
                    operation="prune",
                    initial_B=pruned,
                    screening_score=float(result.B[:, ~result.active_domain_mask].sum()),
                    details={"removed_domains": np.flatnonzero(~result.active_domain_mask).astype(int).tolist()},
                )
            )
        if result.B.shape[1] < self.max_domains and self.split_candidates > 0:
            proposals.extend(
                self._split_proposals(
                    result,
                    graph,
                    edges_u,
                    edges_v,
                    rng,
                    graph_free=graph_free,
                    max_candidates=self.split_candidates,
                )
            )
        if result.B.shape[1] > self.min_domains and self.merge_candidates > 0:
            proposals.extend(
                self._merge_proposals(
                    result,
                    edges_u,
                    edges_v,
                    graph_free=graph_free,
                    max_candidates=self.merge_candidates,
                )
            )
        return proposals

    def _split_proposals(
        self,
        result: VSKILBMResult,
        graph: sparse.csr_matrix,
        edges_u: np.ndarray,
        edges_v: np.ndarray,
        rng: np.random.Generator,
        *,
        graph_free: bool,
        max_candidates: int,
    ) -> list[_DomainProposal]:
        labels = result.B.argmax(axis=1)
        maps = result.module_maps.astype(np.float64, copy=False)
        scale = np.maximum(maps.std(axis=0), 1.0e-6)
        standardized = (maps - maps.mean(axis=0)) / scale
        descriptors: list[_SplitProposalDescriptor] = []
        for domain in np.flatnonzero(result.active_domain_mask):
            rows = np.flatnonzero(labels == domain)
            if rows.size < 2 * self.split_min_spots:
                continue
            sample = rows
            if sample.size > self.split_sample_spots:
                sample = rng.choice(sample, size=self.split_sample_spots, replace=False)
            km = KMeans(n_clusters=2, n_init=10, random_state=self.random_state + int(domain))
            km.fit(standardized[sample])
            child_labels = km.predict(standardized[rows])
            child_sizes = np.bincount(child_labels, minlength=2)
            if child_sizes.min() < self.split_min_spots:
                continue
            centered = standardized[sample] - standardized[sample].mean(axis=0, keepdims=True)
            total_sse = float(np.sum(centered * centered))
            relative_gain = 1.0 - float(km.inertia_) / max(total_sse, 1.0e-12)
            if relative_gain < self.split_min_relative_sse_gain:
                continue

            if graph_free:
                cut_fraction = 0.0
                lcc_fractions = [1.0, 1.0]
            else:
                local_child = np.full(result.B.shape[0], -1, dtype=np.int8)
                local_child[rows] = child_labels.astype(np.int8)
                inside = (labels[edges_u] == domain) & (labels[edges_v] == domain)
                n_inside_edges = int(inside.sum())
                cut_fraction = (
                    float(
                        np.mean(
                            local_child[edges_u[inside]]
                            != local_child[edges_v[inside]]
                        )
                    )
                    if n_inside_edges
                    else 1.0
                )
                if cut_fraction > self.split_max_edge_cut_fraction:
                    continue
                lcc_fractions = []
                for child in range(2):
                    child_rows = rows[child_labels == child]
                    subgraph = graph[child_rows][:, child_rows]
                    n_components, component_labels = connected_components(
                        subgraph, directed=False
                    )
                    if n_components == 0:
                        lcc_fractions.append(0.0)
                    else:
                        lcc_fractions.append(
                            float(
                                np.bincount(component_labels).max()
                                / max(child_rows.size, 1)
                            )
                        )
            min_lcc_fraction = float(min(lcc_fractions))
            if min_lcc_fraction < self.split_min_lcc_fraction:
                continue

            row_distances = np.stack(
                [
                    np.sum(
                        (standardized[rows] - center[None, :]) ** 2,
                        axis=1,
                    )
                    for center in km.cluster_centers_
                ],
                axis=1,
            )
            distance_scale = max(
                float(np.median(np.min(row_distances, axis=1))), 1.0e-6
            )
            screening_score = float(
                relative_gain * (1.0 - cut_fraction) * min_lcc_fraction * np.log1p(rows.size)
            )
            descriptors.append(
                _SplitProposalDescriptor(
                    parent_domain=int(domain),
                    centers=km.cluster_centers_.astype(np.float64, copy=True),
                    distance_scale=distance_scale,
                    screening_score=screening_score,
                    details={
                        "parent_domain": int(domain),
                        "parent_hard_size": int(rows.size),
                        "initial_child_sizes": child_sizes.astype(int).tolist(),
                        "relative_sse_gain": float(relative_gain),
                        "within_parent_edge_cut_fraction": float(cut_fraction),
                        "child_largest_component_fractions": lcc_fractions,
                        "spatial_connectivity_screen_applied": not graph_free,
                    },
                )
            )
        selected = sorted(
            descriptors,
            key=lambda descriptor: descriptor.screening_score,
            reverse=True,
        )[:max_candidates]
        proposals: list[_DomainProposal] = []
        base_B = result.B.astype(np.float64, copy=False)
        for descriptor in selected:
            distances = np.stack(
                [
                    np.sum((standardized - center[None, :]) ** 2, axis=1)
                    for center in descriptor.centers
                ],
                axis=1,
            )
            child_probs = _row_softmax(
                -distances
                / (self.split_temperature * descriptor.distance_scale)
            )
            parent_mass = base_B[:, descriptor.parent_domain]
            split_B = np.empty(
                (base_B.shape[0], base_B.shape[1] + 1), dtype=np.float64
            )
            split_B[:, :-1] = base_B
            split_B[:, descriptor.parent_domain] = parent_mass * child_probs[:, 0]
            split_B[:, -1] = parent_mass * child_probs[:, 1]
            proposals.append(
                _DomainProposal(
                    operation="split",
                    initial_B=split_B,
                    screening_score=descriptor.screening_score,
                    details=descriptor.details,
                )
            )
        return proposals

    def _merge_proposals(
        self,
        result: VSKILBMResult,
        edges_u: np.ndarray,
        edges_v: np.ndarray,
        *,
        graph_free: bool,
        max_candidates: int,
    ) -> list[_DomainProposal]:
        labels = result.B.argmax(axis=1)
        n_domains = result.B.shape[1]
        boundary = np.zeros((n_domains, n_domains), dtype=np.int64)
        left_labels = labels[edges_u]
        right_labels = labels[edges_v]
        crossing = left_labels != right_labels
        np.add.at(boundary, (left_labels[crossing], right_labels[crossing]), 1)
        boundary += boundary.T
        descriptors: list[_MergeProposalDescriptor] = []
        signature_mu, signature_var, signature_model = (
            _identifiability_signature_arrays(result)
        )
        active = np.flatnonzero(result.active_domain_mask)
        for left_pos, left in enumerate(active):
            for right in active[left_pos + 1 :]:
                if not graph_free and boundary[left, right] <= 0:
                    continue
                divergence = _symmetric_diagonal_gaussian_kl(
                    signature_mu[:, left],
                    signature_var[:, left],
                    signature_mu[:, right],
                    signature_var[:, right],
                )
                screening_score = (
                    float(1.0 / max(divergence, 1.0e-12))
                    if graph_free
                    else float(
                        np.log1p(boundary[left, right])
                        / max(divergence, 1.0e-12)
                    )
                )
                descriptors.append(
                    _MergeProposalDescriptor(
                        left_domain=int(left),
                        right_domain=int(right),
                        screening_score=screening_score,
                        details={
                            "domain_a": int(left),
                            "domain_b": int(right),
                            "symmetric_kl": float(divergence),
                            "signature_model": signature_model,
                            "boundary_edge_count": int(boundary[left, right]),
                            "ranking_mode": (
                                "inverse_signature_symmetric_kl"
                                if graph_free
                                else "boundary_edges_over_signature_symmetric_kl"
                            ),
                        },
                    )
                )
        selected = sorted(
            descriptors,
            key=lambda descriptor: descriptor.screening_score,
            reverse=True,
        )[:max_candidates]
        proposals: list[_DomainProposal] = []
        for descriptor in selected:
            keep = np.delete(np.arange(n_domains), descriptor.right_domain)
            merged_B = result.B[:, keep].astype(np.float64, copy=True)
            merged_left = int(
                np.flatnonzero(keep == descriptor.left_domain)[0]
            )
            merged_B[:, merged_left] += result.B[:, descriptor.right_domain]
            proposals.append(
                _DomainProposal(
                    operation="merge",
                    initial_B=merged_B,
                    screening_score=descriptor.screening_score,
                    details=descriptor.details,
                )
            )
        return proposals

    @staticmethod
    def _has_expected_active_count_change(
        current: VSKILBMResult,
        candidate: VSKILBMResult,
        operation: str,
    ) -> bool:
        before = int(current.active_domain_mask.sum())
        after = int(candidate.active_domain_mask.sum())
        if operation == "split":
            return after == before + 1
        if operation == "merge":
            return after == before - 1
        if operation == "prune":
            return after < before
        return after != before

    @staticmethod
    def _history_record(
        *,
        round_index: int,
        operation: str,
        accepted: bool,
        result: VSKILBMResult,
        score: float,
        score_delta: float,
        penalty: float,
        details: dict[str, Any],
    ) -> dict[str, Any]:
        hard_sizes = np.bincount(result.B.argmax(axis=1), minlength=result.B.shape[1])
        return {
            "round": int(round_index),
            "operation": operation,
            "accepted": bool(accepted),
            "n_components": int(result.B.shape[1]),
            "n_active_domains": int(result.active_domain_mask.sum()),
            "n_hard_domains": int(np.sum(hard_sizes > 0)),
            "min_hard_domain_size": int(hard_sizes[hard_sizes > 0].min()) if np.any(hard_sizes > 0) else 0,
            "median_domain_normalized_entropy": float(np.median(_normalized_row_entropy(result.B))),
            "model_selection_score": float(score),
            "score_delta": float(score_delta),
            "active_domain_penalty_per_component": float(penalty),
            "min_posterior_mixture_weight": float(result.domain_mixture_weights.min()),
            "max_posterior_mixture_weight": float(result.domain_mixture_weights.max()),
            **details,
        }


def _validate_responsibilities(
    values: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    responsibilities = np.asarray(values, dtype=np.float64)
    if responsibilities.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {responsibilities.shape}")
    if not np.isfinite(responsibilities).all() or np.any(responsibilities < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative values")
    row_sums = responsibilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"Every row of {name} must have positive mass")
    return responsibilities / row_sums


def _validate_prototype_responsibilities(
    values: np.ndarray,
    *,
    expected_rows: int,
) -> np.ndarray:
    responsibilities = np.asarray(values, dtype=np.float64)
    if (
        responsibilities.ndim != 2
        or responsibilities.shape[0] != expected_rows
        or responsibilities.shape[1] < 2
    ):
        raise ValueError(
            "prototype_responsibilities must have shape "
            f"({expected_rows}, n_prototypes>=2), got {responsibilities.shape}"
        )
    if not np.isfinite(responsibilities).all() or np.any(responsibilities < 0.0):
        raise ValueError("prototype_responsibilities must contain finite nonnegative values")
    row_sums = responsibilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Every prototype responsibility row must have positive mass")
    return responsibilities / row_sums


def _validate_optional_embedding(
    embedding: np.ndarray | None,
    *,
    expected_rows: int,
    name: str,
) -> np.ndarray | None:
    if embedding is None:
        return None
    values = np.asarray(embedding, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != expected_rows or values.shape[1] == 0:
        raise ValueError(
            f"{name} must have shape ({expected_rows}, d) with d > 0; got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    return values


def _embedding_prior_logits(
    embedding: np.ndarray | None,
    responsibilities: np.ndarray,
    *,
    weight: float,
    temperature: float,
) -> np.ndarray:
    """Return continuous centroid-distance prior logits without a 3-D tensor."""
    if embedding is None or weight == 0.0:
        return np.zeros_like(responsibilities, dtype=np.float64)
    values = np.asarray(embedding, dtype=np.float64)
    probs = np.asarray(responsibilities, dtype=np.float64)
    masses = np.maximum(probs.sum(axis=0), 1.0e-12)
    centroids = probs.T @ values / masses[:, None]
    distances = (
        np.square(values).sum(axis=1, keepdims=True)
        + np.square(centroids).sum(axis=1)[None, :]
        - 2.0 * values @ centroids.T
    )
    np.maximum(distances, 0.0, out=distances)
    return -float(weight) * distances / max(float(temperature), 1.0e-12)


def _domain_mixture_terms(
    B: np.ndarray,
    *,
    prior: str,
    alpha0: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return posterior weights, E[log pi], and the optimized pi ELBO term.

    For the Dirichlet model, ``alpha0`` is the total concentration and each of
    the R components receives ``alpha0 / R`` prior mass. The scalar objective
    is the exact conjugate contribution after optimizing q(pi), including the
    assignment-to-mixture, Dirichlet-prior, and q(pi)-entropy terms.
    """

    masses = np.asarray(B, dtype=np.float64).sum(axis=0)
    n_domains = int(masses.size)
    if n_domains == 0:
        raise ValueError("At least one domain is required")
    if prior == "none":
        weights = np.maximum(masses, 1.0e-12)
        weights /= weights.sum()
        return weights, np.log(weights), 0.0
    if prior != "dirichlet":
        raise ValueError(f"Unsupported domain mixture prior: {prior}")

    prior_per_domain = float(alpha0) / n_domains
    posterior_alpha = masses + prior_per_domain
    posterior_total = float(posterior_alpha.sum())
    weights = posterior_alpha / posterior_total
    expected_log_weights = digamma(posterior_alpha) - digamma(posterior_total)
    objective = float(
        gammaln(alpha0)
        - n_domains * gammaln(prior_per_domain)
        - gammaln(posterior_total)
        + np.sum(gammaln(posterior_alpha))
    )
    return weights, expected_log_weights, objective


def peak_module_table(result: VSKILBMResult) -> pd.DataFrame:
    table = result.metadata.copy()
    probs = result.A
    labels = probs.argmax(axis=1)
    table.insert(0, "vski_lbm_peak_module", labels.astype(int))
    table["vski_lbm_peak_module_probability"] = probs.max(axis=1)
    table["vski_lbm_peak_module_entropy"] = _row_entropy(probs)
    table["vski_lbm_peak_module_normalized_entropy"] = _normalized_row_entropy(probs)
    table["vski_lbm_peak_module_effective_candidates"] = np.exp(_row_entropy(probs))
    table["vski_lbm_peak_module_logit_margin"] = result.peak_logit_margin
    if result.prior_labels is not None:
        table["vski_prior_cluster"] = result.prior_labels.astype(int)
        table["changed_from_vski_prior"] = table["vski_lbm_peak_module"].to_numpy() != result.prior_labels
    for k in range(probs.shape[1]):
        table[f"module_{k}_probability"] = probs[:, k]
    return table


def spot_domain_table(result: VSKILBMResult) -> pd.DataFrame:
    table = result.spots.copy()
    probs = result.B
    soft_labels = probs.argmax(axis=1).astype(int)
    if result.final_domain_labels is None:
        labels = soft_labels
    else:
        labels = result.final_domain_labels.astype(int)
        table["vski_lbm_soft_argmax_domain"] = soft_labels
        table["vski_lbm_soft_argmax_probability"] = probs.max(axis=1)
    table["vski_lbm_domain"] = labels
    table["vski_lbm_domain_probability"] = probs[np.arange(probs.shape[0]), labels]
    table["vski_lbm_domain_entropy"] = _row_entropy(probs)
    table["vski_lbm_domain_normalized_entropy"] = _normalized_row_entropy(probs)
    table["vski_lbm_domain_effective_candidates"] = np.exp(_row_entropy(probs))
    table["vski_lbm_domain_logit_margin"] = result.domain_logit_margin
    table["vski_lbm_domain_labeling"] = result.final_domain_labeling
    if result.initial_domain_labels is not None:
        table["initial_domain"] = result.initial_domain_labels.astype(int)
        table["changed_from_initial_domain"] = table["vski_lbm_domain"].to_numpy() != result.initial_domain_labels
    return table


def block_parameter_table(result: VSKILBMResult) -> pd.DataFrame:
    rows = []
    peak_sizes = result.A.sum(axis=0)
    domain_sizes = result.B.sum(axis=0)
    for k in range(result.mu.shape[0]):
        for r in range(result.mu.shape[1]):
            row = {
                    "peak_module": int(k),
                    "domain": int(r),
                    "block_mean": float(result.mu[k, r]),
                    "block_variance": float(result.var[k, r]),
                    "block_mean_posterior_variance": float(result.ubar[k, r]),
                    "soft_peak_module_size": float(peak_sizes[k]),
                    "soft_domain_size": float(domain_sizes[r]),
                    "soft_domain_weight": float(domain_sizes[r] / max(result.B.shape[0], 1)),
                    "posterior_mixture_weight": float(result.domain_mixture_weights[r]),
                    "expected_log_mixture_weight": float(result.domain_expected_log_weights[r]),
                    "active_domain": bool(result.active_domain_mask[r]),
                }
            if result.residual_mu is not None:
                row.update(
                    {
                        "residual_block_mean": float(result.residual_mu[k, r]),
                        "residual_block_variance": float(result.residual_var[k, r]),
                        "residual_block_mean_posterior_variance": float(
                            result.residual_ubar[k, r]
                        ),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def dataset_block_parameter_table(result: VSKILBMResult) -> pd.DataFrame:
    """Tabulate partially pooled child-dataset block signatures."""

    if result.dataset_mu is None or result.dataset_var is None:
        return pd.DataFrame()
    names = result.dataset_names or [f"dataset_{index}" for index in range(result.dataset_mu.shape[0])]
    rows = []
    for dataset_index, name in enumerate(names):
        for module in range(result.dataset_mu.shape[1]):
            for domain in range(result.dataset_mu.shape[2]):
                rows.append(
                    {
                        "dataset": name,
                        "peak_module": module,
                        "domain": domain,
                        "block_mean": float(result.dataset_mu[dataset_index, module, domain]),
                        "block_variance": float(result.dataset_var[dataset_index, module, domain]),
                        "dataset_weight_in_signature": float(
                            result.dataset_signature_shrinkage[dataset_index, domain]
                        ),
                        "shared_block_mean": float(result.mu[module, domain]),
                        "shared_block_variance": float(result.var[module, domain]),
                    }
                )
    return pd.DataFrame(rows)


def feature_likelihood_weight_table(result: VSKILBMResult) -> pd.DataFrame:
    """Return feature-level fractional-likelihood weights and their provenance."""

    if result.feature_likelihood_weights is None:
        return pd.DataFrame()
    table = result.metadata.copy()
    table.insert(0, "feature_likelihood_weight", result.feature_likelihood_weights)
    metadata = (result.initialization_metadata or {}).get(
        "feature_likelihood_weighting", {}
    )
    reliability = metadata.get("gp_signal_reliability")
    if reliability is not None:
        table.insert(1, "gp_signal_reliability", reliability)
    return table


def domain_initialization_module_weight_table(result: VSKILBMResult) -> pd.DataFrame:
    """Tabulate auditable module weights used for domain initialization."""

    domain = (result.initialization_metadata or {}).get("domain", {})
    metadata = domain.get("module_weighting")
    if not metadata:
        return pd.DataFrame()
    fields = {
        "soft_peak_mass": metadata["mass"],
        "gp_signal_reliability": metadata["gp_signal_reliability"],
        "posterior_reliability": metadata["posterior_reliability"],
        "combined_reliability": metadata["combined_reliability"],
        "signal_variance": metadata["signal_variance"],
        "mean_posterior_variance": metadata["mean_posterior_variance"],
        "raw_weight": metadata["raw_weight"],
        "geometry_weight_mean_one": metadata["geometry_weight_mean_one"],
    }
    table = pd.DataFrame(fields)
    table.insert(0, "peak_module", np.arange(table.shape[0], dtype=np.int64))
    table.insert(1, "weighting_mode", metadata["mode"])
    return table


def domain_summary_table(result: VSKILBMResult) -> pd.DataFrame:
    hard_labels = result.B.argmax(axis=1)
    hard_sizes = np.bincount(hard_labels, minlength=result.B.shape[1])
    soft_sizes = result.B.sum(axis=0)
    return pd.DataFrame(
        {
            "domain": np.arange(result.B.shape[1], dtype=int),
            "active_domain": result.active_domain_mask.astype(bool),
            "hard_domain_size": hard_sizes.astype(int),
            "soft_domain_size": soft_sizes,
            "soft_domain_weight": soft_sizes / max(result.B.shape[0], 1),
            "posterior_mixture_weight": result.domain_mixture_weights,
            "expected_log_mixture_weight": result.domain_expected_log_weights,
        }
    )


def _identifiability_signature_arrays(
    result: VSKILBMResult,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the Gaussian signature actually used by the fitted likelihood."""

    dataset_mu = getattr(result, "dataset_mu", None)
    dataset_var = getattr(result, "dataset_var", None)
    if dataset_mu is not None and dataset_var is not None:
        n_datasets, n_modules, n_domains = dataset_mu.shape
        means = dataset_mu.transpose(1, 0, 2).reshape(
            n_modules * n_datasets, n_domains
        )
        variances = dataset_var.transpose(1, 0, 2).reshape(
            n_modules * n_datasets, n_domains
        )
        return means, variances, "concatenated_dataset_specific"
    return result.mu, result.var, "shared"


def domain_signature_distance_table(result: VSKILBMResult) -> pd.DataFrame:
    """Pairwise symmetric KL distances between fitted Gaussian signatures."""

    rows: list[dict[str, float | int]] = []
    signature_mu, signature_var, signature_model = (
        _identifiability_signature_arrays(result)
    )
    for left in range(result.mu.shape[1]):
        for right in range(left + 1, result.mu.shape[1]):
            rows.append(
                {
                    "domain_a": int(left),
                    "domain_b": int(right),
                    "domain_a_active": bool(result.active_domain_mask[left]),
                    "domain_b_active": bool(result.active_domain_mask[right]),
                    "signature_model": signature_model,
                    "symmetric_kl": float(
                        _symmetric_diagonal_gaussian_kl(
                            signature_mu[:, left],
                            signature_var[:, left],
                            signature_mu[:, right],
                            signature_var[:, right],
                        )
                    ),
                    "mean_euclidean_distance": float(
                        np.linalg.norm(signature_mu[:, left] - signature_mu[:, right])
                    ),
                    "pooled_symmetric_kl": float(
                        _symmetric_diagonal_gaussian_kl(
                            result.mu[:, left],
                            result.var[:, left],
                            result.mu[:, right],
                            result.var[:, right],
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def assignment_metrics(result: VSKILBMResult) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {}
    final_peak = result.A.argmax(axis=1)
    final_domain = result.final_domain_labels if result.final_domain_labels is not None else result.B.argmax(axis=1)
    peak_entropy = _row_entropy(result.A)
    domain_entropy = _row_entropy(result.B)
    peak_masses = result.A.sum(axis=0)
    domain_weights = result.B.mean(axis=0)
    signature_distances = domain_signature_distance_table(result)
    active_signature_distances = signature_distances.loc[
        signature_distances["domain_a_active"] & signature_distances["domain_b_active"]
    ] if not signature_distances.empty else signature_distances
    metrics.update(
        {
            "median_peak_max_probability": float(np.median(result.A.max(axis=1))),
            "median_peak_normalized_entropy": float(np.median(_normalized_row_entropy(result.A))),
            "median_peak_effective_candidates": float(np.median(np.exp(peak_entropy))),
            "median_peak_logit_margin": float(np.median(result.peak_logit_margin)),
            "n_active_peak_modules_hard": int(np.unique(final_peak).size),
            "min_soft_peak_module_mass": float(peak_masses.min()),
            "median_domain_max_probability": float(np.median(result.B.max(axis=1))),
            "median_domain_normalized_entropy": float(np.median(_normalized_row_entropy(result.B))),
            "median_domain_effective_candidates": float(np.median(np.exp(domain_entropy))),
            "median_domain_logit_margin": float(np.median(result.domain_logit_margin)),
            "n_active_domains_hard": int(np.unique(final_domain).size),
            "n_active_domains_posterior": int(result.active_domain_mask.sum()),
            "min_soft_domain_weight": float(domain_weights.min()),
            "max_soft_domain_weight": float(domain_weights.max()),
            "min_active_posterior_mixture_weight": float(
                result.domain_mixture_weights[result.active_domain_mask].min()
                if result.active_domain_mask.any()
                else 0.0
            ),
            "max_posterior_mixture_weight": float(result.domain_mixture_weights.max()),
            "min_domain_signature_symmetric_kl": (
                float(active_signature_distances["symmetric_kl"].min())
                if not active_signature_distances.empty
                else None
            ),
        }
    )
    if result.prior_labels is not None:
        metrics["peak_prior_ari"] = float(adjusted_rand_score(result.prior_labels, final_peak))
        metrics["peak_prior_nmi"] = float(normalized_mutual_info_score(result.prior_labels, final_peak))
        metrics["peak_prior_changed_fraction"] = float(np.mean(result.prior_labels != final_peak))
    else:
        metrics["peak_prior_ari"] = None
        metrics["peak_prior_nmi"] = None
        metrics["peak_prior_changed_fraction"] = None
    if result.initial_domain_labels is not None:
        metrics["initial_domain_ari"] = float(adjusted_rand_score(result.initial_domain_labels, final_domain))
        metrics["initial_domain_nmi"] = float(normalized_mutual_info_score(result.initial_domain_labels, final_domain))
        metrics["initial_domain_changed_fraction"] = float(np.mean(result.initial_domain_labels != final_domain))
    else:
        metrics["initial_domain_ari"] = None
        metrics["initial_domain_nmi"] = None
        metrics["initial_domain_changed_fraction"] = None
    return metrics


def _block_parameters(
    A: np.ndarray,
    FtB: np.ndarray,
    F2tB: np.ndarray,
    UtB: np.ndarray,
    nB: np.ndarray,
    variance_floor: float,
    min_peak_module_mass: float = 1.0,
    min_domain_mass: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # A nearly collapsed soft component can otherwise produce unstable block
    # moments from a tiny denominator. The floors act as zero-mean pseudomass;
    # they do not alter A or B and therefore preserve the learned assignments.
    nA = np.maximum(A.sum(axis=0), float(min_peak_module_mass))
    nB = np.maximum(nB, float(min_domain_mass))
    denom = nA[:, None] * nB[None, :]
    mu = (A.T @ FtB) / denom
    ubar = (A.T @ UtB) / denom
    second = (A.T @ F2tB) / denom
    var = second - mu * mu + ubar + float(variance_floor)
    var = np.maximum(var, float(variance_floor))
    return mu, var, ubar


def _peak_evidence(FtB: np.ndarray, F2tB: np.ndarray, nB: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    inv_var = 1.0 / np.maximum(var, 1.0e-12)
    log_var = np.log(np.maximum(var, 1.0e-12))
    const = log_var @ nB
    term1 = F2tB @ inv_var.T
    term2 = FtB @ (mu * inv_var).T
    term3 = (mu * mu * inv_var) @ nB
    return -0.5 * (term1 - 2.0 * term2 + term3[None, :] + const[None, :])


def _spot_evidence(FA: np.ndarray, F2A: np.ndarray, nA: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    inv_var = 1.0 / np.maximum(var, 1.0e-12)
    log_var = np.log(np.maximum(var, 1.0e-12))
    const = nA @ log_var + nA @ (mu * mu * inv_var)
    term1 = F2A @ inv_var
    term2 = FA @ (mu * inv_var)
    return -0.5 * (term1 - 2.0 * term2 + const[None, :])


def _undirected_edges(graph: sparse.spmatrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    graph = sparse.triu(graph.tocsr(), k=1).tocoo()
    edge_w = graph.data.astype(np.float32, copy=False)
    if edge_w.size:
        strengths = np.asarray((graph + graph.T).sum(axis=1)).ravel()
        positive = strengths[strengths > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        edge_w = edge_w / max(scale, 1.0e-6)
    return graph.row.astype(np.int64), graph.col.astype(np.int64), edge_w


def _edge_roughness_scales(
    module_maps: np.ndarray,
    edges_u: np.ndarray,
    edges_v: np.ndarray,
    *,
    random_state: int,
    max_edges: int = 200_000,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    if edges_u.size > max_edges:
        idx = rng.choice(edges_u.size, size=max_edges, replace=False)
        u = edges_u[idx]
        v = edges_v[idx]
    else:
        u = edges_u
        v = edges_v
    dz = module_maps[u].astype(np.float64, copy=False) - module_maps[v].astype(np.float64, copy=False)
    rough = dz * dz
    scales = np.percentile(rough, 90, axis=0)
    scales = np.maximum(scales, 1.0e-6)
    return scales.astype(np.float64)


def _row_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(logits, -80.0, 80.0))
    denom = exp.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return exp / denom


def _top_two_margin(logits: np.ndarray) -> np.ndarray:
    """Return the largest-minus-second-largest logit for every row."""

    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("Expected a nonempty two-dimensional logit matrix")
    if logits.shape[1] == 1:
        return np.full(logits.shape[0], np.inf, dtype=np.float64)
    top_two = np.partition(logits, kth=logits.shape[1] - 2, axis=1)[:, -2:]
    return top_two[:, 1] - top_two[:, 0]


def _smoothed_one_hot(labels: np.ndarray, n_labels: int, smooth: float) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    out = np.full((labels.size, n_labels), float(smooth) / max(n_labels, 1), dtype=np.float32)
    out[np.arange(labels.size), labels] += 1.0 - float(smooth)
    out /= out.sum(axis=1, keepdims=True)
    return out


def _smooth_responsibilities(responsibilities: np.ndarray, smooth: float) -> np.ndarray:
    """Mix soft responsibilities with a uniform distribution."""

    responsibilities = np.asarray(responsibilities, dtype=np.float64)
    if responsibilities.ndim != 2 or responsibilities.shape[1] == 0:
        raise ValueError("Responsibilities must be a nonempty two-dimensional matrix")
    if not 0.0 <= float(smooth) < 1.0:
        raise ValueError("smooth must be in [0, 1)")
    normalized = responsibilities / np.maximum(
        responsibilities.sum(axis=1, keepdims=True),
        1.0e-12,
    )
    mixed = (1.0 - float(smooth)) * normalized + float(smooth) / responsibilities.shape[1]
    return mixed / np.maximum(mixed.sum(axis=1, keepdims=True), 1.0e-12)


def _capacity_constrained_labels(
    logits: np.ndarray,
    *,
    capacity_factor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign hard labels with a maximum domain capacity.

    This is a constrained hard-EM step for spot domains. It keeps the usual
    per-spot logit preference, but blocks assignments to domains that have
    already reached ``capacity_factor * N / R`` spots. The goal is not perfect
    equal-size clustering; it is to stop one broad domain from absorbing most
    ambiguous spots when several domains have similar evidence.
    """

    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("Expected logits with shape n_spots x n_domains")
    n_spots, n_domains = logits.shape
    if n_spots == 0 or n_domains == 0:
        raise ValueError("Cannot assign empty logits")
    if capacity_factor < 1.0:
        raise ValueError("domain_hard_capacity_factor must be >= 1.0 when enabled")
    max_size = int(np.ceil(float(capacity_factor) * n_spots / n_domains))
    max_size = max(max_size, int(np.ceil(n_spots / n_domains)))
    preferences = np.argsort(-logits, axis=1)
    if n_domains > 1:
        top = logits[np.arange(n_spots), preferences[:, 0]]
        second = logits[np.arange(n_spots), preferences[:, 1]]
        confidence = top - second
    else:
        confidence = np.zeros(n_spots, dtype=np.float64)
    jitter = rng.random(n_spots) * 1.0e-9
    order = np.argsort(-(confidence + jitter))
    counts = np.zeros(n_domains, dtype=np.int64)
    labels = np.full(n_spots, -1, dtype=np.int64)
    for idx in order:
        assigned = False
        for domain in preferences[idx]:
            domain = int(domain)
            if counts[domain] < max_size:
                labels[idx] = domain
                counts[domain] += 1
                assigned = True
                break
        if not assigned:
            domain = int(np.argmin(counts))
            labels[idx] = domain
            counts[domain] += 1
    return labels


def _capped_softmax(
    logits: np.ndarray,
    *,
    capacity_factor: float,
    temperature: float,
    step: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Soft capacity-constrained row assignment.

    This is an entropy-regularized relaxation of the greedy hard-cap update.
    Rows still sum to one, but columns that exceed ``capacity_factor * N / R``
    receive a nonnegative dual penalty before the next row-softmax pass.
    """

    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("Expected logits with shape n_spots x n_domains")
    n_spots, n_domains = logits.shape
    if n_spots == 0 or n_domains == 0:
        raise ValueError("Cannot assign empty logits")
    if capacity_factor < 1.0:
        raise ValueError("domain_capacity_factor must be >= 1.0 for capped-softmax")
    if temperature <= 0.0:
        raise ValueError("capped-softmax temperature must be positive")
    if max_iter < 1:
        raise ValueError("capped_softmax_iter must be >= 1")

    cap = np.full(n_domains, float(capacity_factor) * n_spots / n_domains, dtype=np.float64)
    penalty = np.zeros(n_domains, dtype=np.float64)
    step = float(step)
    tol_abs = max(float(tol) * n_spots, 0.0)

    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        B = _row_softmax((logits - penalty[None, :]) / temperature)
        size = B.sum(axis=0)
        positive_violation = np.maximum(size - cap, 0.0)
        if float(np.max(positive_violation)) <= tol_abs:
            break
        penalty = np.maximum(0.0, penalty + step * (size - cap) / np.maximum(cap, 1.0))

    B = _row_softmax((logits - penalty[None, :]) / temperature)
    size = B.sum(axis=0)
    positive_violation = np.maximum(size - cap, 0.0)
    stats = {
        "capped_softmax_iterations": float(iterations),
        "capped_softmax_max_violation": float(np.max(positive_violation)),
        "capped_softmax_mean_positive_violation": float(positive_violation.mean()),
        "capped_softmax_active_penalties": float(np.sum(penalty > 0.0)),
        "capped_softmax_max_penalty": float(np.max(penalty)),
    }
    return B.astype(np.float32, copy=False), stats


def _align_labels_to_count(labels: np.ndarray | None, n_labels: int) -> np.ndarray | None:
    if labels is None:
        return None
    labels = np.asarray(labels, dtype=np.int64)
    unique = sorted(np.unique(labels).tolist())
    if len(unique) <= n_labels and set(unique) != set(range(len(unique))):
        mapping = {old: new for new, old in enumerate(unique)}
        return np.array([mapping[x] for x in labels], dtype=np.int64)
    if labels.max(initial=-1) >= n_labels:
        mapping = {old: idx % n_labels for idx, old in enumerate(unique)}
        return np.array([mapping[x] for x in labels], dtype=np.int64)
    return labels


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    std[std < 1.0e-8] = 1.0
    return (matrix - mean) / std


def _zscore_columns(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    mean = matrix.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = matrix.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std[std < 1.0e-8] = 1.0
    return (matrix - mean) / std


def _weighted_zscore_columns(matrix: np.ndarray, row_weights: np.ndarray) -> np.ndarray:
    """Column z-scores using the same observation weights as domain K-means."""

    matrix64 = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(row_weights, dtype=np.float64)
    if weights.shape != (matrix64.shape[0],):
        raise ValueError("row_weights must align with matrix rows")
    weights = weights / weights.sum()
    mean = weights @ matrix64
    variance = weights @ np.square(matrix64 - mean[None, :])
    std = np.sqrt(np.maximum(variance, 1.0e-16))
    std[std < 1.0e-8] = 1.0
    return ((matrix64 - mean[None, :]) / std[None, :]).astype(np.float32)


def _entropy(probs: np.ndarray) -> float:
    return float(-np.sum(probs * np.log(np.maximum(probs, 1.0e-12))))


def _row_entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.maximum(probs, 1.0e-12)), axis=1)


def _normalized_row_entropy(probs: np.ndarray) -> np.ndarray:
    n_labels = probs.shape[1]
    if n_labels <= 1:
        return np.zeros(probs.shape[0], dtype=np.float64)
    return _row_entropy(probs) / np.log(n_labels)


def _symmetric_diagonal_gaussian_kl(
    mean_a: np.ndarray,
    variance_a: np.ndarray,
    mean_b: np.ndarray,
    variance_b: np.ndarray,
) -> float:
    """Mean symmetric KL for two products of univariate Gaussians."""

    mean_a = np.asarray(mean_a, dtype=np.float64)
    mean_b = np.asarray(mean_b, dtype=np.float64)
    variance_a = np.maximum(np.asarray(variance_a, dtype=np.float64), 1.0e-12)
    variance_b = np.maximum(np.asarray(variance_b, dtype=np.float64), 1.0e-12)
    squared_difference = np.square(mean_a - mean_b)
    kl_ab = 0.5 * (
        variance_a / variance_b
        + squared_difference / variance_b
        - 1.0
        + np.log(variance_b / variance_a)
    )
    kl_ba = 0.5 * (
        variance_b / variance_a
        + squared_difference / variance_a
        - 1.0
        + np.log(variance_a / variance_b)
    )
    return float(np.mean(0.5 * (kl_ab + kl_ba)))
