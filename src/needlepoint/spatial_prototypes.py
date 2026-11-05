"""Uncertainty-aware prototype clustering of independently fitted spatial maps."""

from __future__ import annotations

import hashlib
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .shared_spatial_factorization import GPPosteriorFactorStore, spatially_balanced_sample


@dataclass
class SpatialPrototypeSettings:
    """Settings shared by functional and Gaussian prototype clustering."""

    method: str = "functional"
    n_prototypes: int = 20
    max_iter: int = 20
    n_init: int = 3
    beta: float = 10.0
    tau2: float = 0.1
    update_tau2: bool = False
    tau2_min: float = 1.0e-4
    tau2_max: float = 10.0
    gaussian_evidence_scale: float | None = None
    gaussian_evidence_power: float = 1.0
    mixture_weight_mode: str = "learned"
    mixture_smoothing: float = 1.0
    responsibility_damping: float = 1.0
    tolerance: float = 1.0e-5
    spot_chunk_size: int = 4096
    preload_to_device: bool = True
    initialization_spots: int = 5000
    initialization_smoothing: float = 0.02
    initialization_method: str = "spatial-sample"
    initialization_embedding_dir: str | None = None
    merge_similar_prototypes: bool = True
    merge_map_correlation_threshold: float = 0.9
    merge_membership_correlation_threshold: float = -1.0
    merge_linkage: str = "complete"
    min_prototypes_after_merge: int = 2
    device: str | None = None
    dtype: str = "float32"
    random_state: int = 0

    def validate(self) -> None:
        if self.method not in {"functional", "gaussian"}:
            raise ValueError("method must be functional or gaussian")
        if self.n_prototypes <= 1 or self.max_iter <= 0 or self.n_init <= 0:
            raise ValueError("n_prototypes must exceed one; max_iter and n_init must be positive")
        if self.beta <= 0.0 or self.tau2 <= 0.0:
            raise ValueError("beta and tau2 must be positive")
        if self.tau2_min <= 0.0 or self.tau2_max <= self.tau2_min:
            raise ValueError("invalid tau2 bounds")
        if self.gaussian_evidence_scale is not None and self.gaussian_evidence_scale <= 0.0:
            raise ValueError("gaussian_evidence_scale must be positive")
        if self.gaussian_evidence_power <= 0.0:
            raise ValueError("gaussian_evidence_power must be positive")
        if self.mixture_weight_mode not in {"learned", "uniform"}:
            raise ValueError("mixture_weight_mode must be learned or uniform")
        if self.method == "functional" and self.update_tau2:
            raise ValueError("update_tau2 is defined only for the Gaussian prototype model")
        if self.mixture_smoothing < 0.0:
            raise ValueError("mixture_smoothing must be nonnegative")
        if not 0.0 < self.responsibility_damping <= 1.0:
            raise ValueError("responsibility_damping must be in (0, 1]")
        if self.tolerance < 0.0 or self.spot_chunk_size <= 0 or self.initialization_spots <= 0:
            raise ValueError("invalid convergence or chunk settings")
        if not 0.0 <= self.initialization_smoothing < 1.0:
            raise ValueError("initialization_smoothing must be in [0, 1)")
        if self.initialization_method not in {"spatial-sample", "multiscale-sketch"}:
            raise ValueError("Unknown prototype initialization method")
        if (self.initialization_method == "multiscale-sketch") != bool(self.initialization_embedding_dir):
            raise ValueError("multiscale-sketch requires its validated embedding directory only")
        for name, value in (
            ("merge_map_correlation_threshold", self.merge_map_correlation_threshold),
            (
                "merge_membership_correlation_threshold",
                self.merge_membership_correlation_threshold,
            ),
        ):
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")
        if self.merge_linkage != "complete":
            raise ValueError("Only complete prototype-merge linkage is currently supported")
        if not 2 <= self.min_prototypes_after_merge <= self.n_prototypes:
            raise ValueError(
                "min_prototypes_after_merge must be between 2 and n_prototypes"
            )


@dataclass
class SpatialPrototypeResult:
    responsibilities: np.ndarray
    prototype_maps: np.ndarray
    mixture_weights: np.ndarray
    tau2: np.ndarray
    distances: np.ndarray
    history: pd.DataFrame
    metadata: pd.DataFrame
    spots: pd.DataFrame
    objective: float
    output_dir: Path
    initialization_report: dict | None = None
    null_distances: np.ndarray | None = None


def fit_spatial_prototypes(
    store: GPPosteriorFactorStore,
    output_dir: Path,
    settings: SpatialPrototypeSettings | None = None,
) -> SpatialPrototypeResult:
    """Fit peak-pattern prototypes without refitting the feature-specific GPs."""
    import torch

    settings = settings or SpatialPrototypeSettings()
    settings.validate()
    if settings.n_prototypes > store.n_features:
        raise ValueError("n_prototypes cannot exceed the number of peaks")
    if store.normalization_mode != "shape":
        raise ValueError("Prototype clustering requires shape-normalized GP maps")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = store.prepare()
    started = time.time()
    device = torch.device(
        settings.device
        if settings.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    dtype = torch.float64 if settings.dtype == "float64" else torch.float32
    if device.type == "cuda":
        torch.cuda.set_device(device.index)
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(settings.random_state)
    rng = np.random.default_rng(settings.random_state)
    initialization_report = {"method": settings.initialization_method}
    if settings.initialization_method == "multiscale-sketch":
        from .multiscale_prototypes import load_validated_embedding, torch_kmeans_labels
        initialization_matrix, sketch_manifest = load_validated_embedding(
            settings.initialization_embedding_dir, store, cache_manifest
        )
        initialization_report.update(sketch=sketch_manifest,
            note="Embedding and GPU K-means change initialization only; full-field updates unchanged")
    else:
        sample_rows = spatially_balanced_sample(
            store.spots[["x", "y"]].to_numpy(dtype=np.float64),
            min(settings.initialization_spots, store.n_spots),
        )
        sample_maps, _ = store.read_all_features(sample_rows)
        initialization_matrix = sample_maps.T.astype(np.float64, copy=False)
    tensor_cache = (
        _preload_to_device(torch, store, device, dtype, settings.spot_chunk_size)
        if settings.preload_to_device and device.type == "cuda"
        else None
    )
    functional_precision = (
        _functional_total_precision(
            torch, store, settings.tau2, settings, device, dtype, tensor_cache
        ) if settings.method == "functional" else None
    )
    null_distances = (
        _functional_null_distances(torch, store, settings, device, dtype, tensor_cache)
        if settings.method == "functional" else None
    )

    best: dict | None = None
    all_histories: list[pd.DataFrame] = []
    for initialization in range(settings.n_init):
        seed = int(settings.random_state + 1009 * initialization)
        init_started = time.time()
        if settings.initialization_method == "multiscale-sketch":
            labels = torch_kmeans_labels(initialization_matrix, settings.n_prototypes, seed, device)
        else:
            labels = KMeans(
                n_clusters=settings.n_prototypes,
                n_init=1,
                random_state=seed,
            ).fit_predict(initialization_matrix)
        initialization_seconds = time.time() - init_started
        gamma = np.full(
            (store.n_features, settings.n_prototypes),
            settings.initialization_smoothing / settings.n_prototypes,
            dtype=np.float64,
        )
        gamma[np.arange(store.n_features), labels] += 1.0 - settings.initialization_smoothing
        gamma_t = torch.as_tensor(gamma, dtype=dtype, device=device)
        tau2_t = torch.full(
            (settings.n_prototypes,), float(settings.tau2), dtype=dtype, device=device
        )
        omega_t = _mixture_weights(
            torch, gamma_t, settings.mixture_smoothing, settings.mixture_weight_mode
        )
        records = []
        final_distances = None
        for iteration in range(1, settings.max_iter + 1):
            iteration_started = time.time()
            prototype_t = _prototype_barycenters(
                torch, store, gamma_t, tau2_t, settings, device, dtype, tensor_cache,
                functional_precision=functional_precision,
            )
            if settings.update_tau2:
                tau2_t = _update_tau2(
                    torch,
                    store,
                    gamma_t,
                    prototype_t,
                    tau2_t,
                    settings,
                    device,
                    dtype,
                    tensor_cache,
                )
                prototype_t = _prototype_barycenters(
                    torch, store, gamma_t, tau2_t, settings, device, dtype, tensor_cache
                )
            scores_t, distances_t = _responsibility_scores(
                torch, store, prototype_t, tau2_t, omega_t, settings, device, dtype, tensor_cache
            )
            proposed = torch.softmax(scores_t, dim=1)
            if settings.responsibility_damping < 1.0:
                proposed = (
                    (1.0 - settings.responsibility_damping) * gamma_t
                    + settings.responsibility_damping * proposed
                )
                proposed = proposed / proposed.sum(dim=1, keepdim=True)
            change = float(torch.max(torch.abs(proposed - gamma_t)).detach().cpu())
            gamma_t = proposed
            omega_t = _mixture_weights(
                torch, gamma_t, settings.mixture_smoothing, settings.mixture_weight_mode
            )
            # Distances remain valid after updating the mixture weights. Keep
            # the objective in the current parameter state, including the
            # Dirichlet pseudocount term that the mixture update optimizes.
            if settings.method == "functional":
                updated_scores = omega_t.clamp_min(1.0e-30).log()[None, :] - settings.beta * distances_t
            else:
                evidence_scale = (
                    settings.gaussian_evidence_power / store.n_spots
                    if settings.gaussian_evidence_scale is None
                    else settings.gaussian_evidence_scale
                )
                updated_scores = omega_t.clamp_min(1.0e-30).log()[None, :] - evidence_scale * store.n_spots * distances_t
            posterior_residual = float(
                (torch.softmax(updated_scores, dim=1) - gamma_t).abs().max().detach().cpu()
            )
            entropy = -(gamma_t * gamma_t.clamp_min(1.0e-30).log()).sum(dim=1)
            objective = float(
                _prototype_objective(torch, gamma_t, updated_scores, omega_t, settings)
                .detach()
                .cpu()
            )
            records.append(
                {
                    "initialization": initialization,
                    "iteration": iteration,
                    "objective": objective,
                    "max_responsibility_change": change,
                    "posterior_fixed_point_residual": posterior_residual,
                    "median_max_responsibility": float(
                        gamma_t.max(dim=1).values.median().detach().cpu()
                    ),
                    "median_normalized_entropy": float(
                        (entropy / math.log(settings.n_prototypes)).median().detach().cpu()
                    ),
                    "minimum_soft_cluster_mass": float(gamma_t.sum(dim=0).min().detach().cpu()),
                    "maximum_soft_cluster_mass": float(gamma_t.sum(dim=0).max().detach().cpu()),
                    "tau2_min": float(tau2_t.min().detach().cpu()),
                    "tau2_median": float(tau2_t.median().detach().cpu()),
                    "tau2_max": float(tau2_t.max().detach().cpu()),
                    "initialization_seconds": initialization_seconds,
                    "iteration_seconds": time.time() - iteration_started,
                }
            )
            print(json.dumps({"stage": "prototype_refinement", "start": initialization,
                "iteration": iteration, "objective": objective, "max_change": change,
                "iteration_seconds": records[-1]["iteration_seconds"]}), flush=True)
            final_distances = distances_t
            if not np.isfinite(objective):
                raise FloatingPointError("Nonfinite prototype objective")
            if change <= settings.tolerance and posterior_residual <= settings.tolerance:
                break
        converged = change <= settings.tolerance and posterior_residual <= settings.tolerance
        if not converged:
            warnings.warn(
                f"Prototype initialization {initialization} reached {settings.max_iter} iterations; "
                f"membership change={change:.6g}, fixed-point residual={posterior_residual:.6g}",
                RuntimeWarning,
            )
        # Export memberships scored by exactly the exported centroids and pi.
        # A capped run is valid to inspect but explicitly marked unconverged.
        gamma_t = torch.softmax(updated_scores, dim=1)
        objective = float(_prototype_objective(torch, gamma_t, updated_scores, omega_t, settings).detach().cpu())
        records[-1]["converged"] = converged
        records[-1]["stop_reason"] = "membership_tolerance" if converged else "iteration_cap"
        history = pd.DataFrame(records)
        all_histories.append(history)
        candidate = {
            "objective": objective,
            "gamma": gamma_t.detach().cpu().numpy(),
            "prototype": prototype_t.detach().cpu().numpy(),
            "omega": omega_t.detach().cpu().numpy(),
            "tau2": tau2_t.detach().cpu().numpy(),
            "distances": final_distances.detach().cpu().numpy(),
            "initialization": initialization,
            "history": history,
        }
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate
    assert best is not None
    combined_history = pd.concat(all_histories, ignore_index=True)
    result = SpatialPrototypeResult(
        responsibilities=np.asarray(best["gamma"], dtype=np.float32),
        prototype_maps=np.asarray(best["prototype"], dtype=np.float32),
        mixture_weights=np.asarray(best["omega"], dtype=np.float32),
        tau2=np.asarray(best["tau2"], dtype=np.float32),
        distances=np.asarray(best["distances"], dtype=np.float32),
        history=combined_history,
        metadata=store.metadata.copy(),
        spots=store.spots.copy(),
        objective=float(best["objective"]),
        output_dir=output_dir,
        initialization_report=initialization_report,
        null_distances=null_distances,
    )
    merge_report = None
    if settings.merge_similar_prototypes:
        groups, merge_report = identify_similar_prototype_groups(
            result.prototype_maps,
            result.responsibilities,
            map_correlation_threshold=settings.merge_map_correlation_threshold,
            membership_correlation_threshold=settings.merge_membership_correlation_threshold,
            min_groups=settings.min_prototypes_after_merge,
        )
        if len(groups) < result.responsibilities.shape[1]:
            unmerged = SpatialPrototypeResult(
                **{
                    **result.__dict__,
                    "output_dir": output_dir / "unmerged",
                }
            )
            _write_prototype_outputs(
                unmerged,
                store,
                settings,
                cache_manifest,
                int(best["initialization"]),
                started,
                device,
                torch,
            )
            result = collapse_spatial_prototypes(result, groups, output_dir)
    _write_prototype_outputs(
        result,
        store,
        settings,
        cache_manifest,
        int(best["initialization"]),
        started,
        device,
        torch,
        merge_report=merge_report,
    )
    return result


def identify_similar_prototype_groups(
    prototype_maps: np.ndarray,
    responsibilities: np.ndarray,
    *,
    map_correlation_threshold: float = 0.9,
    membership_correlation_threshold: float = -1.0,
    min_groups: int = 2,
) -> tuple[list[list[int]], dict]:
    """Find redundant prototypes using deterministic complete-linkage merging.

    A merge is allowed only when every cross-pair between two candidate groups
    passes the map-correlation threshold and any enabled membership threshold.
    Membership correlation is disabled by default because responsibilities
    compete within each row and can be anticorrelated even for duplicate maps.
    Complete linkage avoids connected-component or single-linkage chaining.
    """

    maps = np.asarray(prototype_maps, dtype=np.float64)
    gamma = np.asarray(responsibilities, dtype=np.float64)
    if maps.ndim != 2 or gamma.ndim != 2 or maps.shape[1] != gamma.shape[1]:
        raise ValueError(
            "prototype_maps and responsibilities must be two-dimensional with "
            "the same prototype count"
        )
    n_prototypes = maps.shape[1]
    if not 2 <= min_groups <= n_prototypes:
        raise ValueError("min_groups must be between 2 and the prototype count")
    map_correlation = _safe_column_correlation(maps)
    membership_correlation = _safe_column_correlation(gamma)
    groups = [[index] for index in range(n_prototypes)]

    while len(groups) > min_groups:
        candidate = None
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                cross = np.ix_(groups[left], groups[right])
                minimum_map = float(map_correlation[cross].min())
                minimum_membership = float(membership_correlation[cross].min())
                if (
                    minimum_map < map_correlation_threshold
                    or minimum_membership < membership_correlation_threshold
                ):
                    continue
                membership_tiebreaker = (
                    minimum_membership
                    if membership_correlation_threshold > -1.0
                    else 0.0
                )
                score = (
                    minimum_map,
                    membership_tiebreaker,
                    -groups[left][0],
                    -groups[right][0],
                )
                if candidate is None or score > candidate[0]:
                    candidate = (score, left, right)
        if candidate is None:
            break
        _, left, right = candidate
        groups[left] = sorted(groups[left] + groups[right])
        del groups[right]

    groups = sorted(groups, key=lambda group: group[0])
    source_to_merged = np.empty(n_prototypes, dtype=np.int64)
    for merged, group in enumerate(groups):
        source_to_merged[group] = merged
    return groups, {
        "groups": groups,
        "source_to_merged": source_to_merged,
        "map_correlation": map_correlation,
        "membership_correlation": membership_correlation,
        "map_correlation_threshold": float(map_correlation_threshold),
        "membership_correlation_threshold": float(membership_correlation_threshold),
        "linkage": "complete",
        "premerge_n_prototypes": int(n_prototypes),
        "effective_n_prototypes": int(len(groups)),
        "n_merged_away": int(n_prototypes - len(groups)),
    }


def collapse_spatial_prototypes(
    result: SpatialPrototypeResult,
    groups: list[list[int]],
    output_dir: Path | None = None,
) -> SpatialPrototypeResult:
    """Collapse fitted components while preserving each peak's probability mass."""

    gamma = np.asarray(result.responsibilities, dtype=np.float64)
    maps = np.asarray(result.prototype_maps, dtype=np.float64)
    distances = np.asarray(result.distances, dtype=np.float64)
    component_mass = gamma.sum(axis=0)
    merged_gamma = np.column_stack([gamma[:, group].sum(axis=1) for group in groups])
    merged_maps = []
    merged_weights = []
    merged_tau2 = []
    merged_distances = []
    for group in groups:
        mass = component_mass[group]
        normalized_mass = mass / max(float(mass.sum()), 1.0e-30)
        merged_maps.append(maps[:, group] @ normalized_mass)
        merged_weights.append(float(np.asarray(result.mixture_weights)[group].sum()))
        merged_tau2.append(float(np.dot(np.asarray(result.tau2)[group], normalized_mass)))
        local_gamma = gamma[:, group]
        local_mass = local_gamma.sum(axis=1)
        expected_distance = np.sum(local_gamma * distances[:, group], axis=1) / np.maximum(
            local_mass, 1.0e-30
        )
        merged_distances.append(expected_distance)
    mixture_weights = np.asarray(merged_weights, dtype=np.float64)
    mixture_weights /= max(float(mixture_weights.sum()), 1.0e-30)
    return SpatialPrototypeResult(
        responsibilities=merged_gamma.astype(np.float32),
        prototype_maps=np.column_stack(merged_maps).astype(np.float32),
        mixture_weights=mixture_weights.astype(np.float32),
        tau2=np.asarray(merged_tau2, dtype=np.float32),
        distances=np.column_stack(merged_distances).astype(np.float32),
        history=result.history.copy(),
        metadata=result.metadata.copy(),
        spots=result.spots.copy(),
        objective=float(result.objective),
        output_dir=Path(output_dir or result.output_dir),
        initialization_report=result.initialization_report,
        null_distances=result.null_distances,
    )


def merge_spatial_prototype_directory(
    source_dir: Path,
    output_dir: Path,
    *,
    map_correlation_threshold: float = 0.9,
    membership_correlation_threshold: float = -1.0,
    min_groups: int = 2,
) -> SpatialPrototypeResult:
    """Apply the conservative merge stage to an existing prototype fit."""

    import torch

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    manifest = json.loads((source_dir / "prototype_manifest.json").read_text())
    assignments = pd.read_csv(source_dir / "prototype_assignments.csv")
    metadata_columns = [
        column
        for column in assignments.columns
        if not column.startswith("gamma_")
        and column
        not in {
            "prototype_cluster",
            "prototype_probability",
            "prototype_entropy",
            "prototype_distance",
        }
    ]
    summary = pd.read_csv(source_dir / "prototype_summary.csv")
    result = SpatialPrototypeResult(
        responsibilities=np.load(source_dir / "prototype_responsibilities.npy"),
        prototype_maps=np.load(source_dir / "prototype_maps.npy", mmap_mode="r"),
        mixture_weights=summary["mixture_weight"].to_numpy(dtype=np.float32),
        tau2=summary["tau2"].to_numpy(dtype=np.float32),
        distances=np.load(source_dir / "prototype_distances.npy"),
        history=pd.read_csv(source_dir / "prototype_history.csv"),
        metadata=assignments[metadata_columns].copy(),
        spots=pd.read_csv(source_dir / "prototype_spots.csv.gz"),
        objective=float(manifest["best_objective"]),
        output_dir=output_dir,
        initialization_report=manifest.get("initialization"),
        null_distances=(assignments['best_constant_distance'].to_numpy()
                        if 'best_constant_distance' in assignments else None),
    )
    groups, merge_report = identify_similar_prototype_groups(
        result.prototype_maps,
        result.responsibilities,
        map_correlation_threshold=map_correlation_threshold,
        membership_correlation_threshold=membership_correlation_threshold,
        min_groups=min_groups,
    )
    merged = collapse_spatial_prototypes(result, groups, output_dir)
    settings_payload = dict(manifest.get("settings", {}))
    settings_payload.update(
        {
            "merge_similar_prototypes": True,
            "merge_map_correlation_threshold": map_correlation_threshold,
            "merge_membership_correlation_threshold": membership_correlation_threshold,
            "merge_linkage": "complete",
            "min_prototypes_after_merge": min_groups,
        }
    )
    settings = SpatialPrototypeSettings(**settings_payload)
    input_metadata = manifest["input"]
    store = SimpleNamespace(
        reference_vski_dir=Path(input_metadata["reference_vski_dir"]),
        selected_models=None
        if input_metadata.get("selected_models") is None
        else Path(input_metadata["selected_models"]),
        n_spots=int(input_metadata["n_spots"]),
        n_features=int(input_metadata["n_features"]),
        feature_ids=merged.metadata["gene"].astype(str).to_numpy(),
    )
    _write_prototype_outputs(
        merged,
        store,
        settings,
        manifest.get("cache", {}),
        int(manifest.get("best_initialization", 0)),
        time.time(),
        torch.device("cpu"),
        torch,
        merge_report=merge_report,
    )
    merged_manifest_file = output_dir / "prototype_manifest.json"
    merged_manifest = json.loads(merged_manifest_file.read_text())
    # A post-fit aggregation must not relabel a historical optimizer as the
    # current implementation or claim that its merged parameters were fitted.
    merged_manifest["optimizer_version"] = manifest.get("optimizer_version", "legacy-unspecified")
    merged_manifest["convergence"] = {
        **manifest.get("convergence", {"converged": None, "stop_reason": "unknown-legacy"}),
        "final_state": "posthoc aggregated memberships; no optimization performed",
    }
    merged_manifest["posthoc_merge_source"] = str(source_dir.resolve())
    if merged_manifest.get("prototype_merge") is not None:
        merged_manifest["prototype_merge"]["unmerged_artifacts"] = str(
            source_dir.resolve()
        )
    merged_manifest_file.write_text(json.dumps(merged_manifest, indent=2), encoding="utf-8")
    return merged


def _safe_column_correlation(values: np.ndarray) -> np.ndarray:
    correlation = np.corrcoef(np.asarray(values, dtype=np.float64).T)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _preload_to_device(torch, store, device, dtype, spot_chunk_size):
    """Load the reusable map cache once; repeated EM passes then stay on GPU."""

    mean_cache = torch.empty((store.n_spots, store.n_features), dtype=dtype, device=device)
    variance_cache = torch.empty_like(mean_cache)
    features = np.arange(store.n_features, dtype=np.int64)
    for start in range(0, store.n_spots, spot_chunk_size):
        stop = min(start + spot_chunk_size, store.n_spots)
        rows = np.arange(start, stop, dtype=np.int64)
        mean_np, variance_np = store.read_block(rows, features)
        mean_cache[start:stop] = torch.as_tensor(mean_np, dtype=dtype, device=device)
        variance_cache[start:stop] = torch.as_tensor(variance_np, dtype=dtype, device=device)
    return mean_cache, variance_cache


def _read_tensor_block(torch, store, rows, features, device, dtype, tensor_cache):
    if tensor_cache is not None:
        start = int(rows[0])
        stop = int(rows[-1]) + 1
        if not np.array_equal(rows, np.arange(start, stop, dtype=np.int64)):
            index = torch.as_tensor(rows, dtype=torch.long, device=device)
            return tensor_cache[0].index_select(0, index), tensor_cache[1].index_select(0, index)
        return tensor_cache[0][start:stop], tensor_cache[1][start:stop]
    mean_np, variance_np = store.read_block(rows, features)
    return (
        torch.as_tensor(mean_np, dtype=dtype, device=device),
        torch.as_tensor(variance_np, dtype=dtype, device=device),
    )


def _prototype_barycenters(
    torch, store, gamma, tau2, settings, device, dtype, tensor_cache=None,
    functional_precision=None,
):
    if settings.method == "functional" and functional_precision is None:
        functional_precision = _functional_total_precision(
            torch, store, float(tau2[0]), settings, device, dtype, tensor_cache
        )
    prototype = torch.empty(
        (store.n_spots, settings.n_prototypes), dtype=dtype, device=device
    )
    features = np.arange(store.n_features, dtype=np.int64)
    for start in range(0, store.n_spots, settings.spot_chunk_size):
        stop = min(start + settings.spot_chunk_size, store.n_spots)
        rows = np.arange(start, stop, dtype=np.int64)
        mean, variance = _read_tensor_block(
            torch, store, rows, features, device, dtype, tensor_cache
        )
        if settings.method == "functional":
            weight = (variance + tau2[0]).reciprocal() / functional_precision[None, :]
            numerator = (weight * mean) @ gamma
            denominator = weight @ gamma
        else:
            total_variance = variance[:, :, None] + tau2[None, None, :]
            weighted_membership = gamma[None, :, :] / total_variance
            numerator = (weighted_membership * mean[:, :, None]).sum(dim=1)
            denominator = weighted_membership.sum(dim=1)
        prototype[start:stop] = numerator / denominator.clamp_min(1.0e-30)
    return prototype


def _functional_total_precision(torch, store, tau2, settings, device, dtype, tensor_cache=None):
    """Gene precision normalizers for the normalized functional distortion."""
    total = torch.zeros(store.n_features, dtype=torch.float64, device=device)
    features = np.arange(store.n_features, dtype=np.int64)
    for start in range(0, store.n_spots, settings.spot_chunk_size):
        rows = np.arange(start, min(start + settings.spot_chunk_size, store.n_spots))
        _, variance = _read_tensor_block(torch, store, rows, features, device, dtype, tensor_cache)
        total += (variance + tau2).reciprocal().sum(dim=0, dtype=torch.float64)
    if not bool(torch.isfinite(total).all()) or bool((total <= 0).any()):
        raise FloatingPointError("Invalid functional total precision")
    return total.to(dtype=dtype)


def _prototype_objective(torch, gamma, scores, omega, settings):
    value = (gamma * (scores - gamma.clamp_min(1.0e-30).log())).sum()
    if settings.mixture_weight_mode == "learned" and settings.mixture_smoothing:
        value = value + settings.mixture_smoothing / gamma.shape[1] * omega.clamp_min(1.0e-30).log().sum()
    return value


def _functional_null_distances(torch, store, settings, device, dtype, tensor_cache=None):
    """Best weighted constant-map distortion for absolute-fit diagnostics."""
    total = torch.zeros(store.n_features, dtype=torch.float64, device=device)
    wx = torch.zeros_like(total)
    wx2 = torch.zeros_like(total)
    features = np.arange(store.n_features, dtype=np.int64)
    for start in range(0, store.n_spots, settings.spot_chunk_size):
        rows = np.arange(start, min(start + settings.spot_chunk_size, store.n_spots))
        mean, variance = _read_tensor_block(torch, store, rows, features, device, dtype, tensor_cache)
        weight = (variance + settings.tau2).reciprocal()
        weighted_mean = weight * mean
        total += weight.sum(0, dtype=torch.float64)
        wx += weighted_mean.sum(0, dtype=torch.float64)
        wx2 += (weighted_mean * mean).sum(0, dtype=torch.float64)
    value = (wx2 / total - (wx / total).square()).clamp_min(0)
    return value.detach().cpu().numpy()


def _responsibility_scores(
    torch, store, prototype, tau2, omega, settings, device, dtype, tensor_cache=None
):
    n_features = store.n_features
    n_prototypes = settings.n_prototypes
    features = np.arange(n_features, dtype=np.int64)
    distances = torch.zeros((n_features, n_prototypes), dtype=dtype, device=device)
    if settings.method == "functional":
        denominator = torch.zeros(n_features, dtype=dtype, device=device)
        weighted_x2 = torch.zeros(n_features, dtype=dtype, device=device)
        cross = torch.zeros((n_features, n_prototypes), dtype=dtype, device=device)
        weighted_h2 = torch.zeros_like(cross)
        for start in range(0, store.n_spots, settings.spot_chunk_size):
            stop = min(start + settings.spot_chunk_size, store.n_spots)
            rows = np.arange(start, stop, dtype=np.int64)
            mean, variance = _read_tensor_block(
                torch, store, rows, features, device, dtype, tensor_cache
            )
            weight = (variance + tau2[0]).reciprocal()
            weighted_mean = weight * mean
            denominator += weight.sum(dim=0)
            weighted_x2 += (weighted_mean * mean).sum(dim=0)
            cross += weighted_mean.transpose(0, 1) @ prototype[start:stop]
            weighted_h2 += weight.transpose(0, 1) @ prototype[start:stop].square()
        distances = (
            weighted_x2[:, None] - 2.0 * cross + weighted_h2
        ) / denominator[:, None].clamp_min(1.0e-12)
        scores = omega.clamp_min(1.0e-30).log()[None, :] - settings.beta * distances
        return scores, distances

    log_likelihood = torch.zeros_like(distances)
    constant = math.log(2.0 * math.pi)
    for start in range(0, store.n_spots, settings.spot_chunk_size):
        stop = min(start + settings.spot_chunk_size, store.n_spots)
        rows = np.arange(start, stop, dtype=np.int64)
        mean, variance = _read_tensor_block(
            torch, store, rows, features, device, dtype, tensor_cache
        )
        total_variance = variance[:, :, None] + tau2[None, None, :]
        residual = mean[:, :, None] - prototype[start:stop, None, :]
        contribution = -0.5 * (
            constant + total_variance.log() + residual.square() / total_variance
        )
        log_likelihood += contribution.sum(dim=0)
    evidence_scale = (
        float(settings.gaussian_evidence_power) / store.n_spots
        if settings.gaussian_evidence_scale is None
        else float(settings.gaussian_evidence_scale)
    )
    scores = omega.clamp_min(1.0e-30).log()[None, :] + evidence_scale * log_likelihood
    distances = -log_likelihood / max(store.n_spots, 1)
    return scores, distances


def _update_tau2(
    torch, store, gamma, prototype, current_tau2, settings, device, dtype, tensor_cache=None
):
    numerator = torch.zeros(settings.n_prototypes, dtype=dtype, device=device)
    denominator = store.n_spots * gamma.sum(dim=0)
    features = np.arange(store.n_features, dtype=np.int64)
    for start in range(0, store.n_spots, settings.spot_chunk_size):
        stop = min(start + settings.spot_chunk_size, store.n_spots)
        rows = np.arange(start, stop, dtype=np.int64)
        mean, variance = _read_tensor_block(
            torch, store, rows, features, device, dtype, tensor_cache
        )
        residual_signal = (
            mean[:, :, None] - prototype[start:stop, None, :]
        ).square() - variance[:, :, None]
        numerator += (residual_signal * gamma[None, :, :]).sum(dim=(0, 1))
    estimate = numerator / denominator.clamp_min(1.0e-12)
    return estimate.clamp(settings.tau2_min, settings.tau2_max)


def _mixture_weights(torch, gamma, smoothing, mode):
    n_prototypes = gamma.shape[1]
    if mode == "uniform":
        return torch.full(
            (n_prototypes,),
            1.0 / n_prototypes,
            dtype=gamma.dtype,
            device=gamma.device,
        )
    weights = gamma.sum(dim=0) + float(smoothing) / n_prototypes
    return weights / weights.sum()


def _write_prototype_outputs(
    result,
    store,
    settings,
    cache_manifest,
    best_initialization,
    started,
    device,
    torch,
    merge_report=None,
):
    output_dir = result.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "prototype_responsibilities.npy", result.responsibilities)
    np.save(output_dir / "prototype_maps.npy", result.prototype_maps)
    np.save(output_dir / "prototype_distances.npy", result.distances)
    result.spots.to_csv(output_dir / "prototype_spots.csv.gz", index=False)
    result.history.to_csv(output_dir / "prototype_history.csv", index=False)
    assignments = result.metadata.copy()
    n_prototypes = int(result.responsibilities.shape[1])
    hard = result.responsibilities.argmax(axis=1)
    maximum = result.responsibilities.max(axis=1)
    entropy = -np.sum(
        result.responsibilities
        * np.log(np.maximum(result.responsibilities, 1.0e-30)),
        axis=1,
    ) / math.log(n_prototypes)
    assignments["prototype_cluster"] = hard
    assignments["prototype_probability"] = maximum
    assignments["prototype_entropy"] = entropy
    assignments["prototype_distance"] = result.distances[np.arange(store.n_features), hard]
    if result.null_distances is not None:
        assignments["best_constant_distance"] = result.null_distances
        assignments["weighted_reconstruction_r2"] = 1 - assignments["prototype_distance"].to_numpy()/np.maximum(result.null_distances, 1e-20)
    for prototype in range(n_prototypes):
        assignments[f"gamma_{prototype:02d}"] = result.responsibilities[:, prototype]
    assignments.to_csv(output_dir / "prototype_assignments.csv", index=False)
    summary = []
    for prototype in range(n_prototypes):
        members = hard == prototype
        summary.append(
            {
                "prototype": prototype,
                "hard_peak_count": int(members.sum()),
                "soft_peak_mass": float(result.responsibilities[:, prototype].sum()),
                "mixture_weight": float(result.mixture_weights[prototype]),
                "tau2": float(result.tau2[prototype]),
                "median_member_probability": float(np.median(maximum[members]))
                if members.any()
                else float("nan"),
                "median_member_distance": float(
                    np.median(result.distances[members, prototype])
                )
                if members.any()
                else float("nan"),
            }
        )
    pd.DataFrame(summary).to_csv(output_dir / "prototype_summary.csv", index=False)
    if merge_report is not None:
        _write_prototype_merge_outputs(output_dir, merge_report)
    manifest = {
        "format_version": 2,
        "model": "uncertainty-aware spatial prototype clustering",
        "method": settings.method,
        "settings": asdict(settings),
        "initialization": result.initialization_report,
        "requested_n_prototypes": int(settings.n_prototypes),
        "effective_n_prototypes": n_prototypes,
        "prototype_merge": None
        if merge_report is None
        else {
            "enabled": True,
            "linkage": merge_report["linkage"],
            "map_correlation_threshold": merge_report[
                "map_correlation_threshold"
            ],
            "membership_correlation_threshold": merge_report[
                "membership_correlation_threshold"
            ],
            "premerge_n_prototypes": merge_report["premerge_n_prototypes"],
            "effective_n_prototypes": merge_report["effective_n_prototypes"],
            "n_merged_away": merge_report["n_merged_away"],
            "groups": merge_report["groups"],
            "objective_note": (
                "The reported objective is the pre-merge EM objective; the deterministic "
                "post-fit collapse does not refit or recompute it."
            ),
            "unmerged_artifacts": "unmerged"
            if merge_report["n_merged_away"] > 0
            else None,
        },
        "input": {
            "reference_vski_dir": str(store.reference_vski_dir),
            "selected_models": None
            if store.selected_models is None
            else str(store.selected_models),
            "n_spots": store.n_spots,
            "n_features": store.n_features,
            "feature_order_sha256": hashlib.sha256(
                "\n".join(store.feature_ids).encode()
            ).hexdigest(),
            "normalization": "per-peak mean shape normalization",
            "uncertainty": "diagonal inherited feature-GP posterior variance",
        },
        "best_initialization": best_initialization,
        "best_objective": result.objective,
        "optimizer_version": "normalized-functional-centroid-v2" if settings.method == "functional" else "gaussian-centroid-v1",
        "convergence": {
            "converged": bool(result.history.loc[result.history.initialization == best_initialization].iloc[-1].get("converged", False)),
            "stop_reason": str(result.history.loc[result.history.initialization == best_initialization].iloc[-1].get("stop_reason", "unknown")),
            "tolerance_policy": "warn-and-continue-independent-experiments",
            "final_state": (
                "posthoc aggregated memberships; convergence and objective describe the unmerged fit"
                if merge_report is not None else
                "memberships scored with exported pi and centroids; capped runs are not stationary"
            ),
        },
        "cache": cache_manifest,
        "runtime": {
            "elapsed_seconds": float(time.time() - started),
            "device": str(device),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
        },
        "outputs": {
            "responsibilities": "prototype_responsibilities.npy",
            "prototype_maps": "prototype_maps.npy",
            "distances": "prototype_distances.npy",
            "assignments": "prototype_assignments.csv",
            "summary": "prototype_summary.csv",
            "history": "prototype_history.csv",
            "spots": "prototype_spots.csv.gz",
            "merge_mapping": "prototype_merge_mapping.csv"
            if merge_report is not None
            else None,
            "merge_similarity": "prototype_merge_similarity.csv"
            if merge_report is not None
            else None,
            "merge_summary": "prototype_merge_summary.json"
            if merge_report is not None
            else None,
        },
    }
    (output_dir / "prototype_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _write_prototype_merge_outputs(output_dir: Path, report: dict) -> None:
    mapping = pd.DataFrame(
        {
            "source_prototype": np.arange(report["premerge_n_prototypes"], dtype=np.int64),
            "merged_prototype": report["source_to_merged"],
        }
    )
    mapping["merged_group_size"] = mapping.groupby("merged_prototype")[
        "source_prototype"
    ].transform("size")
    mapping.to_csv(output_dir / "prototype_merge_mapping.csv", index=False)

    pair_rows = []
    for left in range(report["premerge_n_prototypes"]):
        for right in range(left + 1, report["premerge_n_prototypes"]):
            same_group = bool(
                report["source_to_merged"][left] == report["source_to_merged"][right]
            )
            pair_rows.append(
                {
                    "source_prototype_a": left,
                    "source_prototype_b": right,
                    "map_correlation": float(report["map_correlation"][left, right]),
                    "membership_correlation": float(
                        report["membership_correlation"][left, right]
                    ),
                    "passes_pair_thresholds": bool(
                        report["map_correlation"][left, right]
                        >= report["map_correlation_threshold"]
                        and report["membership_correlation"][left, right]
                        >= report["membership_correlation_threshold"]
                    ),
                    "merged_together": same_group,
                }
            )
    pd.DataFrame(pair_rows).to_csv(
        output_dir / "prototype_merge_similarity.csv", index=False
    )
    summary = {
        key: value
        for key, value in report.items()
        if key
        not in {
            "source_to_merged",
            "map_correlation",
            "membership_correlation",
        }
    }
    (output_dir / "prototype_merge_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
