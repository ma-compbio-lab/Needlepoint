#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from needlepoint.vski_lbm import (
    AdaptiveVSKILBM,
    CachedSpatialMapStore,
    PairedVSKIRawMapStore,
    RawTEMIMapStore,
    VSKILBM,
    VSKIMapStore,
    SpatialMapStore,
    assignment_metrics,
    block_parameter_table,
    domain_summary_table,
    domain_signature_distance_table,
    domain_initialization_module_weight_table,
    peak_module_table,
    spot_domain_table,
)
from needlepoint.factor_lbm_bridge import load_factor_lbm_bridge
from needlepoint.graph import (
    SpatialGraphConfig,
    audit_spatial_graph,
    build_spatial_graph,
    validate_graph_audit,
)
from needlepoint.artifacts import (
    atomic_write_json,
    require_production_gp_artifacts,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V-SKI-output LBM on TEMI fitted GP maps.")
    parser.add_argument("--input-mode", choices=["vski", "raw"], default="vski")
    parser.add_argument("--vski-dir", type=Path, default=None)
    parser.add_argument(
        "--selected-models",
        type=Path,
        default=None,
        help="Optional per-feature selected GP map table, including model_chunk_path and map_column.",
    )
    parser.add_argument(
        "--allow-nonproduction-fit",
        action="store_true",
        help="Explicitly permit evaluation/unknown GP maps (unsafe for final inference).",
    )
    parser.add_argument(
        "--allow-legacy-full-coverage",
        action="store_true",
        help="Permit legacy GP maps only when their manifest proves full training coverage.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--peak-prior-csv", type=Path, default=None)
    parser.add_argument("--no-peak-prior", action="store_true")
    parser.add_argument("--raw-feature-dir", type=Path, default=None)
    parser.add_argument("--coords-file", type=Path, default=None)
    parser.add_argument("--peak-metadata", type=Path, default=None)
    parser.add_argument("--selected-peak-indices-file", type=Path, default=None)
    parser.add_argument("--target-sum", type=float, default=1.0e4)
    parser.add_argument("--transform", choices=["none", "sqrt", "log1p"], default="none")
    parser.add_argument("--raw-feature-block-size", type=int, default=64)
    parser.add_argument(
        "--map-cache-dir",
        type=Path,
        default=None,
        help="Optional reusable disk-backed cache of aligned, standardized LBM maps.",
    )
    parser.add_argument("--map-cache-feature-block-size", type=int, default=64)
    parser.add_argument(
        "--raw-cache-file",
        type=Path,
        default=None,
        help="Disk-backed aligned raw map cache for posterior-residual; defaults inside output-dir.",
    )
    parser.add_argument(
        "--ignore-posterior-variance",
        action="store_true",
        help="Use V-SKI posterior means only; raw input always has zero posterior uncertainty.",
    )
    parser.add_argument(
        "--spot-init-csv",
        type=Path,
        default=None,
        help=(
            "Optional explicit spot-domain labels. No historical labels are loaded "
            "by default; without this flag B is initialized from module maps."
        ),
    )
    parser.add_argument("--spot-init-column", default="domain_cluster")
    parser.add_argument(
        "--factor-bridge-dir",
        type=Path,
        default=None,
        help="Optional continuous shared-factor bridge for soft initialization and priors.",
    )
    parser.add_argument("--factor-feature-prior-weight", type=float, default=0.0)
    parser.add_argument("--factor-location-prior-weight", type=float, default=0.0)
    parser.add_argument("--factor-feature-prior-temperature", type=float, default=1.0)
    parser.add_argument("--factor-location-prior-temperature", type=float, default=1.0)
    parser.add_argument(
        "--prototype-dir",
        type=Path,
        default=None,
        help="Optional spatial-prototype fit; uses Gamma T as the peak-module prior.",
    )
    parser.add_argument(
        "--initial-peak-responsibilities-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit P x K A^(0). This overrides automatic prototype "
            "initialization and is retained for controlled or historical runs."
        ),
    )
    parser.add_argument("--prototype-transition-smoothing", type=float, default=1.0)
    parser.add_argument("--prototype-transition-damping", type=float, default=0.5)
    parser.add_argument("--prototype-transition-identity-smoothing", type=float, default=0.05)
    parser.add_argument("--prototype-peak-initialization-smoothing", type=float, default=0.05)
    parser.add_argument("--prototype-general-initialization-blend", type=float, default=0.5)
    parser.add_argument("--freeze-prototype-transition", action="store_true")
    parser.add_argument(
        "--domain-initialization",
        choices=["soft-kmeans-module-maps", "hard-kmeans-module-maps"],
        default="soft-kmeans-module-maps",
        help=(
            "Default B initialization when --spot-init-csv is absent. The soft "
            "mode converts distances to K-means centroids in initial module-map space "
            "into responsibilities."
        ),
    )
    parser.add_argument(
        "--domain-initialization-temperature",
        type=float,
        default=None,
        help="T_B for soft module-map K-means initialization; default estimates it robustly.",
    )
    parser.add_argument("--domain-initialization-n-init", type=int, default=20)
    parser.add_argument(
        "--domain-initialization-module-weighting",
        choices=["uniform", "mass", "mass-reliability"],
        default="mass-reliability",
        help=(
            "Weight standardized module-map dimensions used for domain K-means. "
            "The default uses soft peak mass, fitted GP signal/noise reliability, "
            "and posterior uncertainty."
        ),
    )

    parser.add_argument(
        "--n-peak-modules",
        type=int,
        default=None,
        help=(
            "Number of LBM peak modules. By default this matches the prototype "
            "count when --prototype-dir is supplied, otherwise 20."
        ),
    )
    parser.add_argument("--n-domains", type=int, default=12)
    parser.add_argument("--graph-neighbors", type=int, default=16)
    parser.add_argument(
        "--graph-construction",
        choices=["auto", "none", "knn", "radius-knn", "regular-grid"],
        default="auto",
        help=(
            "Graph topology. auto means none when --graph-model=none and knn "
            "otherwise. radius-knn is the bounded physical-distance option."
        ),
    )
    parser.add_argument(
        "--graph-radius",
        type=float,
        default=None,
        help="Maximum coordinate/physical edge distance for radius-knn.",
    )
    parser.add_argument(
        "--graph-distance-bandwidth",
        type=float,
        default=None,
        help=(
            "Fixed exponential-kernel bandwidth in coordinate/physical units; "
            "required for a strictly matched cross-resolution graph."
        ),
    )
    parser.add_argument("--graph-max-isolate-fraction", type=float, default=0.01)
    parser.add_argument(
        "--graph-min-largest-component-fraction", type=float, default=0.5
    )
    parser.add_argument(
        "--graph-min-radius-coverage",
        type=float,
        default=0.0,
        help=(
            "Optional radius-knn gate on realized 95th-percentile edge distance "
            "divided by --graph-radius. Use a positive value (for example 0.8) "
            "to catch a neighbor cap that is too small to realize the intended "
            "physical support."
        ),
    )
    parser.add_argument(
        "--graph-cache-file",
        type=Path,
        default=None,
        help="Optional validated sparse kNN graph cache shared across matched runs.",
    )
    parser.add_argument(
        "--likelihood-scaling",
        choices=["canonical", "legacy-average"],
        default="canonical",
        help="Canonical uses the full likelihood. legacy-average explicitly reproduces the former 1/N and 1/P scaling.",
    )
    parser.add_argument("--likelihood-power-a", type=float, default=1.0, help="Explicit power eta_A for peak evidence.")
    parser.add_argument("--likelihood-power-b", type=float, default=1.0, help="Explicit power eta_B for domain evidence.")
    parser.add_argument("--likelihood-power-coupling", choices=["independent", "shared", "normalized-free-energy"], default="independent")
    parser.add_argument("--shared-likelihood-power", type=float, default=None)
    parser.add_argument("--feature-likelihood-weighting", choices=["uniform", "gp-reliability"], default="uniform")
    parser.add_argument("--graph-module-weighting", choices=["uniform", "initialization"], default="uniform")
    parser.add_argument(
        "--likelihood-model",
        choices=["block-average", "posterior-expected", "posterior-residual"],
        default="block-average",
        help="LBM observation formulation; posterior-residual also requires aligned raw TEMI inputs.",
    )
    parser.add_argument("--lambda-z", type=float, default=0.5, help="Initial decaying V-SKI peak-prior weight.")
    parser.add_argument("--lambda-z-min", type=float, default=0.0, help="Nondecaying floor for the V-SKI peak-prior weight.")
    parser.add_argument(
        "--peak-prior-decay-tau",
        type=float,
        default=5.0,
        help="Iteration time constant for lambda_z(t); 0 keeps lambda_z constant.",
    )
    parser.add_argument(
        "--graph-model",
        choices=["none", "potts", "signed-roughness", "boundary-aware"],
        default="signed-roughness",
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--roughness-weight", type=float, default=0.25)
    parser.add_argument(
        "--graph-boundary-scale",
        type=float,
        default=1.0,
        help="Tau_G in exp(-tau_G * normalized molecular edge roughness).",
    )
    parser.add_argument("--domain-prior-weight", type=float, default=0.0)
    parser.add_argument(
        "--domain-mixture-prior",
        choices=["none", "dirichlet"],
        default="none",
        help="Variational prior over domain proportions; dirichlet is required for adaptive domain selection.",
    )
    parser.add_argument(
        "--domain-dirichlet-alpha0",
        type=float,
        default=1.0,
        help="Total symmetric Dirichlet concentration, distributed as alpha0/R over components.",
    )
    parser.add_argument("--domain-mixture-prior-weight", type=float, default=1.0)
    parser.add_argument("--active-domain-weight-threshold", type=float, default=0.005)
    parser.add_argument("--active-domain-min-spots", type=float, default=1.0)
    parser.add_argument(
        "--domain-update",
        choices=["auto", "softmax", "hardcap", "capped-softmax"],
        default="auto",
        help="Spot-domain update rule. auto preserves legacy behavior: hardcap if a hard capacity factor is set, else softmax.",
    )
    parser.add_argument(
        "--domain-capacity-factor",
        type=float,
        default=0.0,
        help="Capacity factor used by hardcap or capped-softmax. Falls back to --domain-hard-capacity-factor when unset.",
    )
    parser.add_argument(
        "--domain-size-penalty-weight",
        type=float,
        default=0.0,
        help="Penalize already-large spot domains during B updates; useful for encouraging large domains to split.",
    )
    parser.add_argument(
        "--domain-hard-capacity-factor",
        type=float,
        default=0.0,
        help="If >=1, use capacity-constrained hard domain updates with max size factor * n_spots / n_domains.",
    )
    parser.add_argument(
        "--domain-hard-smooth",
        type=float,
        default=0.02,
        help="Row smoothing applied after capacity-constrained hard domain assignment.",
    )
    parser.add_argument("--capped-softmax-iter", type=int, default=50)
    parser.add_argument("--capped-softmax-step", type=float, default=1.0)
    parser.add_argument(
        "--capped-softmax-tol",
        type=float,
        default=1.0e-3,
        help="Stop capped-softmax when max positive soft-size violation is below this fraction of n_spots.",
    )
    parser.add_argument(
        "--final-domain-labeling",
        choices=["argmax", "hardcap"],
        default="argmax",
        help="How to convert final soft B to exported domain labels and maps.",
    )
    parser.add_argument(
        "--final-domain-capacity-factor",
        type=float,
        default=0.0,
        help="Capacity factor for final hardcap labels. Defaults to the domain capacity factor when unset.",
    )
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--variance-floor", type=float, default=1.0e-4)
    parser.add_argument(
        "--min-peak-module-mass",
        type=float,
        default=1.0,
        help="Minimum effective peak mass used in block-moment denominators; assignments remain unchanged.",
    )
    parser.add_argument(
        "--min-domain-mass",
        type=float,
        default=1.0,
        help="Minimum effective spot mass used in block-moment denominators; assignments remain unchanged.",
    )
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-max", type=float, default=0.6)
    parser.add_argument("--a-temperature", type=float, default=1.0)
    parser.add_argument("--b-temperature", type=float, default=1.0)
    parser.add_argument("--edge-batch-size", type=int, default=200_000)
    parser.add_argument("--sample-spots-for-init", type=int, default=60_000)
    parser.add_argument("--peak-init-n-init", type=int, default=20)
    parser.add_argument("--map-standardization", choices=["none", "zscore"], default="zscore")
    parser.add_argument(
        "--compute-backend",
        choices=["numpy", "torch"],
        default="numpy",
        help=(
            "Dense LBM block products and soft domain updates. Use torch with "
            "--compute-device cuda for GPU acceleration; graph messages and hardcap remain NumPy."
        ),
    )
    parser.add_argument("--compute-device", default="cuda")
    parser.add_argument(
        "--compute-dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Torch computation dtype. float32 is recommended for GPU throughput.",
    )
    parser.add_argument(
        "--residual-weight",
        type=float,
        default=0.0,
        help="Tempered raw residual contribution; only valid with posterior-residual.",
    )
    parser.add_argument(
        "--residual-variance",
        choices=["module", "block"],
        default="module",
        help="Share residual variance across domains within a module, or estimate each block separately.",
    )

    parser.add_argument("--adaptive-domains", action="store_true")
    parser.add_argument("--min-domains", type=int, default=4)
    parser.add_argument("--max-domains", type=int, default=20)
    parser.add_argument("--split-merge-rounds", type=int, default=4)
    parser.add_argument("--proposal-max-iter", type=int, default=3)
    parser.add_argument("--final-polish-iter", type=int, default=5)
    parser.add_argument("--split-candidates", type=int, default=1)
    parser.add_argument("--merge-candidates", type=int, default=1)
    parser.add_argument("--split-min-spots", type=int, default=1000)
    parser.add_argument("--split-sample-spots", type=int, default=20000)
    parser.add_argument("--split-min-relative-sse-gain", type=float, default=0.05)
    parser.add_argument("--split-max-edge-cut-fraction", type=float, default=0.35)
    parser.add_argument("--split-min-lcc-fraction", type=float, default=0.25)
    parser.add_argument("--split-temperature", type=float, default=0.5)
    parser.add_argument("--max-median-normalized-entropy", type=float, default=0.95)
    parser.add_argument("--min-signature-symmetric-kl", type=float, default=1.0e-3)
    parser.add_argument(
        "--active-domain-penalty",
        type=float,
        default=None,
        help="Penalty kappa per active domain; default uses 0.5*(2K+1)*log(N) pseudo-BIC.",
    )
    parser.add_argument("--proposal-min-score-improvement", type=float, default=0.0)

    parser.add_argument("--max-spots", type=int, default=None)
    parser.add_argument("--max-peaks", type=int, default=None)
    parser.add_argument(
        "--spot-sampling",
        choices=["prefix", "uniform"],
        default="prefix",
        help="How --max-spots selects V-SKI rows; uniform is preferable for whole-slide smoke tests.",
    )
    parser.add_argument(
        "--max-plot-spots",
        type=int,
        default=0,
        help="Maximum spots per spatial panel; 0 (default) renders all spots.",
    )
    parser.add_argument(
        "--domain-plot-mode",
        choices=["grid", "scatter"],
        default="grid",
        help="Render domain maps as filled coordinate-grid cells by default; scatter is kept for irregular coordinates.",
    )
    parser.add_argument("--point-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.input_mode == "vski" and args.vski_dir is None:
        parser.error("--vski-dir is required with --input-mode vski")
    return args


def main() -> None:
    args = parse_args()
    if args.input_mode == "raw" and args.likelihood_model != "block-average":
        raise ValueError("Raw input mode supports likelihood-model=block-average only")
    if args.selected_models is not None and (
        args.input_mode != "vski" or args.likelihood_model == "posterior-residual"
    ):
        raise ValueError(
            "--selected-models currently supports V-SKI block-average or posterior-expected input"
        )
    started = time.time()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    store = build_store(args)
    if args.map_cache_dir is not None:
        if args.likelihood_model == "posterior-residual":
            raise ValueError("--map-cache-dir does not yet support posterior-residual input")
        store = CachedSpatialMapStore(
            store,
            args.map_cache_dir,
            feature_block_size=args.map_cache_feature_block_size,
        )
    peak_prior_csv = resolve_peak_prior(args)
    peak_prior_labels, peak_prior_distance = load_peak_prior(peak_prior_csv, store) if peak_prior_csv is not None else (None, None)
    initial_domain_labels = load_spot_initialization(args.spot_init_csv, args.spot_init_column, store) if args.spot_init_csv else None
    factor_bridge = load_and_validate_factor_bridge(args.factor_bridge_dir, store)
    if factor_bridge is not None and args.adaptive_domains:
        raise ValueError("The factor bridge currently supports fixed-domain VSKILBM only")
    prototype_gamma = load_prototype_responsibilities(
        args.prototype_dir,
        store,
        allow_nonproduction_fit=args.allow_nonproduction_fit,
        allow_legacy_full_coverage=args.allow_legacy_full_coverage,
    )
    args.n_peak_modules = resolve_n_peak_modules(
        args.n_peak_modules,
        prototype_gamma,
    )
    initial_peak_responsibilities = load_initial_peak_responsibilities(
        args.initial_peak_responsibilities_file,
        store,
        args.n_peak_modules,
    )
    if factor_bridge is not None and initial_peak_responsibilities is not None:
        raise ValueError(
            "Use either --factor-bridge-dir or --initial-peak-responsibilities-file, not both"
        )

    model = VSKILBM(
        n_peak_modules=args.n_peak_modules,
        n_domains=args.n_domains,
        graph_neighbors=args.graph_neighbors,
        graph_construction=args.graph_construction,
        graph_radius=args.graph_radius,
        graph_distance_bandwidth=args.graph_distance_bandwidth,
        lambda_z=args.lambda_z,
        lambda_z_min=args.lambda_z_min,
        peak_prior_decay_tau=args.peak_prior_decay_tau,
        likelihood_power_a=args.likelihood_power_a,
        likelihood_power_b=args.likelihood_power_b,
        likelihood_power_coupling=args.likelihood_power_coupling,
        shared_likelihood_power=args.shared_likelihood_power,
        feature_likelihood_weighting=args.feature_likelihood_weighting,
        graph_module_weighting=args.graph_module_weighting,
        likelihood_scaling=args.likelihood_scaling,
        likelihood_model=args.likelihood_model,
        residual_weight=args.residual_weight,
        residual_variance=args.residual_variance,
        graph_model=args.graph_model,
        beta=args.beta,
        roughness_weight=args.roughness_weight,
        graph_boundary_scale=args.graph_boundary_scale,
        domain_prior_weight=args.domain_prior_weight,
        domain_mixture_prior=args.domain_mixture_prior,
        domain_dirichlet_alpha0=args.domain_dirichlet_alpha0,
        domain_mixture_prior_weight=args.domain_mixture_prior_weight,
        active_domain_weight_threshold=args.active_domain_weight_threshold,
        active_domain_min_spots=args.active_domain_min_spots,
        domain_update=args.domain_update,
        domain_capacity_factor=args.domain_capacity_factor,
        domain_size_penalty_weight=args.domain_size_penalty_weight,
        domain_hard_capacity_factor=args.domain_hard_capacity_factor,
        domain_hard_smooth=args.domain_hard_smooth,
        capped_softmax_iter=args.capped_softmax_iter,
        capped_softmax_step=args.capped_softmax_step,
        capped_softmax_tol=args.capped_softmax_tol,
        final_domain_labeling=args.final_domain_labeling,
        final_domain_capacity_factor=args.final_domain_capacity_factor,
        max_iter=args.max_iter,
        tol=args.tol,
        variance_floor=args.variance_floor,
        min_peak_module_mass=args.min_peak_module_mass,
        min_domain_mass=args.min_domain_mass,
        epsilon_min=args.epsilon_min,
        epsilon_max=args.epsilon_max,
        a_temperature=args.a_temperature,
        b_temperature=args.b_temperature,
        edge_batch_size=args.edge_batch_size,
        sample_spots_for_init=args.sample_spots_for_init,
        peak_init_n_init=args.peak_init_n_init,
        factor_feature_prior_weight=args.factor_feature_prior_weight,
        factor_location_prior_weight=args.factor_location_prior_weight,
        factor_feature_prior_temperature=args.factor_feature_prior_temperature,
        factor_location_prior_temperature=args.factor_location_prior_temperature,
        prototype_transition_smoothing=args.prototype_transition_smoothing,
        prototype_transition_damping=args.prototype_transition_damping,
        update_prototype_transition=not args.freeze_prototype_transition,
        prototype_transition_identity_smoothing=args.prototype_transition_identity_smoothing,
        prototype_peak_initialization_smoothing=args.prototype_peak_initialization_smoothing,
        prototype_general_initialization_blend=args.prototype_general_initialization_blend,
        domain_initialization=args.domain_initialization,
        domain_initialization_temperature=args.domain_initialization_temperature,
        domain_initialization_n_init=args.domain_initialization_n_init,
        domain_initialization_module_weighting=(
            args.domain_initialization_module_weighting
        ),
        compute_backend=args.compute_backend,
        compute_device=args.compute_device,
        compute_dtype=args.compute_dtype,
        random_state=args.random_state,
        verbose=args.verbose,
    )
    fit_kwargs = {
        "peak_prior_labels": peak_prior_labels,
        "peak_prior_distance": peak_prior_distance,
        "initial_domain_labels": initial_domain_labels,
    }
    if prototype_gamma is not None:
        fit_kwargs["prototype_responsibilities"] = prototype_gamma
    precomputed_graph, graph_audit = load_or_build_graph_cache(
        args.graph_cache_file,
        store,
        SpatialGraphConfig(
            construction=args.graph_construction,
            n_neighbors=args.graph_neighbors,
            radius=args.graph_radius,
            distance_bandwidth=args.graph_distance_bandwidth,
            weighted=True,
            row_normalized=False,
        ),
        graph_model=args.graph_model,
    )
    validate_graph_audit(
        graph_audit,
        max_isolate_fraction=args.graph_max_isolate_fraction,
        min_largest_component_fraction=args.graph_min_largest_component_fraction,
        min_radius_coverage=args.graph_min_radius_coverage,
    )
    fit_kwargs["precomputed_graph"] = precomputed_graph
    if initial_peak_responsibilities is not None:
        fit_kwargs["initial_peak_responsibilities"] = initial_peak_responsibilities
    if factor_bridge is not None:
        fit_kwargs.update(
            {
                "initial_peak_responsibilities": factor_bridge.initial_peak_responsibilities,
                "initial_domain_responsibilities": factor_bridge.initial_domain_responsibilities,
                "factor_feature_embedding": factor_bridge.feature_embedding,
                "factor_location_embedding": factor_bridge.location_embedding,
            }
        )
    if args.adaptive_domains:
        adaptive_model = AdaptiveVSKILBM(
            model,
            min_domains=args.min_domains,
            max_domains=args.max_domains,
            max_rounds=args.split_merge_rounds,
            proposal_max_iter=args.proposal_max_iter,
            final_polish_iter=args.final_polish_iter,
            split_candidates=args.split_candidates,
            merge_candidates=args.merge_candidates,
            split_min_spots=args.split_min_spots,
            split_sample_spots=args.split_sample_spots,
            split_min_relative_sse_gain=args.split_min_relative_sse_gain,
            split_max_edge_cut_fraction=args.split_max_edge_cut_fraction,
            split_min_lcc_fraction=args.split_min_lcc_fraction,
            split_temperature=args.split_temperature,
            max_median_normalized_entropy=args.max_median_normalized_entropy,
            min_signature_symmetric_kl=args.min_signature_symmetric_kl,
            active_domain_penalty=args.active_domain_penalty,
            min_score_improvement=args.proposal_min_score_improvement,
            random_state=args.random_state,
            verbose=args.verbose,
        )
        result = adaptive_model.fit(store, **fit_kwargs)
    else:
        result = model.fit(store, **fit_kwargs)
    # Preserve the richer construction-time audit (radius, configured
    # bandwidth, and connectivity gates) instead of the generic precomputed
    # graph validation performed inside the model.
    result.graph_audit = graph_audit

    peak_table = peak_module_table(result)
    spot_table = spot_domain_table(result)
    block_table = block_parameter_table(result)
    domain_summary = domain_summary_table(result)
    signature_distance_table = domain_signature_distance_table(result)
    history = pd.DataFrame(result.history)
    peak_table.to_csv(args.output_dir / "vski_lbm_peak_modules.csv", index=False)
    spot_table.to_csv(args.output_dir / "vski_lbm_spot_domains.csv.gz", index=False)
    block_table.to_csv(args.output_dir / "vski_lbm_block_parameters.csv", index=False)
    domain_summary.to_csv(args.output_dir / "vski_lbm_domain_summary.csv", index=False)
    signature_distance_table.to_csv(args.output_dir / "vski_lbm_domain_signature_distances.csv", index=False)
    history.to_csv(args.output_dir / "vski_lbm_history.csv", index=False)
    initialization_weight_table = domain_initialization_module_weight_table(result)
    if not initialization_weight_table.empty:
        initialization_weight_table.to_csv(
            args.output_dir / "vski_lbm_domain_initialization_module_weights.csv",
            index=False,
        )
    if result.domain_model_history:
        pd.DataFrame(result.domain_model_history).to_csv(
            args.output_dir / "vski_lbm_domain_model_selection.csv", index=False
        )
    np.save(args.output_dir / "vski_lbm_module_maps.npy", result.module_maps.astype(np.float32))
    np.save(args.output_dir / "vski_lbm_peak_module_probabilities.npy", result.A.astype(np.float32))
    np.save(args.output_dir / "vski_lbm_domain_probabilities.npy", result.B.astype(np.float32))
    if result.prototype_transition is not None:
        np.save(
            args.output_dir / "vski_lbm_prototype_transition.npy",
            result.prototype_transition.astype(np.float32),
        )
        np.save(
            args.output_dir / "vski_lbm_prototype_peak_prior.npy",
            result.prototype_peak_prior.astype(np.float32),
        )
        transition_rows = [
            {
                "prototype": int(r),
                "peak_module": int(k),
                "transition_probability": float(result.prototype_transition[r, k]),
            }
            for r in range(result.prototype_transition.shape[0])
            for k in range(result.prototype_transition.shape[1])
        ]
        pd.DataFrame(transition_rows).to_csv(
            args.output_dir / "vski_lbm_prototype_transition.csv",
            index=False,
        )

    plot_rows = choose_plot_rows(store.n_spots, args.max_plot_spots, args.random_state)
    plot_domain_map(
        spot_table,
        plot_rows,
        args.output_dir / "vski_lbm_domain_map.png",
        args.point_size,
        mode=args.domain_plot_mode,
    )
    plot_module_maps(result.spots, result.module_maps, plot_rows, args.output_dir / "vski_lbm_peak_module_prototype_maps.png", args.point_size)
    plot_block_heatmap(result.mu, args.output_dir / "vski_lbm_module_domain_heatmap.png")
    if result.initial_domain_labels is not None:
        plot_crosstab(
            result.initial_domain_labels,
            result.B.argmax(axis=1),
            args.output_dir / "vski_lbm_initial_vs_final_domain_heatmap.png",
            xlabel="final V-SKI LBM domain",
            ylabel="initial domain",
            title="Initial vs final spot domains",
        )
    if result.prior_labels is not None:
        plot_crosstab(
            result.prior_labels,
            result.A.argmax(axis=1),
            args.output_dir / "vski_lbm_vski_prior_vs_final_peak_module_heatmap.png",
            xlabel="final V-SKI LBM peak module",
            ylabel="V-SKI recluster prior",
            title="V-SKI peak prior vs final peak modules",
        )

    manifest = build_manifest(args, store, peak_prior_csv, result, time.time() - started)
    with (args.output_dir / "vski_lbm_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    with (args.output_dir / "vski_lbm_assignment_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(build_assignment_diagnostics(result), handle, indent=2)

    print(f"Wrote V-SKI LBM outputs to {args.output_dir}")
    print(json.dumps({k: v for k, v in manifest.items() if k not in {"outputs", "hyperparameters"}}, indent=2))
    print(block_table.head(20).to_string(index=False))


def load_and_validate_factor_bridge(path: Path | None, store: SpatialMapStore):
    if path is None:
        return None
    bridge = load_factor_lbm_bridge(path)
    feature_column = "gene" if "gene" in store.metadata else "peak_index"
    store_features = store.metadata[feature_column].astype(str).to_numpy()
    if not np.array_equal(bridge.feature_ids.astype(str), store_features):
        raise ValueError("Factor-bridge feature IDs/order do not match the LBM map store")
    if bridge.spots.shape[0] != store.spots.shape[0]:
        raise ValueError("Factor-bridge spot count does not match the LBM map store")
    columns = [column for column in ["spot_index", "x", "y"] if column in store.spots]
    if not np.array_equal(
        bridge.spots[columns].to_numpy(), store.spots[columns].to_numpy()
    ):
        raise ValueError("Factor-bridge spot coordinates/order do not match the LBM map store")
    return bridge


def load_prototype_responsibilities(
    path: Path | None,
    store: SpatialMapStore,
    *,
    allow_nonproduction_fit: bool = False,
    allow_legacy_full_coverage: bool = False,
) -> np.ndarray | None:
    """Load and feature-align prototype responsibilities from a fitted engine."""

    if path is None:
        return None
    path = Path(path)
    manifest_file = path / "prototype_manifest.json"
    if not manifest_file.is_file():
        if not allow_nonproduction_fit:
            raise FileNotFoundError(
                f"Prototype provenance manifest is required: {manifest_file}"
            )
        prototype_manifest = {}
    else:
        prototype_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if prototype_manifest:
        input_metadata = prototype_manifest.get("input", {})
        reference = input_metadata.get("reference_vski_dir")
        if reference is None:
            raise ValueError("Prototype manifest lacks input.reference_vski_dir")
        require_production_gp_artifacts(
            [reference],
            allow_nonproduction=allow_nonproduction_fit,
            allow_legacy_full_coverage=allow_legacy_full_coverage,
        )
        current_vski_dir = _store_vski_dir(store)
        if current_vski_dir is None or Path(reference).resolve() != current_vski_dir.resolve():
            raise ValueError(
                "Prototype fit was not derived from the exact GP map directory used by LBM"
            )
        prototype_selected = input_metadata.get("selected_models")
        current_selected = _store_selected_models(store)
        if (
            (prototype_selected is None) != (current_selected is None)
            or (
                prototype_selected is not None
                and Path(prototype_selected).resolve() != current_selected.resolve()
            )
        ):
            raise ValueError("Prototype and LBM selected-model sources differ")
        if int(input_metadata.get("n_spots", -1)) != int(store.n_spots):
            raise ValueError("Prototype and LBM spot counts differ")
        prototype_spot_file = path / "prototype_spots.csv.gz"
        if not prototype_spot_file.is_file():
            raise FileNotFoundError(prototype_spot_file)
        prototype_spots = pd.read_csv(prototype_spot_file)
        spot_columns = [
            column for column in ["spot_index", "x", "y"] if column in store.spots
        ]
        if (
            prototype_spots.shape[0] != store.spots.shape[0]
            or any(column not in prototype_spots for column in spot_columns)
            or not np.array_equal(
                prototype_spots[spot_columns].to_numpy(),
                store.spots[spot_columns].to_numpy(),
            )
        ):
            raise ValueError("Prototype and LBM spot coordinates/order differ")
        expected_feature_hash = hashlib.sha256(
            "\n".join(np.asarray(store.genes).astype(str)).encode()
        ).hexdigest()
        if (
            input_metadata.get("feature_order_sha256") != expected_feature_hash
            and not allow_nonproduction_fit
        ):
            raise ValueError("Prototype and LBM feature order/signature differ")
    assignments_file = path / "prototype_assignments.csv"
    responsibilities_file = path / "prototype_responsibilities.npy"
    if not assignments_file.is_file():
        raise FileNotFoundError(assignments_file)
    if not responsibilities_file.is_file():
        raise FileNotFoundError(responsibilities_file)
    assignments = pd.read_csv(assignments_file)
    feature_column = "gene" if "gene" in assignments.columns else "peak_index"
    store_feature_column = "gene" if "gene" in store.metadata.columns else "peak_index"
    if feature_column not in assignments.columns or store_feature_column not in store.metadata.columns:
        raise ValueError("Cannot identify feature IDs for prototype/LBM alignment")
    if assignments[feature_column].astype(str).duplicated().any():
        raise ValueError("Prototype assignments contain duplicate feature IDs")
    gamma = np.load(responsibilities_file, mmap_mode="r")
    if gamma.ndim != 2 or gamma.shape[0] != assignments.shape[0]:
        raise ValueError("Prototype responsibility rows do not match prototype assignments")
    lookup = pd.Series(
        np.arange(assignments.shape[0], dtype=np.int64),
        index=assignments[feature_column].astype(str),
    )
    store_ids = store.metadata[store_feature_column].astype(str)
    missing = store_ids[~store_ids.isin(lookup.index)]
    if not missing.empty:
        raise ValueError(f"Prototype fit is missing LBM features, e.g. {missing.iloc[:5].tolist()}")
    rows = lookup.loc[store_ids].to_numpy(dtype=np.int64)
    return np.asarray(gamma[rows], dtype=np.float64)


def _store_vski_dir(store: SpatialMapStore) -> Path | None:
    current = store
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "vski_dir"):
            return Path(current.vski_dir)
        if hasattr(current, "vski"):
            current = current.vski
            continue
        if hasattr(current, "source"):
            current = current.source
            continue
        break
    return None


def _store_selected_models(store: SpatialMapStore) -> Path | None:
    current = store
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "selected_models_file"):
            value = current.selected_models_file
            return None if value is None else Path(value)
        if hasattr(current, "vski"):
            current = current.vski
            continue
        if hasattr(current, "source"):
            current = current.source
            continue
        break
    return None


def resolve_n_peak_modules(
    requested: int | None,
    prototype_responsibilities: np.ndarray | None,
) -> int:
    """Default K to the prototype count, retaining K=20 without prototypes."""

    if requested is not None:
        if requested <= 0:
            raise ValueError("n_peak_modules must be positive")
        return int(requested)
    if prototype_responsibilities is not None:
        return int(prototype_responsibilities.shape[1])
    return 20


def load_initial_peak_responsibilities(
    path: Path | None,
    store: SpatialMapStore,
    n_peak_modules: int,
) -> np.ndarray | None:
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.load(path)
    expected = (store.n_features, int(n_peak_modules))
    if values.shape != expected:
        raise ValueError(f"Initial peak responsibilities must have shape {expected}, got {values.shape}")
    return np.asarray(values, dtype=np.float64)


def load_or_build_graph_cache(
    path: Path | None,
    store: SpatialMapStore,
    config: SpatialGraphConfig,
    *,
    graph_model: str,
) -> tuple[sparse.csr_matrix, dict]:
    """Build once or reuse a content-validated graph cache."""

    coordinates = np.ascontiguousarray(
        store.spots[["x", "y"]].to_numpy(dtype=np.float32)
    )
    expected = {
        "schema_version": 2,
        "n_spots": int(store.n_spots),
        "graph_model": graph_model,
        "graph_config": config.to_dict(graph_model=graph_model),
        "coordinate_sha256": hashlib.sha256(coordinates.tobytes()).hexdigest(),
    }
    if path is None:
        return build_spatial_graph(
            coordinates,
            config=config,
            graph_model=graph_model,
        )
    path = Path(path)
    metadata_path = path.with_suffix(path.suffix + ".json")
    if path.is_file() and metadata_path.is_file():
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        observed_identity = {
            key: observed.get(key)
            for key in ("schema_version", "n_spots", "graph_model", "graph_config", "coordinate_sha256")
        }
        if observed_identity != expected:
            raise ValueError(
                f"Graph cache metadata mismatch for {path}; choose a new cache file"
            )
        if observed.get("graph_sha256") != sha256_file(path):
            raise ValueError(f"Graph cache content hash mismatch for {path}")
        graph = sparse.load_npz(path).tocsr()
        effective = config.resolved_construction(graph_model=graph_model)
        resolved_bandwidth = config.distance_bandwidth
        if effective == "radius-knn" and resolved_bandwidth is None:
            resolved_bandwidth = float(config.radius) / 2.0
        audit = audit_spatial_graph(
            graph,
            coordinates,
            requested_construction=config.construction,
            effective_construction=effective,
            distance_bandwidth=resolved_bandwidth,
        )
        audit.update(
            {
                "n_neighbors": int(config.n_neighbors),
                "maximum_allowed_edge_distance": (
                    None if config.radius is None else float(config.radius)
                ),
                "fixed_coordinate_distance_support": bool(
                    effective == "radius-knn"
                ),
            }
        )
        realized_95th = audit["realized_edge_distance_95th_percentile"]
        audit["radius_coverage_95th_fraction"] = (
            None
            if config.radius is None or realized_95th is None
            else float(realized_95th / config.radius)
        )
        return graph, audit
    if path.exists() or metadata_path.exists():
        raise ValueError(
            f"Incomplete graph cache at {path}; choose a new cache file"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    graph, audit = build_spatial_graph(
        coordinates,
        config=config,
        graph_model=graph_model,
    )
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        sparse.save_npz(temporary, graph, compressed=True)
        generated = (
            temporary
            if temporary.suffix == ".npz"
            else temporary.with_suffix(temporary.suffix + ".npz")
        )
        os.replace(generated, path)
    finally:
        for candidate in (temporary, temporary.with_suffix(temporary.suffix + ".npz")):
            if candidate.exists():
                candidate.unlink()
    atomic_write_json(
        metadata_path,
        {
            **expected,
            "graph_sha256": sha256_file(path),
            "graph_audit": audit,
        },
    )
    return graph, audit


def build_store(args: argparse.Namespace) -> SpatialMapStore:
    if args.input_mode == "vski":
        spot_indices = _factor_bridge_spot_rows(args.factor_bridge_dir, args.max_spots)
        if (
            spot_indices is None
            and args.max_spots is not None
            and args.max_spots > 0
            and args.spot_sampling == "uniform"
        ):
            spot_file = args.vski_dir / "vski_gp_map_spots.csv.gz"
            if not spot_file.exists():
                raise FileNotFoundError(spot_file)
            n_total = pd.read_csv(spot_file, usecols=["map_row"]).shape[0]
            rng = np.random.default_rng(args.random_state)
            spot_indices = np.sort(
                rng.choice(n_total, size=min(int(args.max_spots), n_total), replace=False).astype(np.int64)
            )
        store_kwargs = {
            "max_spots": None if spot_indices is not None else args.max_spots,
            "spot_indices": spot_indices,
            "max_peaks": args.max_peaks,
            "selected_models_file": args.selected_models,
            "map_standardization": args.map_standardization,
            "use_posterior_variance": not args.ignore_posterior_variance,
            "require_production_fit": True,
            "allow_nonproduction_fit": args.allow_nonproduction_fit,
            "allow_legacy_full_coverage": args.allow_legacy_full_coverage,
        }
        if args.likelihood_model == "posterior-residual":
            missing = [
                flag
                for flag, value in [
                    ("--raw-feature-dir", args.raw_feature_dir),
                    ("--coords-file", args.coords_file),
                    ("--peak-metadata", args.peak_metadata),
                ]
                if value is None
            ]
            if missing:
                raise ValueError(f"posterior-residual requires: {', '.join(missing)}")
            raw_cache_file = args.raw_cache_file or (args.output_dir / "aligned_raw_maps.npy")
            return PairedVSKIRawMapStore(
                args.vski_dir,
                raw_feature_dir=args.raw_feature_dir,
                coords_file=args.coords_file,
                peak_metadata=args.peak_metadata,
                raw_cache_file=raw_cache_file,
                target_sum=args.target_sum,
                transform=args.transform,
                feature_block_size=args.raw_feature_block_size,
                **store_kwargs,
            )
        return VSKIMapStore(
            args.vski_dir,
            **store_kwargs,
        )

    missing = [
        flag
        for flag, value in [
            ("--raw-feature-dir", args.raw_feature_dir),
            ("--coords-file", args.coords_file),
            ("--peak-metadata", args.peak_metadata),
        ]
        if value is None
    ]
    if missing:
        raise ValueError(f"Raw input mode requires: {', '.join(missing)}")
    selected_peak_indices = load_selected_peak_indices(args.selected_peak_indices_file)
    return RawTEMIMapStore(
        feature_dir=args.raw_feature_dir,
        coords_file=args.coords_file,
        peak_metadata=args.peak_metadata,
        selected_peak_indices=selected_peak_indices,
        max_spots=args.max_spots,
        max_peaks=args.max_peaks,
        map_standardization=args.map_standardization,
        target_sum=args.target_sum,
        transform=args.transform,
        feature_block_size=args.raw_feature_block_size,
    )


def _factor_bridge_spot_rows(
    factor_bridge_dir: Path | None,
    max_spots: int | None,
) -> np.ndarray | None:
    """Use a bridge's exact source rows for reproducible subset integration."""
    if factor_bridge_dir is None:
        return None
    spot_file = Path(factor_bridge_dir) / "factor_location_spots.csv.gz"
    if not spot_file.is_file():
        raise FileNotFoundError(spot_file)
    table = pd.read_csv(spot_file, usecols=["map_row"])
    rows = table["map_row"].to_numpy(dtype=np.int64)
    if rows.size == 0 or np.unique(rows).size != rows.size:
        raise ValueError("Factor bridge must contain unique map_row values")
    if max_spots is not None and int(max_spots) > 0 and rows.size != int(max_spots):
        raise ValueError(
            "--max-spots conflicts with the number of rows in --factor-bridge-dir: "
            f"{max_spots} != {rows.size}"
        )
    return rows


def load_selected_peak_indices(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
        if "peak_index" not in table.columns:
            raise ValueError(f"{path} must include a peak_index column")
        values = table["peak_index"].to_numpy(dtype=np.int64)
    else:
        tokens = path.read_text(encoding="utf-8").replace(",", " ").split()
        values = np.asarray([int(token) for token in tokens], dtype=np.int64)
    if values.size == 0:
        raise ValueError(f"{path} does not contain any peak indices")
    if np.unique(values).size != values.size:
        raise ValueError(f"{path} contains duplicate peak indices")
    return values


def resolve_peak_prior(args: argparse.Namespace) -> Path | None:
    if args.input_mode == "raw":
        if not args.no_peak_prior or args.peak_prior_csv is not None:
            raise ValueError("Raw input mode cannot use a V-SKI-derived peak prior; pass --no-peak-prior")
        return None
    if args.no_peak_prior:
        return None
    if args.peak_prior_csv is not None:
        if not args.peak_prior_csv.exists():
            raise FileNotFoundError(args.peak_prior_csv)
        return args.peak_prior_csv
    default = args.vski_dir / "vski_spatial_region_reclusters_K20" / "vski_spatial_recluster_assignments.csv"
    if default.exists():
        return default
    return None


def load_peak_prior(path: Path, store: SpatialMapStore) -> tuple[np.ndarray, np.ndarray | None]:
    table = pd.read_csv(path)
    if "gene" not in table.columns or "vski_spatial_cluster" not in table.columns:
        raise ValueError(f"{path} must include gene and vski_spatial_cluster columns")
    lookup = table.set_index("gene")
    missing = [gene for gene in store.genes if gene not in lookup.index]
    if missing:
        raise ValueError(f"Peak prior is missing selected genes, e.g. {missing[:5]}")
    aligned = lookup.loc[store.genes]
    labels = aligned["vski_spatial_cluster"].to_numpy(dtype=np.int64)
    distance = None
    if "vski_spatial_cluster_distance" in aligned.columns:
        distance = aligned["vski_spatial_cluster_distance"].to_numpy(dtype=np.float64)
    return labels, distance


def load_spot_initialization(path: Path, column: str, store: SpatialMapStore) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    if column not in table.columns:
        raise ValueError(f"{path} does not include requested spot init column {column!r}")
    if "map_row" in table.columns:
        lookup = table.set_index("map_row")[column]
        values = lookup.loc[store.spots["map_row"].to_numpy()].to_numpy()
    elif "row_index" in table.columns:
        lookup = table.set_index("row_index")[column]
        values = lookup.loc[store.spots["map_row"].to_numpy()].to_numpy()
    elif table.shape[0] >= store.spot_rows.max() + 1:
        values = table.iloc[store.spot_rows][column].to_numpy()
    else:
        raise ValueError(f"Cannot align {path} to V-SKI map rows")
    if pd.isna(values).any():
        raise ValueError("Spot initialization contains missing labels after alignment")
    return values.astype(np.int64)


def choose_plot_rows(n_spots: int, max_plot_spots: int, random_state: int) -> np.ndarray:
    if max_plot_spots <= 0 or max_plot_spots >= n_spots:
        return np.arange(n_spots, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n_spots, size=int(max_plot_spots), replace=False).astype(np.int64))


def plot_domain_map(
    spot_table: pd.DataFrame,
    rows: np.ndarray,
    output_path: Path,
    point_size: float,
    *,
    mode: str = "grid",
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2), constrained_layout=True)
    domains = np.sort(spot_table["vski_lbm_domain"].dropna().unique())
    label_to_color_index = {label: idx for idx, label in enumerate(domains)}
    cmap = plt.get_cmap("tab20", max(len(domains), 1))

    plotted_as_grid = False
    if mode == "grid":
        plotted_as_grid = _plot_domain_grid(ax, spot_table, label_to_color_index, cmap)
    if not plotted_as_grid:
        sub = spot_table.iloc[rows]
        color_indices = sub["vski_lbm_domain"].map(label_to_color_index).to_numpy()
        ax.scatter(
            sub["x"],
            sub["y"],
            c=color_indices,
            s=point_size,
            cmap=cmap,
            vmin=-0.5,
            vmax=max(len(domains) - 0.5, 0.5),
            linewidths=0,
            rasterized=True,
        )
    ax.set_title("V-SKI LBM spot domains")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=cmap(label_to_color_index[label]),
            markeredgewidth=0,
            markersize=6,
            label=f"Domain {label}",
        )
        for label in domains
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Domain",
        fontsize=8,
        title_fontsize=9,
        borderaxespad=0.0,
    )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_domain_grid(
    ax,
    spot_table: pd.DataFrame,
    label_to_color_index: dict[int, int],
    cmap,
) -> bool:
    """Render observed regular-grid spots as filled cells.

    TEMI coordinates are regular but tissue/background masking leaves missing
    lattice positions. This plot fills observed cells and leaves only truly
    absent coordinate positions blank, avoiding artificial gaps from scatter
    marker size or plotting subsampling.
    """

    x = spot_table["x"].to_numpy(dtype=np.float64)
    y = spot_table["y"].to_numpy(dtype=np.float64)
    ux = np.sort(np.unique(x))
    uy = np.sort(np.unique(y))
    if ux.size < 2 or uy.size < 2:
        return False
    dx = np.diff(ux)
    dy = np.diff(uy)
    step_x = float(np.median(dx))
    step_y = float(np.median(dy))
    if step_x <= 0 or step_y <= 0:
        return False
    if not (np.allclose(dx, step_x, rtol=1.0e-4, atol=1.0e-4) and np.allclose(dy, step_y, rtol=1.0e-4, atol=1.0e-4)):
        return False

    x_index = np.rint((x - ux[0]) / step_x).astype(np.int64)
    y_index = np.rint((y - uy[0]) / step_y).astype(np.int64)
    if (
        x_index.min(initial=0) < 0
        or y_index.min(initial=0) < 0
        or x_index.max(initial=-1) >= ux.size
        or y_index.max(initial=-1) >= uy.size
    ):
        return False
    if not (
        np.allclose(ux[x_index], x, rtol=1.0e-4, atol=1.0e-4)
        and np.allclose(uy[y_index], y, rtol=1.0e-4, atol=1.0e-4)
    ):
        return False

    grid = np.full((uy.size, ux.size), np.nan, dtype=np.float32)
    values = spot_table["vski_lbm_domain"].map(label_to_color_index).to_numpy(dtype=np.float32)
    grid[y_index, x_index] = values

    colors = cmap(np.arange(max(len(label_to_color_index), 1)))
    grid_cmap = ListedColormap(colors)
    grid_cmap.set_bad(color="white", alpha=0.0)
    masked_grid = np.ma.masked_invalid(grid)
    ax.imshow(
        masked_grid,
        origin="lower",
        interpolation="nearest",
        cmap=grid_cmap,
        vmin=-0.5,
        vmax=max(len(label_to_color_index) - 0.5, 0.5),
        extent=(ux[0] - step_x / 2, ux[-1] + step_x / 2, uy[0] - step_y / 2, uy[-1] + step_y / 2),
        rasterized=True,
    )
    return True


def plot_module_maps(spots: pd.DataFrame, module_maps: np.ndarray, rows: np.ndarray, output_path: Path, point_size: float) -> None:
    n_modules = module_maps.shape[1]
    n_cols = min(5, n_modules)
    n_rows = int(np.ceil(n_modules / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.8 * n_rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    values = module_maps[rows]
    vmax = float(np.percentile(np.abs(values[np.isfinite(values)]), 98)) if np.isfinite(values).any() else 1.0
    vmax = max(vmax, 1.0e-6)
    for k, ax in enumerate(axes[:n_modules]):
        ax.scatter(
            spots.iloc[rows]["x"],
            spots.iloc[rows]["y"],
            c=values[:, k],
            s=point_size,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(f"Peak module {k}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[n_modules:]:
        ax.axis("off")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_block_heatmap(mu: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    vmax = float(np.percentile(np.abs(mu[np.isfinite(mu)]), 98)) if np.isfinite(mu).any() else 1.0
    vmax = max(vmax, 1.0e-6)
    im = ax.imshow(mu, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Spot domain")
    ax.set_ylabel("Peak module")
    ax.set_title("V-SKI LBM block means")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_crosstab(row_labels: np.ndarray, col_labels: np.ndarray, output_path: Path, *, xlabel: str, ylabel: str, title: str) -> None:
    table = pd.crosstab(pd.Series(row_labels, name="row"), pd.Series(col_labels, name="col"))
    fig, ax = plt.subplots(figsize=(7.4, 5.8), constrained_layout=True)
    im = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_xticklabels(table.columns.astype(str), rotation=90, fontsize=7)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table.index.astype(str), fontsize=7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01, label="count")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_manifest(
    args: argparse.Namespace,
    store: SpatialMapStore,
    peak_prior_csv: Path | None,
    result,
    elapsed_seconds: float,
) -> dict:
    metrics = assignment_metrics(result)
    outputs = {
        "peak_modules": str(args.output_dir / "vski_lbm_peak_modules.csv"),
        "spot_domains": str(args.output_dir / "vski_lbm_spot_domains.csv.gz"),
        "block_parameters": str(args.output_dir / "vski_lbm_block_parameters.csv"),
        "domain_summary": str(args.output_dir / "vski_lbm_domain_summary.csv"),
        "domain_signature_distances": str(args.output_dir / "vski_lbm_domain_signature_distances.csv"),
        "history": str(args.output_dir / "vski_lbm_history.csv"),
        "domain_initialization_module_weights": (
            str(args.output_dir / "vski_lbm_domain_initialization_module_weights.csv")
            if not domain_initialization_module_weight_table(result).empty
            else None
        ),
        "domain_model_selection": (
            str(args.output_dir / "vski_lbm_domain_model_selection.csv")
            if result.domain_model_history
            else None
        ),
        "assignment_diagnostics": str(args.output_dir / "vski_lbm_assignment_diagnostics.json"),
        "module_maps": str(args.output_dir / "vski_lbm_module_maps.npy"),
        "peak_module_probabilities": str(
            args.output_dir / "vski_lbm_peak_module_probabilities.npy"
        ),
        "domain_probabilities": str(args.output_dir / "vski_lbm_domain_probabilities.npy"),
        "domain_map": str(args.output_dir / "vski_lbm_domain_map.png"),
        "module_prototype_maps": str(args.output_dir / "vski_lbm_peak_module_prototype_maps.png"),
        "module_domain_heatmap": str(args.output_dir / "vski_lbm_module_domain_heatmap.png"),
        "prototype_transition": (
            str(args.output_dir / "vski_lbm_prototype_transition.csv")
            if result.prototype_transition is not None
            else None
        ),
        "prototype_peak_prior": (
            str(args.output_dir / "vski_lbm_prototype_peak_prior.npy")
            if result.prototype_transition is not None
            else None
        ),
    }
    return {
        "method": "V-SKI-output latent block model" if args.input_mode == "vski" else "Raw-TEMI latent block model",
        "input_mode": args.input_mode,
        "vski_dir": str(args.vski_dir) if args.input_mode == "vski" else None,
        "selected_models": str(args.selected_models) if args.selected_models else None,
        "raw_input": {
            "feature_dir": str(args.raw_feature_dir) if args.raw_feature_dir is not None else None,
            "coords_file": str(args.coords_file) if args.coords_file is not None else None,
            "peak_metadata": str(args.peak_metadata) if args.peak_metadata is not None else None,
            "selected_peak_indices_file": str(args.selected_peak_indices_file) if args.selected_peak_indices_file is not None else None,
            "target_sum": float(args.target_sum),
            "transform": args.transform,
            "feature_block_size": int(args.raw_feature_block_size),
            "cache_file": str(
                args.raw_cache_file or (args.output_dir / "aligned_raw_maps.npy")
            )
            if args.likelihood_model == "posterior-residual"
            else None,
        }
        if args.input_mode == "raw" or args.likelihood_model == "posterior-residual"
        else None,
        "uses_posterior_variance": bool(store.uses_posterior_variance),
        "gp_artifact_audits": [
            audit.to_dict() for audit in getattr(store, "gp_artifact_audits", [])
        ],
        "peak_prior_csv": str(peak_prior_csv) if peak_prior_csv is not None else None,
        "spot_init_csv": str(args.spot_init_csv) if args.spot_init_csv else None,
        "spot_init_column": args.spot_init_column if args.spot_init_csv else None,
        "graph_cache_file": str(args.graph_cache_file) if args.graph_cache_file else None,
        "graph_audit": result.graph_audit,
        "map_cache_dir": str(args.map_cache_dir) if args.map_cache_dir else None,
        "factor_bridge_dir": str(args.factor_bridge_dir) if args.factor_bridge_dir else None,
        "prototype_dir": str(args.prototype_dir) if args.prototype_dir else None,
        "initial_peak_responsibilities_file": (
            str(args.initial_peak_responsibilities_file)
            if args.initial_peak_responsibilities_file
            else None
        ),
        "initialization": describe_initialization(args, result.initialization_metadata),
        "n_spots": int(store.n_spots),
        "n_peaks": int(store.n_features),
        "n_iterations": int(len(result.history)),
        "elapsed_seconds": float(elapsed_seconds),
        "hyperparameters": {
            "n_peak_modules": int(args.n_peak_modules),
            "n_domains": int(args.n_domains),
            "graph_neighbors": int(args.graph_neighbors),
            "graph_construction": args.graph_construction,
            "effective_graph_construction": result.graph_audit.get(
                "effective_construction"
            ),
            "graph_radius": (
                None if args.graph_radius is None else float(args.graph_radius)
            ),
            "graph_distance_bandwidth": (
                None
                if args.graph_distance_bandwidth is None
                else float(args.graph_distance_bandwidth)
            ),
            "graph_max_isolate_fraction": float(
                args.graph_max_isolate_fraction
            ),
            "graph_min_largest_component_fraction": float(
                args.graph_min_largest_component_fraction
            ),
            "graph_min_radius_coverage": float(
                args.graph_min_radius_coverage
            ),
            "likelihood_scaling": args.likelihood_scaling,
            "likelihood_power_a": float(args.likelihood_power_a),
            "likelihood_power_coupling": args.likelihood_power_coupling,
            "shared_likelihood_power": args.shared_likelihood_power,
            "feature_likelihood_weighting": args.feature_likelihood_weighting,
            "graph_module_weighting": args.graph_module_weighting,
            "likelihood_power_b": float(args.likelihood_power_b),
            "likelihood_model": args.likelihood_model,
            "effective_likelihood_power_a": float(result.effective_likelihood_power_a),
            "effective_likelihood_power_b": float(result.effective_likelihood_power_b),
            "lambda_z": float(args.lambda_z),
            "lambda_z_min": float(args.lambda_z_min),
            "peak_prior_decay_tau": float(args.peak_prior_decay_tau),
            "graph_model": args.graph_model,
            "beta": float(args.beta),
            "roughness_weight": float(args.roughness_weight),
            "graph_boundary_scale": float(args.graph_boundary_scale),
            "domain_prior_weight": float(args.domain_prior_weight),
            "domain_mixture_prior": args.domain_mixture_prior,
            "domain_dirichlet_alpha0": float(args.domain_dirichlet_alpha0),
            "domain_mixture_prior_weight": float(args.domain_mixture_prior_weight),
            "active_domain_weight_threshold": float(args.active_domain_weight_threshold),
            "active_domain_min_spots": float(args.active_domain_min_spots),
            "domain_update": args.domain_update,
            "domain_capacity_factor": float(args.domain_capacity_factor),
            "domain_size_penalty_weight": float(args.domain_size_penalty_weight),
            "domain_hard_capacity_factor": float(args.domain_hard_capacity_factor),
            "domain_hard_smooth": float(args.domain_hard_smooth),
            "capped_softmax_iter": int(args.capped_softmax_iter),
            "capped_softmax_step": float(args.capped_softmax_step),
            "capped_softmax_tol": float(args.capped_softmax_tol),
            "final_domain_labeling": args.final_domain_labeling,
            "final_domain_capacity_factor": float(args.final_domain_capacity_factor),
            "max_iter": int(args.max_iter),
            "sample_spots_for_init": int(args.sample_spots_for_init),
            "peak_init_n_init": int(args.peak_init_n_init),
            "tol": float(args.tol),
            "variance_floor": float(args.variance_floor),
            "min_peak_module_mass": float(args.min_peak_module_mass),
            "min_domain_mass": float(args.min_domain_mass),
            "assignment_repair": "none",
            "epsilon_min": float(args.epsilon_min),
            "epsilon_max": float(args.epsilon_max),
            "a_temperature": float(args.a_temperature),
            "b_temperature": float(args.b_temperature),
            "factor_feature_prior_weight": float(args.factor_feature_prior_weight),
            "factor_location_prior_weight": float(args.factor_location_prior_weight),
            "factor_feature_prior_temperature": float(
                args.factor_feature_prior_temperature
            ),
            "factor_location_prior_temperature": float(
                args.factor_location_prior_temperature
            ),
            "prototype_transition_smoothing": float(args.prototype_transition_smoothing),
            "prototype_transition_damping": float(args.prototype_transition_damping),
            "prototype_transition_identity_smoothing": float(
                args.prototype_transition_identity_smoothing
            ),
            "prototype_peak_initialization_smoothing": float(
                args.prototype_peak_initialization_smoothing
            ),
            "prototype_general_initialization_blend": float(
                args.prototype_general_initialization_blend
            ),
            "update_prototype_transition": bool(not args.freeze_prototype_transition),
            "domain_initialization": args.domain_initialization,
            "domain_initialization_temperature": args.domain_initialization_temperature,
            "domain_initialization_n_init": int(args.domain_initialization_n_init),
            "domain_initialization_module_weighting": (
                args.domain_initialization_module_weighting
            ),
            "map_standardization": args.map_standardization,
            "compute_backend": args.compute_backend,
            "compute_device": args.compute_device,
            "compute_dtype": args.compute_dtype,
            "compute_audit": result.compute_audit,
            "residual_weight": float(args.residual_weight),
            "residual_variance": args.residual_variance,
            "ignore_posterior_variance": bool(args.ignore_posterior_variance),
            "adaptive_domains": bool(args.adaptive_domains),
            "min_domains": int(args.min_domains),
            "max_domains": int(args.max_domains),
            "split_merge_rounds": int(args.split_merge_rounds),
            "proposal_max_iter": int(args.proposal_max_iter),
            "final_polish_iter": int(args.final_polish_iter),
            "split_candidates": int(args.split_candidates),
            "merge_candidates": int(args.merge_candidates),
            "split_min_spots": int(args.split_min_spots),
            "split_sample_spots": int(args.split_sample_spots),
            "split_min_relative_sse_gain": float(args.split_min_relative_sse_gain),
            "split_max_edge_cut_fraction": float(args.split_max_edge_cut_fraction),
            "split_min_lcc_fraction": float(args.split_min_lcc_fraction),
            "split_temperature": float(args.split_temperature),
            "max_median_normalized_entropy": float(args.max_median_normalized_entropy),
            "min_signature_symmetric_kl": float(args.min_signature_symmetric_kl),
            "active_domain_penalty": (
                None if args.active_domain_penalty is None else float(args.active_domain_penalty)
            ),
            "proposal_min_score_improvement": float(args.proposal_min_score_improvement),
            "final_n_domain_components": int(result.B.shape[1]),
            "final_n_active_domains": int(result.active_domain_mask.sum()),
            "spot_sampling": args.spot_sampling,
        },
        "final_objective_components": result.final_objective_components,
        "metrics": metrics,
        "outputs": outputs,
        "note": (
            "Uses a cut-posterior expected likelihood for V-SKI latent maps plus an aligned tempered raw residual view."
            if args.likelihood_model == "posterior-residual"
            else "Uses V-SKI posterior means and posterior variances in the expected latent-signal block likelihood."
            if args.input_mode == "vski" and args.likelihood_model == "posterior-expected"
            else "Uses V-SKI posterior means and block-average posterior variance; raw residuals are not loaded."
            if args.input_mode == "vski" and store.uses_posterior_variance
            else "Uses V-SKI posterior means only; posterior variance and raw residuals are intentionally not loaded."
            if args.input_mode == "vski"
            else "Uses normalized/transformed raw TEMI peak intensities with zero posterior uncertainty; no V-SKI prior or raw residual view is loaded."
        ),
    }


def describe_initialization(
    args: argparse.Namespace,
    resolved: dict | None = None,
) -> dict:
    """Make direct and inherited initialization dependencies explicit."""

    if args.initial_peak_responsibilities_file is not None:
        peak_mode = "external_responsibility_matrix"
    elif args.factor_bridge_dir is not None:
        peak_mode = "factor_bridge_responsibilities"
    elif args.prototype_dir is not None:
        peak_mode = "prototype_asymmetric_auto"
    elif args.peak_prior_csv is not None:
        peak_mode = "supplied_peak_labels"
    else:
        peak_mode = "sampled_map_kmeans"
    if args.spot_init_csv is not None:
        domain_mode = "supplied_labels_smoothed_one_hot"
        smooth = 0.05
        assigned_probability = 1.0 - smooth + smooth / args.n_domains
        other_probability = smooth / args.n_domains
    else:
        domain_mode = args.domain_initialization
        assigned_probability = None
        other_probability = None

    inherited = None
    if args.initial_peak_responsibilities_file is not None:
        source_manifest = (
            Path(args.initial_peak_responsibilities_file).parent
            / "vski_lbm_manifest.json"
        )
        if source_manifest.is_file():
            payload = json.loads(source_manifest.read_text(encoding="utf-8"))
            inherited = {
                "source_manifest": str(source_manifest),
                "source_spot_init_csv": payload.get("spot_init_csv"),
                "source_spot_init_column": payload.get("spot_init_column"),
                "source_graph_model": payload.get("hyperparameters", {}).get(
                    "graph_model"
                ),
                "source_graph_beta": payload.get("hyperparameters", {}).get("beta"),
                "source_iterations": payload.get("n_iterations"),
            }
    return {
        "peak_mode": peak_mode,
        "domain_mode": domain_mode,
        "supplied_domain_assigned_probability": assigned_probability,
        "supplied_domain_other_probability": other_probability,
        "prototype_prior_role": (
            "initializes A directly when Q=K; otherwise uses an estimated transition; "
            "remains an annealed peak prior during fitting"
            if args.prototype_dir is not None
            else None
        ),
        "inherited_peak_initialization_provenance": inherited,
        "resolved": resolved,
    }


def build_assignment_diagnostics(result) -> dict:
    """Summarize whether soft assignments identify interpretable domains."""

    metrics = assignment_metrics(result)
    n_domains = int(result.B.shape[1])
    certainty_ratio = float(metrics["median_domain_max_probability"] * n_domains)
    warnings = []
    if int(metrics["n_active_peak_modules_hard"]) < int(result.A.shape[1]):
        warnings.append("One or more requested peak modules have no hard assignments.")
    if int(metrics["n_active_domains_hard"]) < n_domains:
        warnings.append("One or more requested domains have no hard assignments.")
    if certainty_ratio <= 1.25:
        warnings.append("Median domain certainty is close to the uniform 1/R baseline.")
    if float(metrics["median_domain_normalized_entropy"]) >= 0.95:
        warnings.append("Median normalized domain entropy is near its maximum.")
    min_signature_kl = metrics["min_domain_signature_symmetric_kl"]
    if min_signature_kl is not None and float(min_signature_kl) < 1.0e-3:
        warnings.append("At least one pair of Gaussian domain signatures is effectively duplicated.")
    return {
        "metrics": metrics,
        "uniform_domain_probability": float(1.0 / n_domains),
        "median_certainty_relative_to_uniform": certainty_ratio,
        "hard_map_interpretation_recommended": len(warnings) == 0,
        "warnings": warnings,
        "threshold_note": (
            "Warning thresholds are diagnostics, not significance tests: certainty ratio <=1.25, "
            "normalized entropy >=0.95, or minimum symmetric KL <1e-3."
        ),
    }


if __name__ == "__main__":
    main()
