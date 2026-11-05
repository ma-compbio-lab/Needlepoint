#!/usr/bin/env python3
"""Cluster fitted feature-GP maps into uncertainty-aware spatial prototypes."""

from __future__ import annotations

import argparse
from pathlib import Path

from needlepoint.shared_spatial_factorization import GPPosteriorFactorStore
from needlepoint.spatial_prototypes import (
    SpatialPrototypeSettings,
    fit_spatial_prototypes,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vski-dir", type=Path, required=True)
    parser.add_argument("--selected-models", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--materialize-input-cache", action="store_true",
                        help="Opt in to full FP32 input caches instead of blockwise reads.")
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
    parser.add_argument("--max-spots", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument(
        "--nonnegative-projection",
        choices=["error", "rectified_gaussian"],
        default="rectified_gaussian",
        help=(
            "Use error for GP outputs already expressed as nonnegative rate "
            "moments; rectified_gaussian preserves the historical intensity path."
        ),
    )
    parser.add_argument("--method", choices=["functional", "gaussian"], default="functional")
    parser.add_argument("--n-prototypes", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--tau2", type=float, default=0.1)
    parser.add_argument("--update-tau2", action="store_true")
    parser.add_argument("--tau2-min", type=float, default=1.0e-4)
    parser.add_argument("--tau2-max", type=float, default=10.0)
    parser.add_argument(
        "--gaussian-evidence-scale",
        type=float,
        default=None,
        help="Power applied to summed Gaussian log likelihood; default 1/n_spots.",
    )
    parser.add_argument(
        "--gaussian-evidence-power",
        type=float,
        default=1.0,
        help="Multiplier of the average Gaussian log likelihood when no explicit scale is supplied.",
    )
    parser.add_argument(
        "--mixture-weight-mode",
        choices=["learned", "uniform"],
        default="learned",
    )
    parser.add_argument("--mixture-smoothing", type=float, default=1.0)
    parser.add_argument("--responsibility-damping", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    parser.add_argument("--spot-chunk-size", type=int, default=4096)
    parser.add_argument("--no-preload-to-device", action="store_true")
    parser.add_argument("--initialization-spots", type=int, default=5000)
    parser.add_argument("--initialization-smoothing", type=float, default=0.02)
    parser.add_argument("--initialization-method", choices=["spatial-sample", "multiscale-sketch"],
                        default="spatial-sample")
    parser.add_argument("--initialization-embedding-dir", type=str, default=None)
    parser.add_argument(
        "--merge-similar-prototypes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Conservatively collapse redundant fitted prototypes after EM.",
    )
    parser.add_argument("--merge-map-correlation-threshold", type=float, default=0.9)
    parser.add_argument(
        "--merge-membership-correlation-threshold",
        type=float,
        default=-1.0,
        help="Optional secondary gate; -1 disables it (recommended).",
    )
    parser.add_argument("--min-prototypes-after-merge", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    store = GPPosteriorFactorStore(
        args.vski_dir,
        args.cache_dir or (args.output_dir / "prototype_input_cache"),
        selected_models=args.selected_models,
        normalization_mode="shape",
        nonnegative_projection=args.nonnegative_projection,
        max_spots=args.max_spots,
        max_features=args.max_features,
        require_production_fit=True,
        allow_nonproduction_fit=args.allow_nonproduction_fit,
        allow_legacy_full_coverage=args.allow_legacy_full_coverage,
        materialize_cache=args.materialize_input_cache,
    )
    settings = SpatialPrototypeSettings(
        method=args.method,
        n_prototypes=args.n_prototypes,
        max_iter=args.max_iter,
        n_init=args.n_init,
        beta=args.beta,
        tau2=args.tau2,
        update_tau2=args.update_tau2,
        tau2_min=args.tau2_min,
        tau2_max=args.tau2_max,
        gaussian_evidence_scale=args.gaussian_evidence_scale,
        gaussian_evidence_power=args.gaussian_evidence_power,
        mixture_weight_mode=args.mixture_weight_mode,
        mixture_smoothing=args.mixture_smoothing,
        responsibility_damping=args.responsibility_damping,
        tolerance=args.tolerance,
        spot_chunk_size=args.spot_chunk_size,
        preload_to_device=not args.no_preload_to_device,
        initialization_spots=args.initialization_spots,
        initialization_smoothing=args.initialization_smoothing,
        initialization_method=args.initialization_method,
        initialization_embedding_dir=args.initialization_embedding_dir,
        merge_similar_prototypes=args.merge_similar_prototypes,
        merge_map_correlation_threshold=args.merge_map_correlation_threshold,
        merge_membership_correlation_threshold=args.merge_membership_correlation_threshold,
        min_prototypes_after_merge=args.min_prototypes_after_merge,
        device=args.device,
        dtype=args.dtype,
        random_state=args.random_state,
    )
    result = fit_spatial_prototypes(store, args.output_dir, settings)
    print(
        f"Wrote {args.method} prototype fit with {args.n_prototypes} requested and "
        f"{result.responsibilities.shape[1]} effective families "
        f"to {result.output_dir} (objective={result.objective:.6g})"
    )


if __name__ == "__main__":
    main()
