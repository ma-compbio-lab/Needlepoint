"""Explicit post-fit Gaussian conditioning; ``--help`` keeps GP imports lazy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _integers(text):
    try:
        result = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("use comma-separated nonnegative integers") from error
    if not result or min(result) < 0 or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("indices must be nonnegative and unique")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=(
        "Condition a verified fitted Gaussian SKI/VNNGP prior jointly, with frozen "
        "hyperparameters. Writes a separate experimental artifact; fixed-draw "
        "Monte Carlo variance maps retain sampling uncertainty."))
    parser.add_argument("--source-dir", "--source", dest="source_dir", type=Path, required=True,
                        help="Completed schema-2 source map directory with hash-bearing receipt.")
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, required=True,
                        help="New or empty directory outside the frozen source directory.")
    parser.add_argument("--raw-input-manifest", type=Path, required=True,
                        help="Verified raw-input mapping required to compute the joint posterior mean.")
    parser.add_argument("--features", type=_integers,
                        help="Comma-separated source metadata feature positions; default: all source features.")
    parser.add_argument("--mean-mode", choices=("joint", "preserve_variational"), default="joint",
                        help="preserve_variational is an explicitly experimental hybrid mean/covariance ablation.")
    parser.add_argument("--draws", type=int, default=256,
                        help="Prespecified perturb-and-solve draw count; does not adapt to measured map error.")
    parser.add_argument("--preconditioner", choices=("identity", "scaled_block_jacobi"), default="identity")
    parser.add_argument("--rtol", type=float, default=1e-9)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--residual-refresh", type=int, default=25)
    parser.add_argument("--rhs-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=2048)
    parser.add_argument("--audit-rows", type=_integers,
                        help="Prespecified comma-separated source map rows for deterministic variance bounds.")
    parser.add_argument("--audit-count", type=int, default=32,
                        help="Default deterministic geometry/training/withheld audit selection size.")
    parser.add_argument("--audit-relative-tolerance", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--memory-cap-gb", type=float, default=8.0,
                        help="Declared in-memory operator/workspace cap in GiB.")
    parser.add_argument("--threads", type=int, default=2,
                        help="Torch CPU threads for this CLI process, bounded by CPU affinity (default: 2).")
    parser.add_argument("--device", default="cpu",
                        help="Explicit Torch device (default: cpu); CUDA is never selected automatically.")
    return parser


def select_audit_rows(snapshot, count, seed):
    """Deterministic, response-independent rows spanning geometry and support."""
    import numpy as np

    p = len(snapshot.query_coordinates)
    if count <= 0:
        raise ValueError("audit count must be positive")
    count = min(int(count), p)
    coordinates = snapshot.query_coordinates
    training = np.asarray(snapshot.training_query_rows)
    withheld = np.setdiff1d(np.arange(p), training)
    boundary = np.flatnonzero(np.any((coordinates == coordinates.min(axis=0)) |
                                     (coordinates == coordinates.max(axis=0)), axis=1))
    interior = np.setdiff1d(np.arange(p), boundary)
    rng = np.random.default_rng(seed)
    strata = [training, withheld, boundary, interior]
    # Each stratum uses a deterministic random permutation; interleaving avoids
    # filling the full budget from a large first stratum.
    permutations = [rng.permutation(rows).tolist() for rows in strata if len(rows)]
    selected = list(dict.fromkeys([0, p-1]))[:count]
    while len(selected) < count and any(permutations):
        for rows in permutations:
            while rows and rows[-1] in selected:
                rows.pop()
            if rows and len(selected) < count:
                selected.append(int(rows.pop()))
    return tuple(sorted(selected))


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if not source.is_dir():
        parser.error("source directory does not exist")
    if output == source or output.is_relative_to(source):
        parser.error("output must be separate from the frozen source directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output must be a new or empty directory")
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    if not 1 <= args.threads <= affinity:
        parser.error(f"threads must be between 1 and this process's CPU affinity ({affinity})")
    if args.seed < 0 or args.audit_count <= 0:
        parser.error("seed must be nonnegative and audit count must be positive")
    if not args.raw_input_manifest.is_file():
        parser.error("raw input manifest does not exist")

    # All model imports happen after argparse/preflight, keeping --help usable
    # without Torch and preventing a help call from changing process resources.
    import torch
    from needlepoint.gp_frozen_state import load_frozen_gaussian_fit
    from needlepoint.gp_joint_conditioning import (
        FrozenConditioningSettings, condition_frozen_gaussian_fit)

    torch.set_num_threads(args.threads)
    try:
        snapshots = load_frozen_gaussian_fit(source, args.raw_input_manifest, args.features)
        prepared = []
        for snapshot in snapshots:
            rows = args.audit_rows if args.audit_rows is not None else select_audit_rows(
                snapshot, args.audit_count, args.seed)
            if max(rows) >= len(snapshot.query_coordinates):
                raise ValueError("audit row outside source prediction support")
            settings = FrozenConditioningSettings(
                work_dtype="float64", output_dtype="float32", device=args.device,
                mean_mode=args.mean_mode, fixed_draws=args.draws, random_seed=args.seed,
                preconditioner=args.preconditioner, rtol=args.rtol, atol=args.atol,
                max_iterations=args.max_iterations, residual_refresh=args.residual_refresh,
                rhs_batch_size=args.rhs_batch_size, query_batch_size=args.query_batch_size,
                audit_rows=rows, audit_relative_tolerance=args.audit_relative_tolerance,
                memory_cap_bytes=int(args.memory_cap_gb * 1024**3), failure_policy="raise")
            destination = output if len(snapshots) == 1 else output / f"feature_{snapshot.feature_position:04d}"
            prepared.append((snapshot, destination, settings))
        results = []
        for snapshot, destination, settings in prepared:
            result = condition_frozen_gaussian_fit(snapshot, destination, settings)
            results.append(dict(feature_position=snapshot.feature_position, feature_name=snapshot.feature_name,
                                output_dir=result["output_dir"], target_identity=snapshot.target_identity,
                                mean_mode=settings.mean_mode, fixed_draws=settings.fixed_draws,
                                solver_status=result["mean_receipt"]["status"],
                                audit_rows=list(settings.audit_rows)))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f"frozen Gaussian conditioning failed: {error}\n")
    print(json.dumps(dict(inference_backend="frozen_kernel_joint_gaussian_precision",
                          source_dir=str(source), results=results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
