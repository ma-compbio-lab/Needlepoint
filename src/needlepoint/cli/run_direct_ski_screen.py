"""Explicit, resumable full-support deterministic-SKI H5AD screen."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

import numpy as np

from needlepoint.cli.run_h5ad_joint_global_local import _load_preprocessed_panel
from needlepoint.direct_ski_screen import atomic_json, file_sha256, screen_direct_ski
from needlepoint.gp_initialization import InitialLengthScaleScreenSettings, mark_uncertain_initial_scales


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", required=True, type=Path)
    parser.add_argument("--selected-genes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--length-scale-policy",
                        choices=("physical_multiscale_v1", "grid_aware_log2"),
                        default="physical_multiscale_v1")
    parser.add_argument("--sampling-interval-um", type=float, default=None)
    parser.add_argument("--feature-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-workspace-gib", type=float, default=24.)
    parser.add_argument("--residual-tolerance", type=float, default=1e-6)
    parser.add_argument("--residual-policy", choices=("warn", "error"), default="warn",
                        help="Finite residual tolerance exceedances warn by default; invalid values still fail")
    parser.add_argument("--minimum-loglik-gap", type=float, default=2.)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output_dir
    if root.is_symlink() or (root.exists() and any(root.iterdir()) and not args.resume):
        raise FileExistsError("Nonempty/symlink screen output; use --resume for identical checkpointed work")
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # A lock prevents two workers from sharing mutable checkpoint progress.
    import fcntl
    with (root/".screen.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        import needlepoint.direct_ski_screen as implementation
        import needlepoint.experimental_direct_ski as solver
        import needlepoint.gp_initialization as initialization
        import needlepoint.cli.run_h5ad_joint_global_local as adapter
        provenance = dict(h5ad=str(args.h5ad.resolve()), h5ad_sha256=file_sha256(args.h5ad),
            dependency_versions={name: version(name) for name in
                                 ("torch", "gpytorch", "linear-operator", "numpy", "scipy", "anndata")},
            selected_genes=str(args.selected_genes.resolve()), panel_sha256=file_sha256(args.selected_genes),
            source_sha256={module.__name__: file_sha256(module.__file__)
                           for module in (implementation, solver, initialization, adapter)},
            normalization_scope="full-library", target_sum=10000., transform="log1p",
            grid_size=args.grid_size, feature_chunk_size=args.feature_chunk_size,
            length_scale_policy=args.length_scale_policy,
            sampling_interval_um=args.sampling_interval_um,
            minimum_loglik_gap=args.minimum_loglik_gap, residual_tolerance=args.residual_tolerance,
            residual_policy=args.residual_policy)
        run_path = root/"request.json"
        if run_path.exists() and json.loads(run_path.read_text()) != provenance:
            raise ValueError("Resume request differs from frozen input/config/source")
        atomic_json(run_path, provenance)
        complete = root/"initial_length_scale_screen_summary.json"
        if complete.exists():
            saved = json.loads(complete.read_text())
            if (saved["input"] != provenance or
                    saved["scores_sha256"] != file_sha256(root/"initial_length_scale_scores.csv")):
                raise ValueError("Completed screen provenance/output mismatch")
            print("Verified completed direct-SKI screen", flush=True)
            return
        try:
            expression, coords, genes, panel = _load_preprocessed_panel(args.h5ad, args.selected_genes,
                coord_key="spatial", layer=None, gene_column="gene", gene_start=0, gene_stop=None,
                target_sum=10000., transform="log1p")
            settings = InitialLengthScaleScreenSettings(grid_size=args.grid_size,
                feature_chunk_size=args.feature_chunk_size, dtype="float32", device=args.device,
                length_scale_policy=args.length_scale_policy,
                sampling_interval_um=args.sampling_interval_um)
            scores, detail = screen_direct_ski(expression, coords, genes, settings,
                checkpoint_dir=root/"checkpoints", residual_tolerance=args.residual_tolerance,
                residual_policy=args.residual_policy,
                max_workspace_gib=args.max_workspace_gib, provenance=provenance)
            scores = mark_uncertain_initial_scales(scores, minimum_loglik_gap=args.minimum_loglik_gap)
            extra = [key for key in panel if key == "gene" or key not in scores]
            scores = scores.merge(panel[extra], on="gene", how="left", validate="one_to_one")
            for name, frame in (("initial_length_scale_scores.csv", scores),
                                ("selective_multistart_genes.csv", scores[scores.recommended_for_multistart])):
                staging = root/f".{name}.partial-{os.getpid()}"
                frame.to_csv(staging, index=False)
                os.replace(staging, root/name)
            atomic_json(complete, dict(status="complete", mode="screen", fit_purpose="evaluation",
                downstream_ready=False, input=provenance, screen=detail,
                shape=dict(n_spots=len(coords), n_genes=len(genes)),
                partition=dict(n_training_spots=len(coords), n_prediction_spots=len(coords),
                               training_coverage_fraction=1., n_holdout_spots=0),
                scores_sha256=file_sha256(root/"initial_length_scale_scores.csv"),
                elapsed_seconds=time.time()-started))
        except Exception as error:
            atomic_json(root/"failure.json", dict(error_type=type(error).__name__, error=str(error),
                failed_epoch=time.time(), completion_marker_written=False))
            raise


if __name__ == "__main__":
    main()
