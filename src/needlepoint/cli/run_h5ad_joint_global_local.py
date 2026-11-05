#!/usr/bin/env python3
"""Screen and fit joint global-local GP maps from a sparse H5AD.

This adapter keeps the modality-neutral GP implementation in ``needlepoint`` and
adds the H5AD contracts needed by sequencing assays: full-library row totals,
stable panel ordering, physical coordinates, common holdouts, feature shards,
and bounded-memory fidelity evaluation.
"""

from __future__ import annotations

from needlepoint.storage import add_storage_arguments, storage_kwargs

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from needlepoint.data import normalize_transform_csr_with_row_sums
from needlepoint.gp_fidelity import (
    SpatialHoldouts,
    evaluate_gp_fidelity,
    make_spatial_holdouts,
    summarize_fidelity,
)
from needlepoint.gp_initialization import (
    InitialLengthScaleScreenSettings,
    interior_lengthscale_initialization,
    mark_uncertain_initial_scales,
    resolve_initial_length_scales,
    screen_initial_length_scales,
)
from needlepoint.gpytorch_global_local import GlobalLocalGPSettings
from needlepoint.gpytorch_joint_global_local import (
    JointGlobalLocalGPSettings,
    fit_joint_global_local_gp_spatial_maps,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-gene GP scale screening and joint global-local H5AD fitting."
    )
    parser.add_argument("--mode", choices=["screen", "fit"], required=True)
    parser.add_argument(
        "--fit-purpose",
        choices=["evaluation", "production"],
        default="evaluation",
        help=(
            "evaluation fits retain holdouts for fidelity assessment and are not "
            "valid prototype/LBM inputs; production fits must use zero holdout "
            "fractions and train on every likelihood-eligible reported spot"
        ),
    )
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--selected-genes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coord-key", default="spatial")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--gene-column", default="gene")
    parser.add_argument("--gene-start", type=int, default=0)
    parser.add_argument("--gene-stop", type=int)
    parser.add_argument(
        "--observation-model",
        choices=["gaussian", "poisson-log"],
        default="gaussian",
        help=(
            "Downstream GP fitting likelihood; length-scale screening always "
            "uses Gaussian likelihood on full-library-normalized expression. "
            "poisson-log fitting uses raw counts with a full-library exposure offset"
        ),
    )
    parser.add_argument(
        "--normalization-scope",
        choices=["full-library", "exposure-offset"],
        default="full-library",
    )
    parser.add_argument("--target-sum", type=float, default=1.0e4)
    parser.add_argument("--transform", choices=["log1p", "sqrt", "none"], default="log1p")
    parser.add_argument("--coordinate-units", choices=["micron", "raw_export_unit"], default="micron")
    parser.add_argument("--random-holdout-fraction", type=float, default=0.02)
    parser.add_argument("--blocked-holdout-fraction", type=float, default=0.02)
    parser.add_argument("--holdout-block-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, required=True)
    parser.add_argument("--global-grid-eigensolver", choices=["torch", "jacobi-experimental"],
                        default="torch", help="Opt-in FP64 batched CUDA grid eigensolver; experimental, not the default.")
    parser.add_argument("--global-reuse-identical-axes", action="store_true",
                        help="Reuse a grid-factor eigendecomposition within an update only for identical axes and isotropic parameters; otherwise use both factors.")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    add_storage_arguments(parser)
    parser.add_argument("--omit-gaussian-latent-aliases", action="store_true",
                        help="Compatibility flag: omit Gaussian latent aliases even when latent-map retention is requested.")

    # Scale-screen options.
    parser.add_argument("--screen-feature-chunk-size", type=int, default=32)
    parser.add_argument(
        "--length-scale-policy",
        choices=["physical_multiscale_v1", "grid_aware_log2"],
        default="physical_multiscale_v1",
        help=(
            "Physical global screening and fixed local scale by default; "
            "grid_aware_log2 reproduces the original screening and trainable local GP."
        ),
    )
    parser.add_argument("--sampling-interval-um", type=float, default=None,
                        help="Physical observation pitch; by default estimate from the full coordinate set.")
    parser.add_argument("--physical-local-floor-um", type=float, default=10.0)
    parser.add_argument("--physical-global-to-local-ratio", type=float, default=4.0)
    parser.add_argument("--physical-upper-fov-fraction", type=float, default=0.5)
    parser.add_argument("--physical-max-candidates", type=int, default=8)
    parser.add_argument("--auto-scale-start-grid-intervals", type=float, default=1.0)
    parser.add_argument("--auto-scale-stop-grid-intervals", type=float, default=16.0)
    parser.add_argument("--auto-scale-steps-per-octave", type=int, default=4)
    parser.add_argument("--max-cg-iterations", type=int, default=1000)
    parser.add_argument("--cg-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--max-preconditioner-size", type=int, default=0)
    parser.add_argument("--minimum-loglik-gap", type=float, default=2.0)
    parser.add_argument("--length-scale-start-index", type=int, default=0)
    parser.add_argument("--length-scale-stop-index", type=int, default=None)
    parser.add_argument(
        "--write-holdout-table",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable duplicate per-spot holdout CSVs in candidate-scale shards.",
    )

    # Joint fit options.
    parser.add_argument("--initial-length-scale-table", type=Path)
    parser.add_argument("--initial-length-scale-column", default="best_length_scale")
    parser.add_argument("--global-epochs", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=30)
    parser.add_argument("--joint-epochs", type=int, default=10)
    parser.add_argument("--batch-size-spots", type=int, default=32768)
    parser.add_argument("--eval-batch-size-spots", type=int, default=65536)
    parser.add_argument("--peak-chunk-size", type=int, default=16)
    parser.add_argument("--global-learning-rate", type=float, default=0.01)
    parser.add_argument("--local-learning-rate", type=float, default=0.01)
    parser.add_argument("--global-hyperparameter-learning-rate", type=float, default=0.05)
    parser.add_argument("--local-hyperparameter-learning-rate", type=float, default=0.02)
    parser.add_argument("--global-initial-length-scale", type=float, default=80.0)
    parser.add_argument("--global-min-length-scale", type=float, default=40.0)
    parser.add_argument("--local-initial-length-scale", type=float, default=16.0)
    parser.add_argument("--local-min-length-scale", type=float, default=8.0)
    parser.add_argument("--local-max-length-scale", type=float, default=32.0)
    parser.add_argument("--local-neighbors", type=int, default=16)
    parser.add_argument(
        "--vnngp-geometry-cache-dir",
        type=Path,
        default=None,
        help=(
            "Shared content-addressed cache for the Morton order and predecessor "
            "DAG. Point every feature shard for one section/fit partition here."
        ),
    )
    parser.add_argument(
        "--vnngp-geometry-cache-mode",
        choices=["off", "read", "readwrite"],
        default="off",
        help="read requires an existing compatible cache; readwrite builds it once and reuses it.",
    )
    parser.add_argument("--local-kl-batch-size", type=int, default=1024)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--max-spatial-metric-spots", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    _validate_args(args)
    _claim_output_dir(args.output_dir)
    # Screening is a shared Gaussian initializer, independent of the eventual
    # observation model. In particular, a Poisson fit must not cause raw counts
    # or zero-exposure row filtering to enter the Gaussian screening stage.
    if args.mode == "screen" or args.observation_model == "gaussian":
        expression, coords, genes, panel = _load_preprocessed_panel(
            args.h5ad,
            args.selected_genes,
            coord_key=args.coord_key,
            layer=args.layer,
            gene_column=args.gene_column,
            gene_start=args.gene_start,
            gene_stop=args.gene_stop,
            target_sum=args.target_sum,
            transform=args.transform,
        )
        exposure = None
    else:
        expression, coords, genes, panel, exposure = _load_count_panel(
            args.h5ad,
            args.selected_genes,
            coord_key=args.coord_key,
            layer=args.layer,
            gene_column=args.gene_column,
            gene_start=args.gene_start,
            gene_stop=args.gene_stop,
        )
    likelihood_eligible = (
        np.ones(coords.shape[0], dtype=bool)
        if exposure is None
        else np.asarray(exposure > 0.0, dtype=bool)
    )
    eligible_rows = np.flatnonzero(likelihood_eligible)
    eligible_holdouts = make_spatial_holdouts(
        coords[eligible_rows],
        random_fraction=args.random_holdout_fraction,
        blocked_fraction=args.blocked_holdout_fraction,
        block_size=args.holdout_block_size,
        random_state=args.random_state,
    )
    holdouts = SpatialHoldouts(
        random=np.zeros(coords.shape[0], dtype=bool),
        blocked=np.zeros(coords.shape[0], dtype=bool),
    )
    holdouts.random[eligible_rows] = eligible_holdouts.random
    holdouts.blocked[eligible_rows] = eligible_holdouts.blocked
    training_mask = likelihood_eligible & ~holdouts.combined
    if args.write_holdout_table:
        _write_holdouts(
            args.output_dir,
            coords,
            holdouts,
            likelihood_eligible=likelihood_eligible,
        )

    if args.mode == "screen":
        _run_screen(args, expression, coords, genes, panel, training_mask, started)
    else:
        _run_fit(
            args,
            expression,
            coords,
            genes,
            panel,
            training_mask,
            holdouts,
            started,
            exposure=exposure,
            likelihood_eligible=likelihood_eligible,
        )


def _load_preprocessed_panel(
    h5ad_path: Path,
    panel_path: Path,
    *,
    coord_key: str,
    layer: str | None,
    gene_column: str,
    gene_start: int,
    gene_stop: int | None,
    target_sum: float,
    transform: str,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, pd.DataFrame]:
    panel = pd.read_csv(panel_path)
    if gene_column not in panel:
        raise ValueError(f"Selected-gene table is missing {gene_column!r}")
    panel = panel.copy()
    panel[gene_column] = panel[gene_column].astype(str)
    if panel[gene_column].duplicated().any():
        raise ValueError("Selected-gene table contains duplicate genes")
    panel["panel_position"] = np.arange(panel.shape[0], dtype=np.int64)
    stop = panel.shape[0] if gene_stop is None else int(gene_stop)
    if not 0 <= int(gene_start) < stop <= panel.shape[0]:
        raise ValueError("Require 0 <= gene_start < gene_stop <= panel size")
    panel = panel.iloc[int(gene_start):stop].reset_index(drop=True)

    adata = ad.read_h5ad(h5ad_path)
    matrix = adata.X if layer is None else adata.layers[layer]
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(np.asarray(matrix))
    matrix = matrix.tocsr()
    if matrix.shape[0] == 0:
        raise ValueError("H5AD contains no observations")
    var_names = pd.Index(adata.var_names.astype(str))
    selected = var_names.get_indexer(panel[gene_column])
    if np.any(selected < 0):
        missing = panel.loc[selected < 0, gene_column].head(10).tolist()
        raise ValueError(f"Selected genes are absent from H5AD: {missing}")
    if coord_key not in adata.obsm:
        raise KeyError(f"H5AD is missing obsm[{coord_key!r}]")
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    if coords.shape[0] != matrix.shape[0] or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must have shape n_obs x >=2")
    coords = np.ascontiguousarray(coords[:, :2])
    if not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates contain non-finite values")
    full_totals = np.asarray(matrix.sum(axis=1), dtype=np.float32).ravel()
    expression = normalize_transform_csr_with_row_sums(
        matrix[:, selected],
        full_totals,
        target_sum=float(target_sum),
        transform=transform,
    )
    return expression, coords, panel[gene_column].to_numpy(str), panel


def _load_count_panel(
    h5ad_path: Path,
    panel_path: Path,
    *,
    coord_key: str,
    layer: str | None,
    gene_column: str,
    gene_start: int,
    gene_stop: int | None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Load raw selected-gene counts plus full-library exposure."""
    panel = pd.read_csv(panel_path)
    if gene_column not in panel:
        raise ValueError(f"Selected-gene table is missing {gene_column!r}")
    panel = panel.copy()
    panel[gene_column] = panel[gene_column].astype(str)
    if panel[gene_column].duplicated().any():
        raise ValueError("Selected-gene table contains duplicate genes")
    panel["panel_position"] = np.arange(panel.shape[0], dtype=np.int64)
    stop = panel.shape[0] if gene_stop is None else int(gene_stop)
    if not 0 <= int(gene_start) < stop <= panel.shape[0]:
        raise ValueError("Require 0 <= gene_start < gene_stop <= panel size")
    panel = panel.iloc[int(gene_start):stop].reset_index(drop=True)

    adata = ad.read_h5ad(h5ad_path)
    matrix = adata.X if layer is None else adata.layers[layer]
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(np.asarray(matrix))
    matrix = matrix.tocsr()
    var_names = pd.Index(adata.var_names.astype(str))
    selected = var_names.get_indexer(panel[gene_column])
    if np.any(selected < 0):
        missing = panel.loc[selected < 0, gene_column].head(10).tolist()
        raise ValueError(f"Selected genes are absent from H5AD: {missing}")
    if coord_key not in adata.obsm:
        raise KeyError(f"H5AD is missing obsm[{coord_key!r}]")
    coords = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    if coords.shape[0] != matrix.shape[0] or coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Spatial coordinates must have shape n_obs x >=2")
    coords = np.ascontiguousarray(coords[:, :2])
    if not np.isfinite(coords).all():
        raise ValueError("Spatial coordinates contain non-finite values")
    exposure = np.asarray(matrix.sum(axis=1), dtype=np.float32).ravel()
    if not np.isfinite(exposure).all() or np.any(exposure < 0.0):
        raise ValueError("Count-GP exposure must be finite and nonnegative")
    if not np.any(exposure > 0.0):
        raise ValueError("Count-GP requires at least one positive-exposure bin")
    expression = matrix[:, selected].tocsr().astype(np.float32)
    if expression.data.size and (
        not np.isfinite(expression.data).all() or np.any(expression.data < 0.0)
    ):
        raise ValueError("Count-GP observations must be finite and nonnegative")
    return (
        expression,
        coords,
        panel[gene_column].to_numpy(str),
        panel,
        exposure,
    )


def _length_scale_screen_settings(args) -> InitialLengthScaleScreenSettings:
    """Use the same screen policy in the screen and fit stages."""
    return InitialLengthScaleScreenSettings(
        length_scales=None,
        length_scale_policy=args.length_scale_policy,
        sampling_interval_um=args.sampling_interval_um,
        physical_local_floor_um=args.physical_local_floor_um,
        physical_global_floor_um=args.global_min_length_scale,
        physical_global_to_local_ratio=args.physical_global_to_local_ratio,
        physical_upper_fov_fraction=args.physical_upper_fov_fraction,
        physical_max_candidates=args.physical_max_candidates,
        auto_scale_start_grid_intervals=args.auto_scale_start_grid_intervals,
        auto_scale_stop_grid_intervals=args.auto_scale_stop_grid_intervals,
        auto_scale_steps_per_octave=args.auto_scale_steps_per_octave,
        auto_scale_interval_reduction="max",
        backend="ski",
        feature_chunk_size=args.screen_feature_chunk_size,
        grid_size=args.grid_size,
        max_cg_iterations=args.max_cg_iterations,
        cg_tolerance=args.cg_tolerance,
        max_preconditioner_size=args.max_preconditioner_size,
        dtype=args.dtype,
        device=args.device,
        random_state=args.random_state,
    )


def _run_screen(args, expression, coords, genes, panel, training_mask, started) -> None:
    settings = _length_scale_screen_settings(args)
    full_scales, full_scale_generation = resolve_initial_length_scales(
        coords[training_mask], settings
    )
    start_index = int(args.length_scale_start_index)
    stop_index = (
        int(full_scales.size)
        if args.length_scale_stop_index is None
        else int(args.length_scale_stop_index)
    )
    if not 0 <= start_index < stop_index <= int(full_scales.size):
        raise ValueError(
            "Require 0 <= length-scale-start-index < length-scale-stop-index "
            f"<= {int(full_scales.size)}"
        )
    if stop_index - start_index < 2:
        raise ValueError("Each candidate-scale shard must contain at least two scales")
    settings.length_scales = tuple(full_scales[start_index:stop_index])
    settings.candidate_scale_offset = start_index
    scores, screen_summary = screen_initial_length_scales(
        expression[training_mask], coords[training_mask], genes, settings
    )
    screen_summary["candidate_partition"] = {
        "full_length_scales": full_scales.tolist(),
        "full_length_scale_generation": full_scale_generation,
        "start_index": start_index,
        "stop_index": stop_index,
        "n_full_candidates": int(full_scales.size),
        "n_shard_candidates": int(stop_index - start_index),
        "complete": bool(start_index == 0 and stop_index == full_scales.size),
    }
    scores = mark_uncertain_initial_scales(
        scores, minimum_loglik_gap=args.minimum_loglik_gap
    )
    add = [column for column in panel.columns if column == args.gene_column or column not in scores]
    scores = scores.merge(
        panel[add], left_on="gene", right_on=args.gene_column,
        how="left", validate="one_to_one"
    )
    if args.gene_column != "gene":
        scores = scores.drop(columns=[args.gene_column])
    scores.to_csv(args.output_dir / "initial_length_scale_scores.csv", index=False)
    scores.loc[scores["recommended_for_multistart"]].to_csv(
        args.output_dir / "selective_multistart_genes.csv", index=False
    )
    summary = {
        "status": "complete",
        "mode": "screen",
        "screening_likelihood": "gaussian",
        "downstream_observation_model": args.observation_model,
        "length_scale_policy": args.length_scale_policy,
        "fit_purpose": "evaluation",
        "downstream_ready": False,
        "input": _input_provenance(args),
        "preprocessing": _preprocessing_provenance(args),
        "shape": {"n_spots": int(expression.shape[0]), "n_genes": int(expression.shape[1])},
        "partition": _partition_summary(training_mask),
        "screen": screen_summary,
        "minimum_loglik_gap": float(args.minimum_loglik_gap),
        "n_recommended_for_multistart": int(scores["recommended_for_multistart"].sum()),
        "outputs": {
            "scores": "initial_length_scale_scores.csv",
            "selective_multistart_genes": "selective_multistart_genes.csv",
            "holdouts": (
                "spatial_holdouts.csv.gz" if args.write_holdout_table else None
            ),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    _write_json(args.output_dir / "initial_length_scale_screen_summary.json", summary)
    print(f"Screened {genes.size} genes into {args.output_dir}", flush=True)


def _run_fit(
    args,
    expression,
    coords,
    genes,
    panel,
    training_mask,
    holdouts,
    started,
    *,
    exposure=None,
    likelihood_eligible=None,
) -> None:
    if args.initial_length_scale_table is None:
        raise ValueError("--initial-length-scale-table is required in fit mode")
    _, scale_policy = resolve_initial_length_scales(
        coords[training_mask], _length_scale_screen_settings(args)
    )
    physical = args.length_scale_policy == "physical_multiscale_v1"
    global_min = (
        float(scale_policy["global_min_length_scale_um"])
        if physical else float(args.global_min_length_scale)
    )
    local_initial = (
        float(scale_policy["local_length_scale_um"])
        if physical else float(args.local_initial_length_scale)
    )
    initial_scales, init_audit = _align_initial_scales(
        args.initial_length_scale_table,
        genes,
        column=args.initial_length_scale_column,
        global_min=global_min,
    )
    base = GlobalLocalGPSettings(
        global_grid_size=args.grid_size,
        global_epochs=args.global_epochs,
        local_epochs=args.local_epochs,
        batch_size_spots=args.batch_size_spots,
        peak_chunk_size=args.peak_chunk_size,
        eval_batch_size_spots=args.eval_batch_size_spots,
        global_learning_rate=args.global_learning_rate,
        local_learning_rate=args.local_learning_rate,
        global_hyperparameter_learning_rate=args.global_hyperparameter_learning_rate,
        local_hyperparameter_learning_rate=args.local_hyperparameter_learning_rate,
        global_initial_length_scale=(
            max(args.global_initial_length_scale, 1.05 * global_min)
            if physical else args.global_initial_length_scale
        ),
        local_initial_length_scale=local_initial,
        local_min_length_scale=(
            0.5 * local_initial if physical else args.local_min_length_scale
        ),
        local_max_length_scale=(
            2.0 * local_initial if physical else args.local_max_length_scale
        ),
        local_neighbors=args.local_neighbors,
        geometry_cache_dir=(
            None
            if args.vnngp_geometry_cache_dir is None
            else str(args.vnngp_geometry_cache_dir)
        ),
        geometry_cache_mode=args.vnngp_geometry_cache_mode,
        dtype=args.dtype,
        **storage_kwargs(args),
        device=args.device,
        random_state=args.random_state,
    )
    settings = JointGlobalLocalGPSettings(
        **asdict(base),
        joint_epochs=args.joint_epochs,
        global_min_length_scale=global_min,
        fixed_local_length_scale=physical,
        local_kl_batch_size=args.local_kl_batch_size,
        local_kl_mode="stochastic",
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        local_pretrain_target="global_residual",
        observation_model=args.observation_model,
        global_grid_eigensolver=getattr(args, "global_grid_eigensolver", "torch"),
        global_reuse_identical_axes=getattr(args, "global_reuse_identical_axes", False),
    )
    model_output = args.output_dir / "fitted_maps"
    training_rows = np.flatnonzero(training_mask)
    fit_started = time.time()
    peak_gpu_memory_bytes = _reset_peak_gpu_memory(args.device)
    manifest = fit_joint_global_local_gp_spatial_maps(
        expression[training_mask],
        coords[training_mask],
        genes,
        model_output,
        settings=settings,
        prediction_coords=coords,
        prediction_spot_indices=np.arange(coords.shape[0], dtype=np.int64),
        training_prediction_rows=training_rows,
        global_initial_length_scales=initial_scales,
        exposure=(None if exposure is None else exposure[training_mask]),
        include_gaussian_latent_aliases=(False if getattr(args, "omit_gaussian_latent_aliases", False) else None),
    )
    coverage = _fit_coverage_contract(
        args.fit_purpose,
        training_mask,
        required_training_mask=likelihood_eligible,
    )
    manifest.update(coverage)
    peak_gpu_memory_bytes = _read_peak_gpu_memory(
        args.device, peak_gpu_memory_bytes
    )
    manifest["coordinate_provenance"] = {
        "coordinate_scale": "raw",
        "input_coordinate_units": args.coordinate_units,
        "model_coordinate_units": args.coordinate_units,
    }
    manifest["preprocessing"] = _preprocessing_provenance(args)
    manifest["panel_sha256"] = _file_sha256(args.selected_genes)
    manifest["global_length_scale_initialization"] = {
        key: value
        for key, value in init_audit.items()
        if key != "n_values_floored_above_global_minimum"
    }
    manifest["length_scale_policy"] = scale_policy
    _write_json(model_output / "vski_gp_maps_manifest.json", manifest)
    _annotate_prediction_sites(
        model_output,
        training_mask,
        likelihood_eligible=likelihood_eligible,
    )
    _annotate_model_metadata(model_output, panel)
    metrics = (
        _stream_fidelity(
            model_output,
            expression,
            coords,
            genes,
            training_mask,
            holdouts,
            max_spatial_metric_spots=args.max_spatial_metric_spots,
            random_state=args.random_state,
            grid_size=args.grid_size,
            grid_axes=manifest.get("inducing_grid_axes"),
        )
        if args.observation_model == "gaussian"
        else _stream_count_fidelity(
            model_output,
            expression,
            genes,
            exposure,
            training_mask,
        )
    )
    metrics.to_csv(args.output_dir / "gp_fidelity_metrics.csv", index=False)
    summary = {
        "status": "complete",
        "backend": "joint_global_local",
        **coverage,
        "inputs": _common_input_provenance(args),
        "feature_shard": {
            "gene_start": int(args.gene_start),
            "gene_stop": args.gene_stop,
            "n_values_floored_above_global_minimum": int(
                init_audit["n_values_floored_above_global_minimum"]
            ),
        },
        "shape": {
            "n_spots": int(expression.shape[0]),
            "n_training_spots": int(training_mask.sum()),
            "n_prediction_spots": int(expression.shape[0]),
            "n_features": int(expression.shape[1]),
        },
        "preprocessing": _preprocessing_provenance(args),
        "holdouts": {
            "random_spots": int(holdouts.random.sum()),
            "blocked_spots": int(holdouts.blocked.sum()),
        },
        "settings": asdict(settings),
        "length_scale_policy": scale_policy,
        "fit_manifest": manifest,
        "fidelity_summary": (
            {
                name: summarize_fidelity(table.drop(columns=["evaluation_set"]))
                for name, table in metrics.groupby("evaluation_set")
            }
            if args.observation_model == "gaussian"
            else {
                "training_reconstruction": {
                    "median_poisson_deviance_explained": float(
                        metrics["poisson_deviance_explained"].median()
                    ),
                    "median_observed_mean_count": float(
                        metrics["observed_mean_count"].median()
                    ),
                    "median_predicted_mean_count": float(
                        metrics["predicted_mean_count"].median()
                    ),
                }
            }
        ),
        "fit_elapsed_seconds": float(time.time() - fit_started),
        "peak_gpu_memory_bytes": int(peak_gpu_memory_bytes),
        "elapsed_seconds": float(time.time() - started),
    }
    _write_json(args.output_dir / "gp_fidelity_benchmark_summary.json", summary)
    _write_json(model_output / "vski_run_summary.json", summary)
    print(f"Fitted {genes.size} genes into {args.output_dir}", flush=True)


def _stream_fidelity(
    model_output, expression, coords, genes, training_mask, holdouts, *,
    max_spatial_metric_spots, random_state, grid_size, grid_axes,
) -> pd.DataFrame:
    metadata = pd.read_csv(model_output / "vski_gp_model_metadata.csv").sort_values("feature_position")
    chunks = pd.read_csv(model_output / "vski_gp_map_chunks.csv").sort_values("chunk_id")
    tables: list[pd.DataFrame] = []
    masks = [
        ("training_reconstruction", training_mask),
        ("random_holdout", holdouts.random),
        ("blocked_holdout", holdouts.blocked),
        ("combined_holdout", holdouts.combined),
    ]
    for row in chunks.itertuples(index=False):
        members = metadata.loc[metadata["map_chunk_id"].astype(int) == int(row.chunk_id)].sort_values("map_column")
        positions = members["feature_position"].to_numpy(dtype=np.int64)
        observed = expression[:, positions].toarray().astype(np.float32, copy=False)
        means = members["standardization_mean"].to_numpy(dtype=np.float32)
        stds = members["standardization_std"].to_numpy(dtype=np.float32)
        stds[stds < 1.0e-8] = 1.0
        observed = (observed - means[None, :]) / stds[None, :]
        with np.load(model_output / str(row.file), allow_pickle=False) as archive:
            fitted_mean = np.asarray(archive["posterior_mean"], dtype=np.float32)
            if "posterior_predictive_variance" in archive:
                fitted_var = np.asarray(archive["posterior_predictive_variance"], dtype=np.float32)
            else:
                fitted_var = np.asarray(archive["posterior_variance"], dtype=np.float32)
                fitted_var = fitted_var + members["learned_noise_variance"].to_numpy(dtype=np.float32)[None, :]
        chunk_genes = genes[positions]
        for name, mask in masks:
            if not mask.any():
                continue
            table = evaluate_gp_fidelity(
                observed, fitted_mean, fitted_var, coords, chunk_genes,
                evaluation_mask=mask,
                max_spatial_samples=max_spatial_metric_spots,
                random_state=random_state,
                grid_size=grid_size,
                grid_axes=grid_axes,
            )
            table.insert(0, "evaluation_set", name)
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _stream_count_fidelity(
    model_output: Path,
    expression: sparse.csr_matrix,
    genes: np.ndarray,
    exposure: np.ndarray,
    training_mask: np.ndarray,
) -> pd.DataFrame:
    """Compute exposure-aware Poisson reconstruction diagnostics per gene."""
    if exposure is None:
        raise ValueError("Count fidelity requires full-library exposure")
    exposure = np.asarray(exposure, dtype=np.float64)
    reference = float(np.median(exposure[training_mask]))
    normalized_exposure = exposure / reference
    metadata = pd.read_csv(
        model_output / "vski_gp_model_metadata.csv"
    ).sort_values("feature_position")
    chunks = pd.read_csv(model_output / "vski_gp_map_chunks.csv").sort_values(
        "chunk_id"
    )
    records: list[dict] = []
    for row in chunks.itertuples(index=False):
        members = metadata.loc[
            metadata["map_chunk_id"].astype(int) == int(row.chunk_id)
        ].sort_values("map_column")
        positions = members["feature_position"].to_numpy(dtype=np.int64)
        observed = expression[:, positions].toarray().astype(np.float64, copy=False)
        with np.load(model_output / str(row.file), allow_pickle=False) as archive:
            fitted_rate = np.asarray(archive["posterior_mean"], dtype=np.float64)
        predicted = fitted_rate * normalized_exposure[:, None]
        observed_training = observed[training_mask]
        predicted_training = predicted[training_mask]
        exposure_training = normalized_exposure[training_mask]
        null_rate = (
            observed_training.sum(axis=0, dtype=np.float64) + 0.5
        ) / (exposure_training.sum(dtype=np.float64) + 0.5)
        null_prediction = exposure_training[:, None] * null_rate[None, :]
        model_deviance = _mean_poisson_deviance(
            observed_training, predicted_training
        )
        null_deviance = _mean_poisson_deviance(
            observed_training, null_prediction
        )
        explained = 1.0 - model_deviance / np.maximum(null_deviance, 1.0e-12)
        for column, position in enumerate(positions):
            records.append(
                {
                    "evaluation_set": "training_reconstruction",
                    "gene": str(genes[position]),
                    "feature_position": int(position),
                    "observed_mean_count": float(observed_training[:, column].mean()),
                    "predicted_mean_count": float(predicted_training[:, column].mean()),
                    "positive_bin_fraction": float(
                        np.count_nonzero(observed_training[:, column])
                        / observed_training.shape[0]
                    ),
                    "mean_poisson_deviance": float(model_deviance[column]),
                    "null_mean_poisson_deviance": float(null_deviance[column]),
                    "poisson_deviance_explained": float(explained[column]),
                }
            )
    return pd.DataFrame.from_records(records)


def _mean_poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    expected = np.maximum(np.asarray(expected, dtype=np.float64), 1.0e-12)
    observed = np.asarray(observed, dtype=np.float64)
    terms = expected - observed
    positive = observed > 0.0
    terms[positive] += observed[positive] * np.log(
        observed[positive] / expected[positive]
    )
    return 2.0 * terms.mean(axis=0)


def _align_initial_scales(table_path: Path, genes: np.ndarray, *, column: str, global_min: float):
    table = pd.read_csv(table_path)
    if "gene" not in table or column not in table:
        raise ValueError(f"Initializer table must contain gene and {column}")
    if table["gene"].astype(str).duplicated().any():
        raise ValueError("Initializer table contains duplicate genes")
    lookup = pd.Series(pd.to_numeric(table[column], errors="coerce").to_numpy(), index=table["gene"].astype(str))
    aligned = pd.Series(genes.astype(str)).map(lookup)
    if aligned.isna().any():
        missing = genes[aligned.isna().to_numpy()][:10].tolist()
        raise ValueError(f"Initializer table lacks genes: {missing}")
    values = aligned.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Initial length scales must be finite and positive")
    values, interior_audit = interior_lengthscale_initialization(values, global_min)
    return values, {
        "mode": "per_feature_screen",
        "source_table": str(table_path),
        "source_table_sha256": _file_sha256(table_path),
        "source_column": column,
        "global_min_length_scale": float(global_min),
        **interior_audit,
    }


def _annotate_model_metadata(model_output: Path, panel: pd.DataFrame) -> None:
    path = model_output / "vski_gp_model_metadata.csv"
    metadata = pd.read_csv(path)
    add = [column for column in panel.columns if column == "gene" or column not in metadata]
    annotated = metadata.merge(panel[add], on="gene", how="left", validate="one_to_one")
    if annotated["panel_position"].isna().any():
        raise ValueError("Failed to align model metadata to selected panel")
    annotated.to_csv(path, index=False)


def _write_holdouts(
    output_dir: Path,
    coords: np.ndarray,
    holdouts,
    *,
    likelihood_eligible: np.ndarray | None = None,
) -> None:
    partition = np.full(coords.shape[0], "training", dtype=object)
    if likelihood_eligible is not None:
        likelihood_eligible = np.asarray(likelihood_eligible, dtype=bool)
        partition[~likelihood_eligible] = "prediction_only_zero_exposure"
    partition[holdouts.random] = "random_holdout"
    partition[holdouts.blocked] = "blocked_holdout"
    pd.DataFrame({
        "map_row": np.arange(coords.shape[0], dtype=np.int64),
        "spot_index": np.arange(coords.shape[0], dtype=np.int64),
        "x": coords[:, 0], "y": coords[:, 1], "partition": partition,
    }).to_csv(output_dir / "spatial_holdouts.csv.gz", index=False)


def _annotate_prediction_sites(
    model_output: Path,
    training_mask: np.ndarray,
    *,
    likelihood_eligible: np.ndarray | None,
) -> None:
    """Record why each complete-support map row was or was not in likelihood."""
    path = model_output / "vski_gp_map_spots.csv.gz"
    spots = pd.read_csv(path)
    training_mask = np.asarray(training_mask, dtype=bool)
    eligible = (
        np.ones(training_mask.size, dtype=bool)
        if likelihood_eligible is None
        else np.asarray(likelihood_eligible, dtype=bool)
    )
    if spots.shape[0] != training_mask.size or eligible.shape != training_mask.shape:
        raise ValueError("Spot provenance masks do not match fitted map rows")
    spots["used_for_training"] = training_mask
    spots["prediction_only"] = ~training_mask
    spots["likelihood_eligible"] = eligible
    spots["zero_exposure"] = ~eligible
    provenance = np.full(spots.shape[0], "observed_likelihood", dtype=object)
    provenance[eligible & ~training_mask] = "held_out_observation"
    provenance[~eligible] = "zero_exposure_model_inference"
    spots["inference_provenance"] = provenance
    spots.to_csv(path, index=False)


def _reset_peak_gpu_memory(device: str | None) -> int:
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            torch.cuda.reset_peak_memory_stats()
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return 0


def _read_peak_gpu_memory(device: str | None, fallback: int) -> int:
    try:
        import torch

        if torch.cuda.is_available() and (device is None or str(device).startswith("cuda")):
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return int(fallback)


def _partition_summary(
    training_mask: np.ndarray,
    required_training_mask: np.ndarray | None = None,
) -> dict:
    result = {
        "n_total_spots": int(training_mask.size),
        "n_training_spots": int(training_mask.sum()),
        "n_prediction_only_spots": int((~training_mask).sum()),
        "training_coverage_fraction": float(training_mask.mean()),
    }
    if required_training_mask is not None:
        result.update(
            _likelihood_coverage_summary(training_mask, required_training_mask)
        )
    return result


def _fit_coverage_contract(
    fit_purpose: str,
    training_mask: np.ndarray,
    required_training_mask: np.ndarray | None = None,
) -> dict:
    training_mask = np.asarray(training_mask, dtype=bool)
    n_prediction = int(training_mask.size)
    n_training = int(training_mask.sum())
    full = n_training == n_prediction
    required = (
        np.ones(training_mask.size, dtype=bool)
        if required_training_mask is None
        else np.asarray(required_training_mask, dtype=bool)
    )
    if required.shape != training_mask.shape or np.any(training_mask & ~required):
        raise ValueError("Training mask must be a subset of required likelihood rows")
    likelihood = _likelihood_coverage_summary(training_mask, required)
    return {
        "fit_purpose": str(fit_purpose),
        "n_training_spots": n_training,
        "n_prediction_spots": n_prediction,
        "n_prediction_only_spots": n_prediction - n_training,
        "training_coverage_fraction": float(n_training / n_prediction),
        "full_training_coverage": bool(full),
        **likelihood,
        "downstream_ready": bool(
            fit_purpose == "production" and likelihood["full_likelihood_coverage"]
        ),
    }


def _likelihood_coverage_summary(
    training_mask: np.ndarray,
    required_training_mask: np.ndarray,
) -> dict:
    required = int(required_training_mask.sum())
    trained_required = int(np.count_nonzero(training_mask & required_training_mask))
    return {
        "n_likelihood_eligible_spots": required,
        "n_zero_exposure_prediction_only_spots": int(required_training_mask.size - required),
        "n_trained_likelihood_spots": trained_required,
        "likelihood_coverage_fraction": float(trained_required / required),
        "full_likelihood_coverage": bool(trained_required == required),
        "complete_prediction_support": True,
    }


def _input_provenance(args) -> dict:
    result = _common_input_provenance(args)
    result.update({
        "gene_start": int(args.gene_start),
        "gene_stop": args.gene_stop,
    })
    return result


def _common_input_provenance(args) -> dict:
    return {
        "h5ad": str(args.h5ad),
        "h5ad_sha256": _file_sha256(args.h5ad),
        "selected_genes": str(args.selected_genes),
        "selected_genes_sha256": _file_sha256(args.selected_genes),
        "coord_key": args.coord_key,
        "layer": args.layer,
    }


def _preprocessing_provenance(args) -> dict:
    gaussian = args.mode == "screen" or args.observation_model == "gaussian"
    return {
        "observation_model": "gaussian" if args.mode == "screen" else args.observation_model,
        "normalization_scope": args.normalization_scope,
        "target_sum": (
            float(args.target_sum)
            if gaussian
            else None
        ),
        "transform": args.transform,
        "exposure": (
            None
            if gaussian
            else "full-library row count normalized by its training-set median"
        ),
        "coordinate_scale": "raw",
        "coordinate_units": args.coordinate_units,
    }


def _validate_args(args) -> None:
    for name in ["grid_size", "screen_feature_chunk_size", "batch_size_spots", "eval_batch_size_spots", "peak_chunk_size"]:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.target_sum <= 0:
        raise ValueError("target_sum must be positive")
    if args.length_scale_policy == "physical_multiscale_v1":
        if args.coordinate_units != "micron":
            raise ValueError(
                "physical_multiscale_v1 requires coordinates in micrometres; "
                "convert the input coordinates or use --length-scale-policy grid_aware_log2"
            )
        physical_values = [
            args.physical_local_floor_um, args.physical_global_to_local_ratio,
            args.physical_upper_fov_fraction,
        ]
        if not np.isfinite(physical_values).all() or min(physical_values) <= 0:
            raise ValueError("Physical length-scale parameters must be finite and positive")
        if args.physical_max_candidates < 2:
            raise ValueError("physical-max-candidates must be at least two")
        if args.sampling_interval_um is not None and (
            not np.isfinite(args.sampling_interval_um) or args.sampling_interval_um <= 0
        ):
            raise ValueError("sampling-interval-um must be finite and positive")
    if args.mode == "screen" or args.observation_model == "gaussian":
        if args.normalization_scope != "full-library":
            raise ValueError("Gaussian screening and Gaussian fits require full-library normalization")
    else:
        if args.normalization_scope != "exposure-offset" or args.transform != "none":
            raise ValueError(
                "poisson-log requires --normalization-scope exposure-offset "
                "and --transform none"
            )
    if args.mode == "fit":
        if args.fit_purpose == "production" and (
            args.random_holdout_fraction != 0.0
            or args.blocked_holdout_fraction != 0.0
        ):
            raise ValueError(
                "Production fits require --random-holdout-fraction 0 and "
                "--blocked-holdout-fraction 0 so every likelihood-eligible "
                "output spot is used for training"
            )
        if args.fit_purpose == "evaluation" and (
            args.random_holdout_fraction == 0.0
            and args.blocked_holdout_fraction == 0.0
        ):
            raise ValueError(
                "Evaluation fits require a nonzero random or blocked holdout fraction; "
                "use --fit-purpose production for a full-support refit"
            )
        if (
            not np.isfinite(args.global_min_length_scale)
            or args.global_min_length_scale <= 0
        ):
            raise ValueError("Global minimum length scale must be finite and positive")
        if (
            not np.isfinite(args.local_min_length_scale)
            or args.local_min_length_scale <= 0
        ):
            raise ValueError("Local minimum length scale must be finite and positive")
        if args.length_scale_policy == "grid_aware_log2" and not (
            args.local_min_length_scale
            < args.local_initial_length_scale
            < args.local_max_length_scale
        ):
            raise ValueError(
                "Local initial length scale must lie strictly inside its bounds"
            )
        if (
            getattr(args, "vnngp_geometry_cache_mode", "off") != "off"
            and getattr(args, "vnngp_geometry_cache_dir", None) is None
        ):
            raise ValueError(
                "--vnngp-geometry-cache-dir is required when the cache mode is enabled"
            )


def _claim_output_dir(path: Path) -> None:
    if path.is_symlink():
        raise FileExistsError(f"Refusing symlink output directory: {path}")
    if path.exists() and any(
        item.is_file() or item.is_symlink() for item in path.rglob("*")
    ):
        raise FileExistsError(f"Refusing output directory containing files: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(_jsonable(value), indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
