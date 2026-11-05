#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SPOT_COLUMNS = ("map_row", "spot_index", "x", "y")
HISTORY_FILES = (
    "vski_training_history.csv",
    "vski_hyperparameter_history.csv",
    "joint_global_local_pretraining_history.csv",
    "joint_global_local_training_history.csv",
    "joint_global_local_hyperparameter_history.csv",
    "joint_global_local_stage_timing.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge independently fitted TEMI V-SKI peak shards."
    )
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument(
        "--shard-data-subdir",
        default=".",
        help="Subdirectory containing map files within each shard, e.g. fitted_maps.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["copy", "symlink"],
        default="copy",
        help=(
            "copy rewrites shard-local feature positions into one global schema. "
            "symlink is accepted only when source positions are already global."
        ),
    )
    parser.add_argument(
        "--annotation-table",
        type=Path,
        help="Optional peak table merged into model metadata by peak_index.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all shard schemas and alignments without copying map arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.validate_only:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError(f"Refusing nonempty output directory: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

    history_frames = {name: [] for name in HISTORY_FILES}
    metadata_frames: list[pd.DataFrame] = []
    chunk_frames: list[pd.DataFrame] = []
    score_frames: list[pd.DataFrame] = []
    selected_indices: list[int] = []
    shard_manifests: list[dict] = []
    shard_summaries: list[dict] = []
    next_chunk_id = 0
    next_feature_position = 0
    first_spot_file: Path | None = None
    reference_alignment: dict | None = None
    reference_spot_signature: dict | None = None
    reference_holdout_signature: dict | None = None
    reference_history_presence: tuple[str, ...] | None = None

    for shard_id in range(args.n_shards):
        shard_dir = args.shard_root / "shards" / f"shard_{shard_id:02d}"
        if not shard_dir.exists():
            raise FileNotFoundError(shard_dir)
        data_dir = shard_dir / args.shard_data_subdir

        manifest_path = data_dir / "vski_gp_maps_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if shard_manifests:
            first = shard_manifests[0]["manifest"]
            for key in ("storage", "output_dtype"):
                if shard_manifest.get(key) != first.get(key):
                    raise ValueError(f"Incompatible shard {key}; do not mix storage policies")
        shard_manifests.append({"shard_id": shard_id, "manifest": shard_manifest})

        summary_path = shard_dir / "gp_fidelity_benchmark_summary.json"
        if not summary_path.exists():
            summary_path = data_dir / "vski_run_summary.json"
        shard_summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.exists()
            else None
        )
        if shard_summary is not None:
            shard_summaries.append({"shard_id": shard_id, "summary": shard_summary})

        alignment = _alignment_signature(shard_manifest, shard_summary)
        reference_alignment = _require_matching_signature(
            "model/preprocessing configuration",
            shard_id,
            alignment,
            reference_alignment,
        )

        spot_file = data_dir / "vski_gp_map_spots.csv.gz"
        spot_signature = _semantic_csv_signature(spot_file, SPOT_COLUMNS)
        reference_spot_signature = _require_matching_signature(
            "spot rows and coordinate order",
            shard_id,
            spot_signature,
            reference_spot_signature,
        )
        if first_spot_file is None:
            first_spot_file = spot_file

        holdout_file = shard_dir / "spatial_holdouts.csv.gz"
        holdout_signature = (
            _semantic_csv_signature(holdout_file, None)
            if holdout_file.exists()
            else None
        )
        reference_holdout_signature = _require_matching_optional_signature(
            "holdout partition",
            shard_id,
            holdout_signature,
            reference_holdout_signature,
            first_shard=(shard_id == 0),
        )

        history_presence = tuple(
            name for name in HISTORY_FILES if (data_dir / name).exists()
        )
        if reference_history_presence is None:
            reference_history_presence = history_presence
        elif history_presence != reference_history_presence:
            raise ValueError(
                f"Shard {shard_id} history files differ from shard 0: "
                f"{history_presence!r} != {reference_history_presence!r}"
            )

        metadata = pd.read_csv(data_dir / "vski_gp_model_metadata.csv")
        chunks = pd.read_csv(data_dir / "vski_gp_map_chunks.csv")
        _validate_shard_tables(metadata, chunks, shard_id)
        original_metadata = metadata.copy()
        chunk_id_map: dict[int, int] = {}
        chunk_positions: dict[int, np.ndarray] = {}

        rewritten_chunks: list[pd.Series] = []
        for _, row in chunks.sort_values("chunk_id").iterrows():
            old_chunk_id = int(row["chunk_id"])
            source_members = original_metadata.loc[
                original_metadata["map_chunk_id"].astype(int) == old_chunk_id
            ].sort_values("map_column")
            n_features = int(row.get("n_features", source_members.shape[0]))
            if source_members.shape[0] != n_features:
                raise ValueError(
                    f"Shard {shard_id} chunk {old_chunk_id} metadata has "
                    f"{source_members.shape[0]} features, expected {n_features}"
                )
            expected_columns = np.arange(n_features, dtype=np.int64)
            np.testing.assert_array_equal(
                source_members["map_column"].to_numpy(dtype=np.int64),
                expected_columns,
                err_msg=f"Shard {shard_id} chunk {old_chunk_id} map columns are invalid",
            )

            new_chunk_id = next_chunk_id
            next_chunk_id += 1
            positions = np.arange(
                next_feature_position,
                next_feature_position + n_features,
                dtype=np.int64,
            )
            next_feature_position += n_features
            chunk_id_map[old_chunk_id] = new_chunk_id
            chunk_positions[old_chunk_id] = positions

            source = data_dir / str(row["file"])
            if not source.exists():
                raise FileNotFoundError(source)
            new_name = (
                f"vski_gp_maps_shard_{shard_id:02d}_"
                f"chunk_{old_chunk_id:04d}.npz"
            )
            destination = args.output_dir / new_name
            _copy_map_chunk(
                source,
                destination,
                positions,
                link_mode=args.link_mode,
                validate_only=args.validate_only,
            )

            rewritten = row.copy()
            rewritten["source_shard_id"] = shard_id
            rewritten["source_shard_dir"] = str(shard_dir)
            rewritten["source_chunk_id"] = old_chunk_id
            rewritten["chunk_id"] = new_chunk_id
            rewritten["file"] = new_name
            rewritten["feature_start"] = int(positions[0])
            rewritten["feature_stop"] = int(positions[-1] + 1)
            rewritten["n_features"] = int(n_features)
            if not args.validate_only:
                rewritten["file_size_bytes"] = int(destination.stat().st_size)
                rewritten["file_sha256"] = _file_sha256(destination)
            rewritten_chunks.append(rewritten)
        chunk_frames.append(pd.DataFrame(rewritten_chunks))

        metadata = original_metadata.copy()
        metadata["source_shard_id"] = shard_id
        metadata["source_chunk_id"] = metadata["map_chunk_id"].astype(int)
        metadata["source_feature_position"] = metadata["feature_position"].astype(int)
        metadata["feature_position"] = [
            int(chunk_positions[int(chunk_id)][int(map_column)])
            for chunk_id, map_column in zip(
                metadata["source_chunk_id"], metadata["map_column"], strict=True
            )
        ]
        metadata["map_chunk_id"] = (
            metadata["source_chunk_id"].map(chunk_id_map).astype(int)
        )
        metadata["map_chunk_file"] = [
            f"vski_gp_maps_shard_{shard_id:02d}_chunk_{int(old):04d}.npz"
            for old in metadata["source_chunk_id"]
        ]
        metadata_frames.append(metadata)
        feature_position_map = dict(
            zip(
                metadata["source_feature_position"].astype(int),
                metadata["feature_position"].astype(int),
                strict=True,
            )
        )

        for name in HISTORY_FILES:
            _append_history(
                data_dir / name,
                history_frames[name],
                shard_id,
                chunk_id_map,
                feature_position_map,
            )

        score_path = (
            args.shard_root / f"selected_vski_score_table_shard_{shard_id:02d}.csv"
        )
        if score_path.exists():
            score = pd.read_csv(score_path)
            score["source_shard_id"] = shard_id
            score_frames.append(score)

        index_path = (
            args.shard_root / f"selected_peak_indices_shard_{shard_id:02d}.txt"
        )
        if index_path.exists():
            selected_indices.extend(
                int(value) for value in index_path.read_text(encoding="utf-8").split()
            )

    if args.validate_only:
        print(
            f"Validated {args.n_shards} aligned shards with "
            f"{next_feature_position} total features"
        )
        return
    if first_spot_file is None:
        raise FileNotFoundError("No shard spot coordinate file was found")
    shutil.copy2(first_spot_file, args.output_dir / "vski_gp_map_spots.csv.gz")

    metadata_all = pd.concat(metadata_frames, ignore_index=True)
    if "peak_index" not in metadata_all.columns and "gene" in metadata_all.columns:
        extracted = metadata_all["gene"].astype(str).str.extract(r"peak_(\d+)$")[0]
        if extracted.notna().all():
            metadata_all["peak_index"] = extracted.astype(int)
    if args.annotation_table is not None:
        annotations = pd.read_csv(args.annotation_table).drop_duplicates("peak_index")
        if "peak_index" not in metadata_all.columns:
            raise ValueError("Cannot merge annotations because peak_index is unavailable")
        add_columns = [
            column
            for column in annotations.columns
            if column == "peak_index" or column not in metadata_all.columns
        ]
        metadata_all = metadata_all.merge(
            annotations[add_columns],
            on="peak_index",
            how="left",
            validate="one_to_one",
        )
    if metadata_all["gene"].duplicated().any():
        duplicated = (
            metadata_all.loc[metadata_all["gene"].duplicated(), "gene"].head().tolist()
        )
        raise ValueError(f"Peak shards contain duplicate genes, e.g. {duplicated}")
    expected_positions = np.arange(metadata_all.shape[0], dtype=np.int64)
    actual_positions = np.sort(metadata_all["feature_position"].to_numpy(dtype=np.int64))
    np.testing.assert_array_equal(
        actual_positions,
        expected_positions,
        err_msg="Merged feature positions are not globally contiguous",
    )
    if "peak_index" in metadata_all.columns:
        metadata_all = metadata_all.sort_values("peak_index").reset_index(drop=True)
    metadata_all.to_csv(args.output_dir / "vski_gp_model_metadata.csv", index=False)

    chunks_all = pd.concat(chunk_frames, ignore_index=True).sort_values("chunk_id")
    chunks_all.to_csv(args.output_dir / "vski_gp_map_chunks.csv", index=False)
    for name, frames in history_frames.items():
        _write_history(frames, args.output_dir / name)
    if score_frames:
        pd.concat(score_frames, ignore_index=True).to_csv(
            args.output_dir / "selected_vski_score_table.csv", index=False
        )
    if selected_indices:
        selected_sorted = sorted(set(selected_indices))
        (args.output_dir / "selected_peak_indices.txt").write_text(
            "\n".join(map(str, selected_sorted)) + "\n", encoding="utf-8"
        )

    first_manifest = shard_manifests[0]["manifest"]
    n_spots = int(first_manifest.get("n_spots") or first_manifest["n_prediction_spots"])
    combined_manifest = {
        "storage": first_manifest.get("storage"),
        "output_dtype": first_manifest.get("output_dtype"),
        "method": first_manifest.get("method", "V-SKI Gaussian variational GP"),
        "merge_mode": "validated_peak_shards",
        "shard_root": str(args.shard_root),
        "n_shards": int(args.n_shards),
        "n_spots": n_spots,
        "n_features": int(metadata_all.shape[0]),
        "fit_purpose": first_manifest.get("fit_purpose"),
        "n_training_spots": first_manifest.get("n_training_spots"),
        "n_prediction_spots": first_manifest.get("n_prediction_spots", n_spots),
        "n_prediction_only_spots": first_manifest.get("n_prediction_only_spots"),
        "training_coverage_fraction": first_manifest.get("training_coverage_fraction"),
        "full_training_coverage": first_manifest.get("full_training_coverage"),
        "n_likelihood_eligible_spots": first_manifest.get(
            "n_likelihood_eligible_spots"
        ),
        "n_zero_exposure_prediction_only_spots": first_manifest.get(
            "n_zero_exposure_prediction_only_spots"
        ),
        "n_trained_likelihood_spots": first_manifest.get(
            "n_trained_likelihood_spots"
        ),
        "likelihood_coverage_fraction": first_manifest.get(
            "likelihood_coverage_fraction"
        ),
        "full_likelihood_coverage": first_manifest.get(
            "full_likelihood_coverage"
        ),
        "complete_prediction_support": first_manifest.get(
            "complete_prediction_support"
        ),
        "downstream_ready": first_manifest.get("downstream_ready", False),
        "grid_size": first_manifest.get(
            "grid_size", first_manifest.get("settings", {}).get("global_grid_size")
        ),
        "grid_bounds": first_manifest.get("grid_bounds"),
        "inducing_grid_axes": first_manifest.get("inducing_grid_axes"),
        "coordinate_provenance": first_manifest.get("coordinate_provenance"),
        "observation_model": first_manifest.get("observation_model", "gaussian"),
        "exposure": first_manifest.get("exposure"),
        "posterior_map_scale": first_manifest.get("posterior_map_scale"),
        "posterior_variance_semantics": first_manifest.get(
            "posterior_variance_semantics"
        ),
        "posterior_predictive_variance_semantics": first_manifest.get(
            "posterior_predictive_variance_semantics"
        ),
        "metadata_file": "vski_gp_model_metadata.csv",
        "spot_file": "vski_gp_map_spots.csv.gz",
        "chunk_index_file": "vski_gp_map_chunks.csv",
        "spot_alignment_signature": reference_spot_signature,
        "holdout_alignment_signature": reference_holdout_signature,
        "configuration_signature": reference_alignment,
        "feature_position_schema": "global contiguous positions 0..n_features-1",
        "shards": shard_manifests,
        "shard_summaries": shard_summaries,
    }
    _add_existing_history_paths(combined_manifest, history_frames)
    (args.output_dir / "vski_gp_maps_manifest.json").write_text(
        json.dumps(_jsonable(combined_manifest), indent=2), encoding="utf-8"
    )

    first_summary = shard_summaries[0]["summary"] if shard_summaries else {}
    summary = {
        "source": "validated_merged_peak_shards",
        "input_source": first_summary.get("source"),
        "shard_root": str(args.shard_root),
        "output_dir": str(args.output_dir),
        "n_shards": int(args.n_shards),
        "shape": {
            "n_spots": int(combined_manifest["n_spots"]),
            "n_training_spots": combined_manifest.get("n_training_spots"),
            "n_prediction_spots": combined_manifest.get("n_prediction_spots"),
            "n_features": int(metadata_all.shape[0]),
        },
        "fit_purpose": combined_manifest.get("fit_purpose"),
        "downstream_ready": bool(combined_manifest.get("downstream_ready", False)),
        "preprocessing": first_summary.get("preprocessing"),
        "target_sum": first_summary.get("target_sum"),
        "transform": first_summary.get("transform"),
        "normalization_scope": first_summary.get("normalization_scope"),
        "coordinate_scale": first_summary.get("coordinate_scale"),
        "manifest": combined_manifest,
    }
    (args.output_dir / "vski_run_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2), encoding="utf-8"
    )
    print(
        f"Merged {metadata_all.shape[0]} fitted peaks from {args.n_shards} "
        f"validated shards into {args.output_dir}"
    )


def _alignment_signature(manifest: dict, summary: dict | None) -> dict:
    variable_manifest_keys = {
        "n_features",
        "chunks",
        "elapsed_seconds",
        "shared_setup_seconds",
        "metadata_file",
        "training_history_file",
        "pretraining_history_file",
        "hyperparameter_history_file",
        "stage_timing_file",
        "spot_file",
        "chunk_index_file",
    }
    manifest_fixed = {
        key: value
        for key, value in manifest.items()
        if key not in variable_manifest_keys
    }
    # These records combine immutable configuration/provenance with values that
    # are expected to differ between disjoint feature shards.  Compare only the
    # stable part here; the complete records remain embedded in ``shards`` in
    # the merged manifest.
    length_scale_initialization = manifest_fixed.get(
        "global_length_scale_initialization"
    )
    if isinstance(length_scale_initialization, dict):
        variable_initialization_keys = {
            "minimum",
            "median",
            "maximum",
            "n_unique",
            "n_missing_filled",
            "n_values_floored_above_global_minimum",
        }
        manifest_fixed["global_length_scale_initialization"] = {
            key: value
            for key, value in length_scale_initialization.items()
            if key not in variable_initialization_keys
        }
    geometry_cache = manifest_fixed.get("vnngp_geometry_cache")
    if isinstance(geometry_cache, dict):
        manifest_fixed["vnngp_geometry_cache"] = {
            key: value for key, value in geometry_cache.items() if key != "cache_hit"
        }
    signature = {"manifest": manifest_fixed}
    if summary is not None:
        shape = summary.get("shape", {})
        signature["summary"] = {
            "backend": summary.get("backend"),
            "inputs": summary.get("inputs"),
            "preprocessing": summary.get("preprocessing"),
            "holdouts": summary.get("holdouts"),
            "n_spots": shape.get("n_spots"),
            "n_training_spots": shape.get("n_training_spots"),
        }
    return _jsonable(signature)


def _semantic_csv_signature(path: Path, columns: tuple[str, ...] | None) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    n_rows = 0
    selected_columns: tuple[str, ...] | None = None
    for table in pd.read_csv(path, chunksize=100_000):
        if selected_columns is None:
            selected_columns = tuple(columns or table.columns.tolist())
            missing = [column for column in selected_columns if column not in table]
            if missing:
                raise ValueError(f"{path} is missing columns {missing}")
            digest.update(json.dumps(selected_columns).encode("utf-8"))
        hashed = pd.util.hash_pandas_object(
            table.loc[:, selected_columns], index=False, categorize=False
        ).to_numpy(dtype=np.uint64)
        digest.update(hashed.tobytes())
        n_rows += int(table.shape[0])
    return {
        "sha256": digest.hexdigest(),
        "n_rows": n_rows,
        "columns": list(selected_columns or ()),
    }


def _require_matching_signature(
    name: str,
    shard_id: int,
    current: dict,
    reference: dict | None,
) -> dict:
    if reference is None:
        return current
    if _canonical_json(current) != _canonical_json(reference):
        raise ValueError(f"Shard {shard_id} has mismatched {name}")
    return reference


def _require_matching_optional_signature(
    name: str,
    shard_id: int,
    current: dict | None,
    reference: dict | None,
    *,
    first_shard: bool,
) -> dict | None:
    if first_shard:
        return current
    if (current is None) != (reference is None):
        raise ValueError(f"Shard {shard_id} has inconsistent presence of {name}")
    if current is not None and _canonical_json(current) != _canonical_json(reference):
        raise ValueError(f"Shard {shard_id} has mismatched {name}")
    return reference


def _validate_shard_tables(
    metadata: pd.DataFrame, chunks: pd.DataFrame, shard_id: int
) -> None:
    metadata_required = {
        "gene",
        "feature_position",
        "map_chunk_id",
        "map_chunk_file",
        "map_column",
    }
    chunk_required = {"chunk_id", "file"}
    if missing := sorted(metadata_required - set(metadata.columns)):
        raise ValueError(f"Shard {shard_id} metadata is missing {missing}")
    if missing := sorted(chunk_required - set(chunks.columns)):
        raise ValueError(f"Shard {shard_id} chunk index is missing {missing}")
    if chunks["chunk_id"].duplicated().any():
        raise ValueError(f"Shard {shard_id} has duplicate chunk IDs")
    if metadata["gene"].duplicated().any():
        raise ValueError(f"Shard {shard_id} has duplicate feature names")


def _copy_map_chunk(
    source: Path,
    destination: Path,
    feature_positions: np.ndarray,
    *,
    link_mode: str,
    validate_only: bool = False,
) -> None:
    with np.load(source, allow_pickle=False) as archive:
        if "gene" not in archive.files or "feature_position" not in archive.files:
            raise ValueError(f"{source} lacks gene or feature_position arrays")
        if archive["gene"].shape[0] != feature_positions.size:
            raise ValueError(f"{source} feature count does not match its chunk index")
        existing = archive["feature_position"].astype(np.int64)
    rewrite_required = not np.array_equal(existing, feature_positions)
    if link_mode == "symlink":
        if rewrite_required:
            raise ValueError(
                "Unsafe symlink merge: shard-local NPZ feature positions require "
                "rewriting. Use --link-mode copy."
            )
        if validate_only:
            return
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.resolve())
        return
    if validate_only:
        return
    if not rewrite_required:
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            shutil.copy2(source, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return
    _rewrite_npz_feature_positions(source, destination, feature_positions)


def _rewrite_npz_feature_positions(
    source: Path, destination: Path, feature_positions: np.ndarray
) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with zipfile.ZipFile(source, "r") as input_zip, zipfile.ZipFile(
            temporary, "w", allowZip64=True
        ) as output_zip:
            feature_info = None
            for info in input_zip.infolist():
                if info.filename == "feature_position.npy":
                    feature_info = info
                    continue
                _copy_zip_member(input_zip, output_zip, info)
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(feature_positions, dtype=np.int64))
            replacement = _replacement_zip_info(
                feature_info, "feature_position.npy"
            )
            output_zip.writestr(replacement, buffer.getvalue())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_zip_member(
    input_zip: zipfile.ZipFile, output_zip: zipfile.ZipFile, info: zipfile.ZipInfo
) -> None:
    replacement = _replacement_zip_info(info, info.filename)
    with input_zip.open(info, "r") as source, output_zip.open(
        replacement, "w", force_zip64=True
    ) as destination:
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)


def _replacement_zip_info(
    source: zipfile.ZipInfo | None, filename: str
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(
        filename,
        date_time=source.date_time if source is not None else (1980, 1, 1, 0, 0, 0),
    )
    info.compress_type = (
        source.compress_type if source is not None else zipfile.ZIP_DEFLATED
    )
    if source is not None:
        info.external_attr = source.external_attr
        info.comment = source.comment
    return info


def _append_history(
    path: Path,
    frames: list[pd.DataFrame],
    shard_id: int,
    chunk_id_map: dict[int, int],
    feature_position_map: dict[int, int],
) -> None:
    if not path.exists():
        return
    table = pd.read_csv(path)
    table["source_shard_id"] = shard_id
    if "chunk_id" in table.columns:
        table["source_chunk_id"] = table["chunk_id"].astype(int)
        table["chunk_id"] = table["source_chunk_id"].map(chunk_id_map)
        if table["chunk_id"].isna().any():
            raise ValueError(f"{path} contains unknown chunk IDs")
        table["chunk_id"] = table["chunk_id"].astype(int)
    if "feature_position" in table.columns:
        table["source_feature_position"] = table["feature_position"].astype(int)
        table["feature_position"] = table["source_feature_position"].map(
            feature_position_map
        )
        if table["feature_position"].isna().any():
            raise ValueError(f"{path} contains unknown feature positions")
        table["feature_position"] = table["feature_position"].astype(int)
    frames.append(table)


def _write_history(frames: list[pd.DataFrame], path: Path) -> None:
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def _add_existing_history_paths(
    manifest: dict, history_frames: dict[str, list[pd.DataFrame]]
) -> None:
    mapping = {
        "vski_training_history.csv": "training_history_file",
        "vski_hyperparameter_history.csv": "hyperparameter_history_file",
        "joint_global_local_pretraining_history.csv": "pretraining_history_file",
        "joint_global_local_training_history.csv": "training_history_file",
        "joint_global_local_hyperparameter_history.csv": "hyperparameter_history_file",
        "joint_global_local_stage_timing.csv": "stage_timing_file",
    }
    for filename, manifest_key in mapping.items():
        if history_frames[filename]:
            manifest[manifest_key] = filename


def _canonical_json(value) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    main()
