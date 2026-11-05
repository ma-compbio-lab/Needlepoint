#!/usr/bin/env python3
"""Strictly merge disjoint candidate-scale screen shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from needlepoint.gp_initialization import (
    mark_uncertain_initial_scales,
    merge_initial_length_scale_screen_tables,
    scale_fraction_column,
    scale_loglik_column,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge exhaustive, nonoverlapping candidate-scale shards and "
            "recompute global best/runner-up scale statistics."
        )
    )
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-loglik-gap", type=float, default=2.0)
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "If positive, preload the merger and wait for every shard's "
            "summary-last completion marker, overlapping import time with GPU work."
        ),
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.shard_dir) < 2:
        raise ValueError("At least two candidate-scale shards are required")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output path: {args.output_dir}")
    if args.wait_timeout_seconds < 0.0 or args.poll_interval_seconds <= 0.0:
        raise ValueError("Wait timeout must be nonnegative and poll interval positive")
    if args.wait_timeout_seconds > 0.0:
        _wait_for_shards(
            args.shard_dir,
            timeout_seconds=float(args.wait_timeout_seconds),
            poll_interval_seconds=float(args.poll_interval_seconds),
        )

    summaries = [_read_summary(path) for path in args.shard_dir]
    full_scales, partitions = _validate_summaries(summaries)
    tables = [
        pd.read_csv(path / "initial_length_scale_scores.csv")
        for path in args.shard_dir
    ]
    parallel_elapsed = max(float(item["elapsed_seconds"]) for item in summaries)
    devices = sorted({str(item["screen"]["device"]) for item in summaries})
    merged = merge_initial_length_scale_screen_tables(
        tables,
        length_scales=full_scales,
        elapsed_seconds=parallel_elapsed,
        device="scale_sharded:" + ",".join(devices),
    )
    merged = mark_uncertain_initial_scales(
        merged, minimum_loglik_gap=args.minimum_loglik_gap
    )

    reference = summaries[0]
    selective_aliases = ["selective_multistart_features.csv"]
    for filename in (
        "selective_multistart_genes.csv",
        "selective_multistart_peaks.csv",
    ):
        if any((path / filename).is_file() for path in args.shard_dir):
            selective_aliases.append(filename)
    screen = dict(reference["screen"])
    screen.update(
        {
            "device": "scale_sharded:" + ",".join(devices),
            "length_scales": full_scales.tolist(),
            "length_scale_generation": partitions[0][
                "full_length_scale_generation"
            ],
            "scale_loglik_columns": [
                scale_loglik_column(float(value)) for value in full_scales
            ],
            "scale_fraction_columns": [
                scale_fraction_column(float(value)) for value in full_scales
            ],
            "n_ski_logdet_evaluations": int(
                sum(
                    int(item["screen"].get("n_ski_logdet_evaluations", 0))
                    for item in summaries
                )
            ),
            "elapsed_seconds": parallel_elapsed,
            "candidate_partition": {
                "full_length_scales": full_scales.tolist(),
                "full_length_scale_generation": partitions[0][
                    "full_length_scale_generation"
                ],
                "start_index": 0,
                "stop_index": int(full_scales.size),
                "n_full_candidates": int(full_scales.size),
                "n_shard_candidates": int(full_scales.size),
                "complete": True,
                "merged_from_partitions": partitions,
            },
        }
    )
    summary = {
        "status": "complete",
        "mode": "screen",
        "screening_likelihood": reference.get("screening_likelihood", "gaussian"),
        "downstream_observation_model": reference.get("downstream_observation_model"),
        "fit_purpose": "evaluation",
        "downstream_ready": False,
        "input": reference.get("input", reference.get("inputs")),
        "preprocessing": reference.get("preprocessing"),
        "shape": reference.get("shape"),
        "partition": reference.get("partition"),
        "screen": screen,
        "minimum_loglik_gap": float(args.minimum_loglik_gap),
        "n_recommended_for_multistart": int(
            merged["recommended_for_multistart"].sum()
        ),
        "candidate_scale_shards": [
            {
                "directory": str(path),
                "scores_sha256": _file_sha256(
                    path / "initial_length_scale_scores.csv"
                ),
                "summary_sha256": _file_sha256(
                    path / "initial_length_scale_screen_summary.json"
                ),
                "start_index": int(part["start_index"]),
                "stop_index": int(part["stop_index"]),
                "elapsed_seconds": float(item["elapsed_seconds"]),
            }
            for path, part, item in zip(args.shard_dir, partitions, summaries)
        ],
        "elapsed_seconds": parallel_elapsed,
        "elapsed_definition": (
            "Maximum shard wrapper elapsed time; merge time is negligible and "
            "should be measured externally for an exact workflow wall time."
        ),
        "outputs": {
            "scores": "initial_length_scale_scores.csv",
            "selective_multistart_features": selective_aliases,
        },
    }
    _write_atomic_output(
        args.output_dir,
        merged,
        summary,
        selective_aliases=selective_aliases,
    )
    print(
        f"Merged {len(args.shard_dir)} shards, {full_scales.size} scales, "
        f"and {merged.shape[0]} features into {args.output_dir}",
        flush=True,
    )


def _read_summary(shard_dir: Path) -> dict:
    path = shard_dir / "initial_length_scale_screen_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    scores = shard_dir / "initial_length_scale_scores.csv"
    if not scores.is_file():
        raise FileNotFoundError(scores)
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_for_shards(
    shard_dirs: list[Path], *, timeout_seconds: float, poll_interval_seconds: float
) -> None:
    deadline = time.monotonic() + float(timeout_seconds)
    pending = set(shard_dirs)
    while pending:
        complete = {
            path
            for path in pending
            if (path / "initial_length_scale_screen_summary.json").is_file()
            and (path / "initial_length_scale_scores.csv").is_file()
        }
        pending.difference_update(complete)
        if not pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Timed out waiting for candidate-scale shards: "
                + ", ".join(str(path) for path in sorted(pending))
            )
        time.sleep(min(float(poll_interval_seconds), max(deadline - time.monotonic(), 0.0)))


def _validate_summaries(summaries: list[dict]) -> tuple[np.ndarray, list[dict]]:
    partitions: list[dict] = []
    for summary in summaries:
        if "screen" not in summary:
            raise ValueError("Shard summary lacks screen provenance")
        partition = summary["screen"].get("candidate_partition")
        if not isinstance(partition, dict):
            raise ValueError("Shard summary lacks candidate_partition provenance")
        partitions.append(partition)
    full_scales = np.asarray(partitions[0]["full_length_scales"], dtype=np.float64)
    if full_scales.ndim != 1 or full_scales.size < 2:
        raise ValueError("Invalid full candidate-scale grid")

    invariant_paths = [
        ("input",),
        ("inputs",),
        ("preprocessing",),
        ("screening_likelihood",),
        ("downstream_observation_model",),
        ("shape",),
        ("partition",),
        ("screen", "backend"),
        ("screen", "spatial_fractions"),
        ("screen", "coordinate_bounds"),
        ("screen", "approximate_grid_spacing"),
    ]
    reference = summaries[0]
    for other in summaries[1:]:
        for path in invariant_paths:
            left = _nested_get(reference, path)
            right = _nested_get(other, path)
            if left is None and right is None:
                continue
            if _canonical_json(left) != _canonical_json(right):
                raise ValueError(
                    "Candidate-scale shard summaries disagree at " + ".".join(path)
                )
        left_settings = dict(reference["screen"].get("settings", {}))
        right_settings = dict(other["screen"].get("settings", {}))
        left_settings.pop("candidate_scale_offset", None)
        right_settings.pop("candidate_scale_offset", None)
        if _canonical_json(left_settings) != _canonical_json(right_settings):
            raise ValueError("Candidate-scale shard summaries disagree at screen.settings")

    intervals: list[tuple[int, int]] = []
    for summary, partition in zip(summaries, partitions):
        candidate = np.asarray(partition["full_length_scales"], dtype=np.float64)
        if not np.array_equal(candidate, full_scales):
            raise ValueError("Candidate-scale shards use different full grids")
        if _canonical_json(partition["full_length_scale_generation"]) != _canonical_json(
            partitions[0]["full_length_scale_generation"]
        ):
            raise ValueError("Candidate-scale shards use different grid generation")
        start = int(partition["start_index"])
        stop = int(partition["stop_index"])
        if not 0 <= start < stop <= full_scales.size:
            raise ValueError("Invalid candidate-scale shard interval")
        actual = np.asarray(summary["screen"]["length_scales"], dtype=np.float64)
        if not np.array_equal(actual, full_scales[start:stop]):
            raise ValueError("Shard scale profiles do not match its declared interval")
        offset = int(summary["screen"].get("settings", {}).get("candidate_scale_offset", 0))
        if offset != start:
            raise ValueError(
                "Shard candidate_scale_offset does not match its declared start_index"
            )
        intervals.append((start, stop))
    if sorted(intervals) != _contiguous_intervals(sorted(intervals), full_scales.size):
        raise ValueError("Candidate-scale shard intervals are not exhaustive and disjoint")
    return full_scales, partitions


def _contiguous_intervals(
    intervals: list[tuple[int, int]], n_candidates: int
) -> list[tuple[int, int]]:
    cursor = 0
    expected: list[tuple[int, int]] = []
    for start, stop in intervals:
        if start != cursor:
            return []
        expected.append((start, stop))
        cursor = stop
    return expected if cursor == n_candidates else []


def _nested_get(value: dict, path: tuple[str, ...]):
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _write_atomic_output(
    output_dir: Path,
    table: pd.DataFrame,
    summary: dict,
    *,
    selective_aliases: list[str],
) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(f"Refusing existing staging path: {staging}")
    staging.mkdir()
    try:
        table.to_csv(staging / "initial_length_scale_scores.csv", index=False)
        selected = table.loc[table["recommended_for_multistart"]]
        for filename in selective_aliases:
            selected.to_csv(staging / filename, index=False)
        (staging / "initial_length_scale_screen_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
