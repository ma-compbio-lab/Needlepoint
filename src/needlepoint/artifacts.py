"""Validation and provenance helpers for fitted GP map artifacts.

The critical distinction is between an evaluation fit, where some usable
observations were held out, and a production fit, where every
likelihood-eligible location contributed to fitting. Zero-exposure sequencing
bins may still be exhaustive prediction sites. Array completeness alone is
therefore insufficient for deciding whether prototype or LBM inference is
scientifically valid.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class GPArtifactAudit:
    path: str
    status: str
    fit_purpose: str | None
    downstream_ready: bool | None
    n_training_spots: int | None
    n_prediction_spots: int | None
    training_coverage_fraction: float | None
    n_likelihood_eligible_spots: int | None
    likelihood_coverage_fraction: float | None
    full_likelihood_coverage: bool | None
    evidence: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def audit_gp_artifact(vski_dir: str | Path) -> GPArtifactAudit:
    """Classify one fitted-map directory without loading posterior arrays."""

    root = Path(vski_dir)
    manifest = _read_json(root / "vski_gp_maps_manifest.json")
    summary = _read_json(root / "vski_run_summary.json")
    evidence: list[str] = []

    fit_purpose = _first_value(manifest, summary, key="fit_purpose")
    downstream_ready = _first_bool(manifest, summary, key="downstream_ready")
    n_training = _first_int(manifest, summary, key="n_training_spots")
    n_prediction = _first_int(manifest, summary, key="n_prediction_spots")
    n_likelihood_eligible = _first_int(
        manifest, summary, key="n_likelihood_eligible_spots"
    )
    likelihood_coverage = _first_value(
        manifest, summary, key="likelihood_coverage_fraction"
    )
    full_likelihood_coverage = _first_bool(
        manifest, summary, key="full_likelihood_coverage"
    )
    if n_training is None and manifest.get("n_spots") is not None:
        n_training = int(manifest["n_spots"])
        evidence.append("legacy_manifest.n_spots_training_contract")
    if n_prediction is None:
        n_prediction = _nested_int(summary, "shape", "n_spots")
    if n_training is None:
        n_training = _nested_int(summary, "shape", "n_training_spots")

    spot_file = root / "vski_gp_map_spots.csv.gz"
    spot_training: tuple[int, int] | None = None
    spot_likelihood: tuple[int, int] | None = None
    if spot_file.is_file():
        columns = pd.read_csv(spot_file, nrows=0).columns
        if "used_for_training" in columns:
            values = pd.read_csv(spot_file, usecols=["used_for_training"])[
                "used_for_training"
            ]
            used = _coerce_bool_series(values)
            spot_training = (int(used.sum()), int(used.size))
            evidence.append("spot_table.used_for_training")
            if n_training is None:
                n_training = spot_training[0]
            if n_prediction is None:
                n_prediction = spot_training[1]
        elif n_prediction is None:
            n_prediction = sum(
                int(chunk.shape[0])
                for chunk in pd.read_csv(
                    spot_file, usecols=[columns[0]], chunksize=250_000
                )
            )
            evidence.append("spot_table.row_count")
        if "likelihood_eligible" in columns:
            values = pd.read_csv(spot_file, usecols=["likelihood_eligible"])[
                "likelihood_eligible"
            ]
            eligible = _coerce_bool_series(values)
            spot_likelihood = (int(eligible.sum()), int(eligible.size))
            evidence.append("spot_table.likelihood_eligible")
            if n_likelihood_eligible is None:
                n_likelihood_eligible = spot_likelihood[0]

    if fit_purpose is not None:
        evidence.append("manifest.fit_purpose")
    if downstream_ready is not None:
        evidence.append("manifest.downstream_ready")
    if n_training is not None and n_prediction is not None:
        evidence.append("training_prediction_counts")
    if full_likelihood_coverage is not None:
        evidence.append("manifest.full_likelihood_coverage")

    contradictions: list[str] = []
    if n_training is not None and n_prediction is not None:
        if n_training <= 0 or n_prediction <= 0 or n_training > n_prediction:
            contradictions.append("invalid training/prediction counts")
    if spot_training is not None and n_training is not None:
        if spot_training[0] != n_training or spot_training[1] != n_prediction:
            contradictions.append("spot table disagrees with manifest counts")
    if spot_likelihood is not None and n_likelihood_eligible is not None:
        if spot_likelihood[0] != n_likelihood_eligible:
            contradictions.append("spot table disagrees with likelihood-eligible count")

    full_coverage = (
        None
        if n_training is None or n_prediction is None
        else n_training == n_prediction
    )
    complete_likelihood_support = bool(
        full_likelihood_coverage is True
        and n_training is not None
        and n_likelihood_eligible is not None
        and n_training == n_likelihood_eligible
        and n_prediction is not None
        and n_likelihood_eligible <= n_prediction
    )
    if (
        fit_purpose == "production"
        and full_coverage is False
        and not complete_likelihood_support
    ):
        contradictions.append("production fit does not have full training coverage")
    if (
        downstream_ready is True
        and full_coverage is False
        and not complete_likelihood_support
    ):
        contradictions.append("downstream-ready fit does not have full training coverage")
    if fit_purpose == "evaluation" and downstream_ready is True:
        contradictions.append("evaluation fit is marked downstream-ready")

    if contradictions:
        status = "invalid"
        evidence.extend(contradictions)
    elif fit_purpose == "evaluation" or (
        full_coverage is False and not complete_likelihood_support
    ):
        status = "evaluation"
    elif (
        fit_purpose == "production"
        and (full_coverage or complete_likelihood_support)
        and downstream_ready is not False
    ):
        status = "production"
    elif full_coverage and downstream_ready is True:
        status = "production"
    elif full_coverage and fit_purpose in {None, "full_support"}:
        # Older manifests and a native full_support marker do not declare
        # production intent. Their complete training count is still positive
        # evidence, but callers may
        # choose to require an explicitly declared production artifact.
        status = "legacy_full_coverage"
    else:
        status = "unknown"

    coverage = (
        float(n_training) / float(n_prediction)
        if n_training is not None and n_prediction not in (None, 0)
        else None
    )
    return GPArtifactAudit(
        path=str(root),
        status=status,
        fit_purpose=fit_purpose,
        downstream_ready=downstream_ready,
        n_training_spots=n_training,
        n_prediction_spots=n_prediction,
        training_coverage_fraction=coverage,
        n_likelihood_eligible_spots=n_likelihood_eligible,
        likelihood_coverage_fraction=(
            None if likelihood_coverage is None else float(likelihood_coverage)
        ),
        full_likelihood_coverage=full_likelihood_coverage,
        evidence=tuple(evidence),
    )


def require_production_gp_artifacts(
    paths: Iterable[str | Path],
    *,
    allow_nonproduction: bool = False,
    allow_legacy_full_coverage: bool = False,
) -> list[GPArtifactAudit]:
    """Validate that all source map directories are production-ready."""

    audits = [audit_gp_artifact(path) for path in sorted({Path(p) for p in paths})]
    if allow_nonproduction:
        return audits
    rejected = []
    for audit in audits:
        accepted = audit.status == "production" or (
            allow_legacy_full_coverage and audit.status == "legacy_full_coverage"
        )
        if not accepted:
            rejected.append(audit)
    if rejected:
        details = "; ".join(
            f"{item.path}: status={item.status}, "
            f"training={item.n_training_spots}, prediction={item.n_prediction_spots}"
            for item in rejected
        )
        raise ValueError(
            "Prototype/LBM input must be a declared production GP refit covering "
            "every likelihood-eligible observation and every reported prediction "
            "site. Evaluation, partial-likelihood, invalid, and "
            f"unknown artifacts are rejected. {details}"
        )
    return audits


def file_fingerprint(path: str | Path, *, sha256: str | None = None) -> dict:
    """Return a cache signature that changes when a source map is replaced."""

    source = Path(path)
    stat = source.stat()
    result = {
        "path": str(source.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if sha256:
        result["sha256"] = str(sha256)
    else:
        result["content_sample_sha256"] = _sampled_file_sha256(source, stat.st_size)
    return result


def map_file_fingerprints(paths: Iterable[str | Path]) -> list[dict]:
    """Fingerprint map files, using producer-recorded SHA256 values when present."""

    result: list[dict] = []
    tables: dict[Path, dict[str, str]] = {}
    for value in sorted({Path(path) for path in paths}, key=lambda item: str(item)):
        parent = value.parent
        if parent not in tables:
            index_path = parent / "vski_gp_map_chunks.csv"
            lookup: dict[str, str] = {}
            if index_path.is_file():
                table = pd.read_csv(index_path)
                if {"file", "file_sha256"}.issubset(table.columns):
                    lookup = dict(
                        zip(
                            table["file"].astype(str).map(lambda name: Path(name).name),
                            table["file_sha256"].astype(str),
                            strict=True,
                        )
                    )
            tables[parent] = lookup
        result.append(
            file_fingerprint(value, sha256=tables[parent].get(value.name))
        )
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_savez_compressed(path: str | Path, **arrays) -> None:
    destination = Path(path)
    temporary = destination.with_name(
        f".{destination.stem}.partial-{os.getpid()}.npz"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: dict) -> None:
    destination = Path(path)
    temporary = destination.with_name(
        f".{destination.name}.partial-{os.getpid()}"
    )
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sampled_file_sha256(path: Path, size: int, block_size: int = 1024 * 1024) -> str:
    """Hash bounded head/middle/tail blocks for legacy files lacking producer hashes."""

    offsets = sorted(
        {
            0,
            max(0, int(size // 2) - block_size // 2),
            max(0, int(size) - block_size),
        }
    )
    digest = hashlib.sha256()
    digest.update(str(int(size)).encode("ascii"))
    with path.open("rb") as handle:
        for offset in offsets:
            handle.seek(offset)
            digest.update(str(offset).encode("ascii"))
            digest.update(handle.read(block_size))
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _first_value(*payloads: dict, key: str):
    for payload in payloads:
        if key in payload and payload[key] is not None:
            return payload[key]
        fit_manifest = payload.get("fit_manifest", {})
        if isinstance(fit_manifest, dict) and fit_manifest.get(key) is not None:
            return fit_manifest[key]
        manifest = payload.get("manifest", {})
        if isinstance(manifest, dict) and manifest.get(key) is not None:
            return manifest[key]
        shard_values = []
        for record in payload.get("shards", []):
            shard_manifest = record.get("manifest", {}) if isinstance(record, dict) else {}
            if shard_manifest.get(key) is not None:
                shard_values.append(shard_manifest[key])
        for record in payload.get("shard_summaries", []):
            shard_summary = record.get("summary", {}) if isinstance(record, dict) else {}
            value = _first_value(shard_summary, key=key)
            if value is not None:
                shard_values.append(value)
        if shard_values:
            first = shard_values[0]
            if any(value != first for value in shard_values[1:]):
                raise ValueError(f"Merged GP shards disagree on {key}")
            return first
    return None


def _first_bool(*payloads: dict, key: str) -> bool | None:
    value = _first_value(*payloads, key=key)
    return None if value is None else bool(value)


def _first_int(*payloads: dict, key: str) -> int | None:
    value = _first_value(*payloads, key=key)
    return None if value is None else int(value)


def _nested_int(payload: dict, parent: str, key: str) -> int | None:
    value = payload.get(parent, {}).get(key)
    return None if value is None else int(value)


def _coerce_bool_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0", "yes", "no"})
    if invalid.any():
        raise ValueError("used_for_training contains invalid boolean values")
    return normalized.isin({"true", "1", "yes"})
