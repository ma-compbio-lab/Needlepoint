"""Build a portable, self-contained package of original joint-domain maps.

The package deliberately preserves the source figures byte for byte.  In
particular, the historical uncertainty-masked figures are retained as legacy
artifacts even though their plotting code assigned colors only over the
surviving domain labels.  Consequently, their colors are not comparable with
the colors in the corresponding complete hard-domain maps.

The full soft assignment matrix and its row-aligned spot table are included so
that downstream users can regenerate maps with a fixed palette or inspect
individual domain probabilities without relying on the rendered PNGs.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import sha256_file


SOURCE_MANIFEST_NAME = "joint_vski_lbm_manifest.json"
PORTABLE_MANIFEST_NAME = "manifest_portable.json"
README_NAME = "README.md"
CHECKSUM_NAME = "SHA256SUMS.txt"

CORE_FIGURE_SUFFIXES: tuple[tuple[str, str], ...] = (
    (
        "joint_domain_map.png",
        "complete hard assignment map; domain = argmax of the saved soft probabilities",
    ),
    (
        "joint_domain_probability.png",
        "maximum saved domain probability at each spot",
    ),
    (
        "joint_domain_entropy.png",
        "normalized entropy of the saved domain-probability vector at each spot",
    ),
    (
        "joint_domain_map_uncertainty_masked.png",
        "legacy confidence-gated hard map with an unstable subset-dependent color palette",
    ),
)

CONTEXT_FIGURES: tuple[tuple[str, str], ...] = (
    (
        "joint_domain_dataset_prevalence.png",
        "hard and soft domain prevalence by section",
    ),
    (
        "joint_module_domain_heatmap.png",
        "fitted peak-module by domain signature heatmap",
    ),
)

DATA_ARTIFACTS: tuple[tuple[str, str], ...] = (
    (
        "vski_lbm_domain_probabilities.npy",
        "full row-by-domain soft assignment matrix",
    ),
    (
        "vski_lbm_spot_domains_uncertainty_aware.csv.gz",
        "row-aligned coordinates, hard assignments, uncertainty metrics, and display gate",
    ),
)

SUMMARY_ARTIFACTS: tuple[tuple[str, str], ...] = (
    (
        "joint_dataset_domain_summary.csv",
        "section-by-domain hard, soft, and balanced prevalence",
    ),
    (
        "joint_domain_uncertainty_summary.csv",
        "section-level confidence and uncertainty summary",
    ),
    (
        "joint_spatial_fragmentation_summary.csv",
        "native-grid fragmentation and boundary uncertainty summary",
    ),
    (
        "vski_lbm_domain_summary.csv",
        "global fitted-domain sizes and mixture weights",
    ),
    (
        "vski_lbm_assignment_diagnostics.json",
        "global peak- and domain-assignment diagnostics",
    ),
)


@dataclass(frozen=True)
class OriginalDomainMapPackageResult:
    """Summary of a completed package."""

    output_dir: Path
    copied_file_count: int
    checksummed_file_count: int
    total_file_count: int
    total_size_bytes: int
    manifest: dict[str, Any]


@dataclass(frozen=True)
class _ArtifactSpec:
    source_name: str
    packaged_path: str
    category: str
    semantics: str


def package_original_domain_maps(
    source_dir: str | Path,
    output_dir: str | Path,
) -> OriginalDomainMapPackageResult:
    """Copy the original joint-domain artifacts into a portable package.

    Source artifacts are never rewritten.  ``output_dir`` may be absent or an
    existing empty directory, but a nonempty destination is rejected to avoid
    silently mixing runs or overwriting an earlier package.
    """

    source = Path(source_dir).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source run directory does not exist: {source}")
    _reject_source_destination_overlap(source, destination)
    _reject_nonempty_output(destination)

    source_manifest_path = source / SOURCE_MANIFEST_NAME
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing required source manifest: {source_manifest_path}"
        )
    source_manifest = _read_json_object(source_manifest_path)
    sections = _source_sections(source_manifest)
    if len(sections) != 2:
        raise ValueError(
            "The original-domain package requires exactly two source sections; "
            f"observed {len(sections)}"
        )

    min_probability, max_entropy = _confidence_thresholds(source_manifest)
    specs = _artifact_specs([section["name"] for section in sections])
    records = _fingerprint_artifacts(source, specs)

    probability_path = source / "vski_lbm_domain_probabilities.npy"
    probability_metadata = _probability_metadata(
        probability_path,
        expected_rows=sum(int(section["n_spots"]) for section in sections),
        source_manifest=source_manifest,
    )
    section_metadata = _audit_spot_table(
        source / "vski_lbm_spot_domains_uncertainty_aware.csv.gz",
        sections,
        probability_path=probability_path,
        n_domains=int(probability_metadata["shape"][1]),
        min_probability=min_probability,
        max_entropy=max_entropy,
    )

    sanitized_source_manifest, sanitized_count = sanitize_absolute_paths(
        source_manifest,
        source_root=source,
    )
    _assert_no_absolute_paths(sanitized_source_manifest)
    source_manifest_fingerprint = _fingerprint(source_manifest_path)
    manifest = _build_portable_manifest(
        source=source,
        source_manifest=sanitized_source_manifest,
        source_manifest_fingerprint=source_manifest_fingerprint,
        sanitized_path_count=sanitized_count,
        artifact_records=records,
        probability_metadata=probability_metadata,
        section_metadata=section_metadata,
        min_probability=min_probability,
        max_entropy=max_entropy,
    )
    _assert_no_absolute_paths(manifest)

    destination.mkdir(parents=True, exist_ok=True)
    _copy_and_verify(source, destination, records)
    (destination / README_NAME).write_text(
        _render_readme(manifest), encoding="utf-8"
    )
    (destination / PORTABLE_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checksummed_count = _write_checksums(destination)

    output_files = sorted(path for path in destination.rglob("*") if path.is_file())
    return OriginalDomainMapPackageResult(
        output_dir=destination,
        copied_file_count=len(records),
        checksummed_file_count=checksummed_count,
        total_file_count=len(output_files),
        total_size_bytes=sum(path.stat().st_size for path in output_files),
        manifest=manifest,
    )


def sanitize_absolute_paths(
    value: Any,
    *,
    source_root: str | Path | None = None,
) -> tuple[Any, int]:
    """Recursively replace absolute path strings with portable references.

    Paths below ``source_root`` become ``source_run/<relative path>``.  Other
    absolute paths retain only their basename under ``external_input/``.  The
    second return value is the number of strings replaced.
    """

    root = Path(source_root).resolve() if source_root is not None else None

    def visit(item: Any) -> tuple[Any, int]:
        if isinstance(item, Mapping):
            converted: dict[Any, Any] = {}
            count = 0
            for key, child in item.items():
                sanitized, child_count = visit(child)
                converted[key] = sanitized
                count += child_count
            return converted, count
        if isinstance(item, list):
            converted_list = []
            count = 0
            for child in item:
                sanitized, child_count = visit(child)
                converted_list.append(sanitized)
                count += child_count
            return converted_list, count
        if isinstance(item, tuple):
            converted_tuple = []
            count = 0
            for child in item:
                sanitized, child_count = visit(child)
                converted_tuple.append(sanitized)
                count += child_count
            return converted_tuple, count
        if isinstance(item, str) and _is_absolute_path(item):
            return _portable_absolute_path(item, root), 1
        return item, 0

    return visit(value)


def _reject_source_destination_overlap(source: Path, destination: Path) -> None:
    try:
        destination.relative_to(source)
    except ValueError:
        return
    raise ValueError(
        "output_dir must not be the source directory or a descendant of it"
    )


def _reject_nonempty_output(destination: Path) -> None:
    if not destination.exists():
        return
    if not destination.is_dir():
        raise FileExistsError(f"Output path exists and is not a directory: {destination}")
    if next(destination.iterdir(), None) is not None:
        raise FileExistsError(f"Output directory must be empty: {destination}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _source_sections(source_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        raw_sections = source_manifest["input"]["datasets"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Source manifest must contain input.datasets"
        ) from error
    if not isinstance(raw_sections, list) or not raw_sections:
        raise ValueError("Source manifest input.datasets must be a nonempty list")
    sections: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_sections):
        if not isinstance(raw, Mapping):
            raise ValueError(f"input.datasets[{position}] must be an object")
        name = str(raw.get("name", "")).strip()
        try:
            n_spots = int(raw["n_spots"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"input.datasets[{position}].n_spots must be a positive integer"
            ) from error
        if not name or n_spots <= 0:
            raise ValueError(
                f"input.datasets[{position}] requires a name and positive n_spots"
            )
        sections.append({"name": name, "n_spots": n_spots})
    names = [section["name"] for section in sections]
    if len(set(names)) != len(names):
        raise ValueError("Source section names must be unique")
    return sections


def _confidence_thresholds(
    source_manifest: Mapping[str, Any],
) -> tuple[float, float]:
    try:
        hyperparameters = source_manifest["hyperparameters"]
        min_probability = float(hyperparameters["display_min_domain_probability"])
        max_entropy = float(hyperparameters["display_max_normalized_entropy"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Source manifest lacks finite display confidence thresholds"
        ) from error
    if not (
        math.isfinite(min_probability)
        and math.isfinite(max_entropy)
        and 0.0 <= min_probability <= 1.0
        and 0.0 <= max_entropy <= 1.0
    ):
        raise ValueError("Display confidence thresholds must lie in [0, 1]")
    return min_probability, max_entropy


def _artifact_specs(section_names: Sequence[str]) -> list[_ArtifactSpec]:
    specs: list[_ArtifactSpec] = []
    for section_name in section_names:
        for suffix, semantics in CORE_FIGURE_SUFFIXES:
            source_name = f"{section_name}_{suffix}"
            specs.append(
                _ArtifactSpec(
                    source_name=source_name,
                    packaged_path=f"figures/original/{source_name}",
                    category="original_section_figure",
                    semantics=semantics,
                )
            )
    for source_name, semantics in CONTEXT_FIGURES:
        specs.append(
            _ArtifactSpec(
                source_name=source_name,
                packaged_path=f"context/{source_name}",
                category="context_figure",
                semantics=semantics,
            )
        )
    for source_name, semantics in DATA_ARTIFACTS:
        specs.append(
            _ArtifactSpec(
                source_name=source_name,
                packaged_path=f"data/{source_name}",
                category="numeric_data",
                semantics=semantics,
            )
        )
    for source_name, semantics in SUMMARY_ARTIFACTS:
        specs.append(
            _ArtifactSpec(
                source_name=source_name,
                packaged_path=f"summaries/{source_name}",
                category="interpretation_summary",
                semantics=semantics,
            )
        )
    return specs


def _fingerprint_artifacts(
    source: Path, specs: Sequence[_ArtifactSpec]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    missing = [spec.source_name for spec in specs if not (source / spec.source_name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required original-domain artifacts:\n" + "\n".join(missing)
        )
    for spec in specs:
        path = source / spec.source_name
        record: dict[str, Any] = {
            "category": spec.category,
            "source_name": spec.source_name,
            "packaged_path": spec.packaged_path,
            "semantics": spec.semantics,
            **_fingerprint(path),
        }
        if path.suffix.lower() == ".png":
            width, height = _png_dimensions(path)
            record["image_width_pixels"] = width
            record["image_height_pixels"] = height
        records.append(record)
    return records


def _fingerprint(path: Path) -> dict[str, Any]:
    return {
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"Required figure is not a valid PNG: {path.name}")
    return struct.unpack(">II", header[16:24])


def _probability_metadata(
    path: Path,
    *,
    expected_rows: int,
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    probabilities = np.load(path, mmap_mode="r", allow_pickle=False)
    if probabilities.ndim != 2 or probabilities.shape[0] != expected_rows:
        raise ValueError(
            "vski_lbm_domain_probabilities.npy must have shape "
            f"({expected_rows}, n_domains); observed {probabilities.shape}"
        )
    if probabilities.shape[1] <= 0:
        raise ValueError("Domain-probability matrix has an empty domain axis")
    configured_domains = source_manifest.get("hyperparameters", {}).get("n_domains")
    if configured_domains is not None and int(configured_domains) != probabilities.shape[1]:
        raise ValueError(
            "Configured domain count does not match the probability matrix: "
            f"{configured_domains} != {probabilities.shape[1]}"
        )
    return {
        "shape": [int(value) for value in probabilities.shape],
        "dtype": str(probabilities.dtype),
        "row_axis": "joint_map_row order",
        "domain_axis": list(range(int(probabilities.shape[1]))),
        "individual_domain_probability_pngs_in_package": False,
    }


def _audit_spot_table(
    path: Path,
    sections: Sequence[Mapping[str, Any]],
    *,
    probability_path: Path,
    n_domains: int,
    min_probability: float,
    max_entropy: float,
) -> list[dict[str, Any]]:
    required = {
        "map_row",
        "joint_map_row",
        "dataset",
        "dataset_map_row",
        "native_x",
        "native_y",
        "vski_lbm_domain",
        "vski_lbm_domain_probability",
        "vski_lbm_domain_normalized_entropy",
        "vski_lbm_domain_confident",
        "vski_lbm_display_domain",
    }
    offsets = [0]
    for section in sections:
        offsets.append(offsets[-1] + int(section["n_spots"]))
    audits = [
        {
            "name": str(section["name"]),
            "n_spots": int(section["n_spots"]),
            "row_start_inclusive": offsets[index],
            "row_stop_exclusive": offsets[index + 1],
            "x_values": set(),
            "y_values": set(),
            "hard_domain_ids": set(),
            "confident_domain_ids": set(),
            "confident_spot_count": 0,
        }
        for index, section in enumerate(sections)
    ]
    saved_domains = np.empty(offsets[-1], dtype=np.int16)
    saved_max_probabilities = np.empty(offsets[-1], dtype=np.float32)

    row_count = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"Spot table lacks required columns: {missing}")
        section_index = 0
        for global_row, row in enumerate(reader):
            row_count = global_row + 1
            if global_row >= offsets[-1]:
                raise ValueError("Spot table has more rows than the source manifest")
            while global_row >= offsets[section_index + 1]:
                section_index += 1
            audit = audits[section_index]
            expected_local_row = global_row - int(audit["row_start_inclusive"])
            if _parse_int(row["joint_map_row"], "joint_map_row") != global_row:
                raise ValueError(
                    "Spot table joint_map_row is not contiguous joint-row order"
                )
            if row["dataset"] != audit["name"]:
                raise ValueError(
                    "Spot table dataset blocks do not match source-manifest order"
                )
            if _parse_int(row["map_row"], "map_row") != expected_local_row:
                raise ValueError("Spot table map_row is not contiguous within a section")
            if _parse_int(row["dataset_map_row"], "dataset_map_row") != expected_local_row:
                raise ValueError(
                    "Spot table dataset_map_row is not contiguous within a section"
                )
            x = _parse_finite_float(row["native_x"], "native_x")
            y = _parse_finite_float(row["native_y"], "native_y")
            domain = _parse_int(row["vski_lbm_domain"], "vski_lbm_domain")
            if not 0 <= domain < n_domains:
                raise ValueError(f"Domain label lies outside [0, {n_domains}): {domain}")
            probability = _parse_finite_float(
                row["vski_lbm_domain_probability"],
                "vski_lbm_domain_probability",
            )
            saved_domains[global_row] = domain
            saved_max_probabilities[global_row] = probability
            entropy = _parse_finite_float(
                row["vski_lbm_domain_normalized_entropy"],
                "vski_lbm_domain_normalized_entropy",
            )
            confident = _parse_bool(
                row["vski_lbm_domain_confident"],
                "vski_lbm_domain_confident",
            )
            expected_confident = (
                probability >= min_probability and entropy <= max_entropy
            )
            if confident != expected_confident:
                raise ValueError(
                    "Saved confidence flag disagrees with source-manifest thresholds"
                )
            display = row["vski_lbm_display_domain"].strip()
            if confident:
                if _parse_int(display, "vski_lbm_display_domain") != domain:
                    raise ValueError("Confident display-domain label differs from hard label")
            elif display and display.lower() != "nan":
                raise ValueError("Uncertain row has a nonempty display-domain label")
            audit["x_values"].add(x)
            audit["y_values"].add(y)
            audit["hard_domain_ids"].add(domain)
            if confident:
                audit["confident_spot_count"] += 1
                audit["confident_domain_ids"].add(domain)

    if row_count != offsets[-1]:
        raise ValueError(
            f"Spot table row count {row_count} does not match expected {offsets[-1]}"
        )
    _validate_probability_alignment(
        probability_path,
        saved_domains,
        saved_max_probabilities,
    )

    results: list[dict[str, Any]] = []
    for audit in audits:
        x_axis = _axis_metadata(audit.pop("x_values"))
        y_axis = _axis_metadata(audit.pop("y_values"))
        confident_count = int(audit["confident_spot_count"])
        n_spots = int(audit["n_spots"])
        results.append(
            {
                **audit,
                "hard_domain_ids": sorted(audit["hard_domain_ids"]),
                "confident_domain_ids": sorted(audit["confident_domain_ids"]),
                "confident_spot_fraction": confident_count / n_spots,
                "uncertain_spot_fraction": 1.0 - confident_count / n_spots,
                "native_grid": {
                    "shape_y_x": [y_axis["count"], x_axis["count"]],
                    "x": x_axis,
                    "y": y_axis,
                    "observed_spot_fraction": n_spots
                    / (x_axis["count"] * y_axis["count"]),
                },
            }
        )
    return results


def _validate_probability_alignment(
    probability_path: Path,
    saved_domains: np.ndarray,
    saved_max_probabilities: np.ndarray,
) -> None:
    """Verify that the row table and packaged soft matrix are one coherent run."""

    probabilities = np.load(probability_path, mmap_mode="r", allow_pickle=False)
    block_size = 100_000
    for start in range(0, saved_domains.size, block_size):
        stop = min(start + block_size, saved_domains.size)
        block = np.asarray(probabilities[start:stop])
        if not np.all(np.isfinite(block)) or np.any(block < 0.0):
            raise ValueError("Domain-probability matrix contains invalid values")
        if not np.allclose(block.sum(axis=1), 1.0, rtol=1.0e-5, atol=1.0e-5):
            raise ValueError("Domain-probability rows do not sum to one")
        if not np.array_equal(block.argmax(axis=1), saved_domains[start:stop]):
            raise ValueError("Hard domain labels are not argmaxes of the soft matrix")
        if not np.allclose(
            block.max(axis=1),
            saved_max_probabilities[start:stop],
            rtol=1.0e-6,
            atol=1.0e-7,
        ):
            raise ValueError(
                "Saved maximum domain probabilities do not match the soft matrix"
            )


def _axis_metadata(values: Iterable[float]) -> dict[str, Any]:
    axis = np.asarray(sorted(values), dtype=np.float64)
    if axis.size == 0:
        raise ValueError("Native coordinate axis is empty")
    if axis.size == 1:
        return {
            "count": 1,
            "minimum": float(axis[0]),
            "maximum": float(axis[0]),
            "pitch": None,
            "regular": True,
        }
    differences = np.diff(axis)
    pitch = float(np.median(differences))
    return {
        "count": int(axis.size),
        "minimum": float(axis[0]),
        "maximum": float(axis[-1]),
        "pitch": pitch,
        "regular": bool(
            np.allclose(differences, pitch, rtol=1.0e-4, atol=1.0e-4)
        ),
    }


def _parse_int(value: str, name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer; observed {value!r}") from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{name} must be an integer; observed {value!r}")
    return int(numeric)


def _parse_finite_float(value: str, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite; observed {value!r}") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite; observed {value!r}")
    return numeric


def _parse_bool(value: str, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{name} must be boolean; observed {value!r}")


def _build_portable_manifest(
    *,
    source: Path,
    source_manifest: Mapping[str, Any],
    source_manifest_fingerprint: Mapping[str, Any],
    sanitized_path_count: int,
    artifact_records: Sequence[Mapping[str, Any]],
    probability_metadata: Mapping[str, Any],
    section_metadata: Sequence[Mapping[str, Any]],
    min_probability: float,
    max_entropy: float,
) -> dict[str, Any]:
    diagnostics = source_manifest.get("diagnostics", {})
    diagnostic_metrics = diagnostics.get("metrics", {}) if isinstance(diagnostics, Mapping) else {}
    hard_recommended = (
        diagnostics.get("hard_map_interpretation_recommended")
        if isinstance(diagnostics, Mapping)
        else None
    )
    warnings = diagnostics.get("warnings", []) if isinstance(diagnostics, Mapping) else []
    return {
        "schema_version": "1.0.0",
        "package_kind": "original_joint_domain_maps",
        "portable_paths": True,
        "source_run_id": f"{source.parent.name}/{source.name}",
        "code_revision": "not_recorded_in_source_run",
        "originality": (
            "All copied artifacts are byte-for-byte copies; no fitted labels, "
            "probabilities, tables, or source PNGs were modified."
        ),
        "source_manifest": {
            "source_name": SOURCE_MANIFEST_NAME,
            **source_manifest_fingerprint,
            "absolute_path_sanitization": {
                "replacement_count": int(sanitized_path_count),
                "strategy": (
                    "source-run descendants become source_run/<relative>; other "
                    "absolute paths become external_input/<basename>"
                ),
            },
            "content": source_manifest,
        },
        "copied_artifacts": list(artifact_records),
        "copied_artifact_count": len(artifact_records),
        "copied_source_bytes": int(
            sum(int(record["size_bytes"]) for record in artifact_records)
        ),
        "probability_matrix": dict(probability_metadata),
        "row_offsets": [
            int(section["row_start_inclusive"]) for section in section_metadata
        ]
        + [int(section_metadata[-1]["row_stop_exclusive"])],
        "sections": list(section_metadata),
        "confidence_gate": {
            "definition": (
                "maximum_domain_probability >= display_min_domain_probability "
                "AND normalized_domain_entropy <= display_max_normalized_entropy"
            ),
            "display_min_domain_probability": float(min_probability),
            "display_max_normalized_entropy": float(max_entropy),
        },
        "interpretation_caveats": {
            "hard_map_interpretation_recommended_by_source_run": hard_recommended,
            "source_run_warnings": (
                list(warnings) if isinstance(warnings, list) else [str(warnings)]
            ),
            "minimum_domain_signature_symmetric_kl": diagnostic_metrics.get(
                "min_domain_signature_symmetric_kl"
            )
            if isinstance(diagnostic_metrics, Mapping)
            else None,
            "uncertainty_masked_figures": (
                "LEGACY: labels and masking are preserved, but colors were assigned "
                "over only the surviving labels. A domain color can therefore differ "
                "from the same domain in the complete hard map. Use labels or the "
                "numeric artifacts for comparison."
            ),
            "soft_map_scope": (
                "Rendered probability maps show only the maximum probability. "
                "Per-domain probabilities are available in the NPY matrix."
            ),
        },
        "generated_files": {
            "readme": README_NAME,
            "portable_manifest": PORTABLE_MANIFEST_NAME,
            "checksums": CHECKSUM_NAME,
            "checksum_scope": "all package files except SHA256SUMS.txt itself",
        },
    }


def _copy_and_verify(
    source: Path,
    destination: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    for record in records:
        source_path = source / str(record["source_name"])
        output_path = destination / str(record["packaged_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
        if output_path.stat().st_size != int(record["size_bytes"]):
            raise OSError(f"Copied size mismatch for {output_path.name}")
        if sha256_file(output_path) != record["sha256"]:
            raise OSError(f"Copied SHA256 mismatch for {output_path.name}")


def _render_readme(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Original joint-domain maps",
        "",
        "This package preserves the original domain-map outputs from "
        f"`{manifest['source_run_id']}`. The copied artifacts are byte-for-byte "
        "unchanged; `manifest_portable.json` contains their source names, byte "
        "sizes, SHA-256 hashes, axes, and path-sanitized run provenance.",
        "",
        "## Read the maps with uncertainty",
        "",
        "For each section, `*_joint_domain_map.png` is the complete hard/argmax "
        "map, `*_joint_domain_probability.png` is the maximum soft probability, "
        "and `*_joint_domain_entropy.png` is normalized entropy. The confidence "
        "gate used for `*_joint_domain_map_uncertainty_masked.png` is:",
        "",
        f"`max_probability >= {manifest['confidence_gate']['display_min_domain_probability']}` "
        "AND `normalized_entropy <= "
        f"{manifest['confidence_gate']['display_max_normalized_entropy']}`.",
        "",
        "> **Legacy palette warning:** the uncertainty-masked source figures are "
        "> preserved exactly, but their colors were assigned over only the domain "
        "> labels that survived masking. Match domains by their legend labels, not "
        "> by color against the complete hard maps.",
        "",
        "The source run did not recommend interpreting the hard map by itself. "
        "Use the probability, entropy, confidence gate, and diagnostics alongside it.",
        "",
        "## Numeric reconstruction",
        "",
        "`data/vski_lbm_domain_probabilities.npy` has rows in `joint_map_row` "
        "order and columns in the domain-axis order recorded in the portable "
        "manifest. `data/vski_lbm_spot_domains_uncertainty_aware.csv.gz` maps "
        "those rows to sections and native coordinates. There is no separate "
        "per-domain probability PNG stack; use the NPY matrix to regenerate one.",
        "",
        "Section row blocks:",
        "",
        "| Section | Rows `[start, stop)` | Spots | Native grid `(y, x)` | Confident fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for section in manifest["sections"]:
        grid = section["native_grid"]["shape_y_x"]
        lines.append(
            f"| `{section['name']}` | "
            f"[{section['row_start_inclusive']}, {section['row_stop_exclusive']}) | "
            f"{section['n_spots']} | {grid[0]} x {grid[1]} | "
            f"{section['confident_spot_fraction']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Layout",
            "",
            "- `figures/original/`: eight original section figures.",
            "- `context/`: domain prevalence and module-by-domain heatmap.",
            "- `data/`: complete soft probabilities and row-aligned spot table.",
            "- `summaries/`: small interpretation and diagnostic tables.",
            "- `SHA256SUMS.txt`: integrity hashes for every other package file.",
            "",
            "`code_revision` is explicitly `not_recorded_in_source_run`; the "
            "current checkout must not be treated as the historical run revision.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_checksums(destination: Path) -> int:
    paths = sorted(
        path
        for path in destination.rglob("*")
        if path.is_file() and path.name != CHECKSUM_NAME
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(destination).as_posix()}"
        for path in paths
    ]
    (destination / CHECKSUM_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(paths)


def _is_absolute_path(value: str) -> bool:
    return Path(value).is_absolute() or PureWindowsPath(value).is_absolute()


def _portable_absolute_path(value: str, source_root: Path | None) -> str:
    if Path(value).is_absolute():
        candidate = Path(os.path.normpath(value))
        if source_root is not None:
            try:
                relative = candidate.relative_to(source_root)
            except ValueError:
                pass
            else:
                return f"source_run/{relative.as_posix()}"
        basename = candidate.name or "root"
    else:
        windows = PureWindowsPath(value)
        basename = windows.name or "root"
    return f"external_input/{basename}"


def _assert_no_absolute_paths(value: Any, location: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_no_absolute_paths(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_absolute_paths(child, f"{location}[{index}]")
    elif isinstance(value, str) and _is_absolute_path(value):
        raise ValueError(f"Portable manifest contains an absolute path at {location}")
