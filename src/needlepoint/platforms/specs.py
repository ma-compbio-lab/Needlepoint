"""Declarative boundaries between assay ingestion and the shared workflow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformSpec:
    """Metadata required before a dataset enters the shared H5AD contract."""

    name: str
    feature_kind: str
    native_input: str
    regular_lattice_expected: bool
    coordinate_units: str = "micron"


PLATFORM_SPECS = {
    "stereoseq": PlatformSpec(
        name="stereoseq",
        feature_kind="gene",
        native_input="GEM or H5AD",
        regular_lattice_expected=True,
    ),
    "visium_hd": PlatformSpec(
        name="visium_hd",
        feature_kind="gene",
        native_input="10x binned matrix plus tissue positions, or H5AD",
        regular_lattice_expected=True,
    ),
    "temi": PlatformSpec(
        name="temi",
        feature_kind="mass-spectral peak",
        native_input="sparse peak chunks plus coordinate table",
        regular_lattice_expected=True,
    ),
}


def get_platform_spec(name: str) -> PlatformSpec:
    """Return a registered platform after normalizing common separators."""

    key = name.strip().lower().replace("-", "_")
    try:
        return PLATFORM_SPECS[key]
    except KeyError as exc:
        choices = ", ".join(sorted(PLATFORM_SPECS))
        raise KeyError(f"Unknown platform {name!r}; choose one of: {choices}") from exc
