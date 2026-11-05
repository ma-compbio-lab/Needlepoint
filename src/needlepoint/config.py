"""Immutable configuration loading and hashing for scientific runs."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON-compatible configuration deterministically."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def config_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ResolvedConfig:
    source: Path
    values: Mapping[str, Any]
    sha256: str


def load_toml_config(path: str | Path) -> ResolvedConfig:
    source = Path(path).resolve()
    with source.open("rb") as handle:
        values = tomllib.load(handle)
    return ResolvedConfig(source=source, values=values, sha256=config_sha256(values))
