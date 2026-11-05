"""Write a non-overwriting source/environment manifest for a pipeline stage."""

from __future__ import annotations

import argparse
from pathlib import Path

from needlepoint.config import load_toml_config
from needlepoint.provenance import (
    environment_provenance,
    source_provenance,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--package", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "source": source_provenance(args.repo_root, args.source_file),
        "environment": environment_provenance(args.package),
    }
    if args.config is not None:
        config = load_toml_config(args.config)
        manifest["configuration"] = {
            "source": str(config.source),
            "sha256": config.sha256,
            "resolved": config.values,
        }
    write_json_atomic(args.output, manifest)


if __name__ == "__main__":
    main()
