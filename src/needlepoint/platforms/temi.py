"""TEMI-specific input-selection helpers for shared pipeline entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_selected_peak_indices(args: argparse.Namespace) -> list[int] | None:
    """Resolve an inline or file-backed TEMI peak-index selection."""
    inline = getattr(args, "selected_peak_indices", None)
    path = getattr(args, "selected_peak_indices_file", None)
    if inline is not None and path is not None:
        raise ValueError(
            "Use either --selected-peak-indices or --selected-peak-indices-file, not both"
        )
    if path is None:
        return inline
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
        if "peak_index" not in table.columns:
            raise ValueError(f"{path} must contain a peak_index column")
        return table["peak_index"].astype(int).tolist()
    return [
        int(token)
        for token in path.read_text(encoding="utf-8").replace(",", " ").split()
    ]


def load_prepared_input(input_dir: Path):
    """Load the historical TEMI prepared-sanity CSV contract."""
    input_dir = Path(input_dir)
    matrix_file = input_dir / "nnsvg_input_matrix.csv"
    coords_file = input_dir / "nnsvg_input_coords.csv"
    metadata_file = input_dir / "nnsvg_input_peak_metadata.csv"
    for path in (matrix_file, coords_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(path)
    peak_by_spot = pd.read_csv(matrix_file, index_col=0)
    coords_table = pd.read_csv(coords_file)
    metadata = pd.read_csv(metadata_file)
    required_coords = {"x", "y"}
    missing_coords = required_coords.difference(coords_table.columns)
    if missing_coords:
        raise ValueError(f"{coords_file} is missing columns: {sorted(missing_coords)}")
    feature_names = peak_by_spot.index.astype(str).to_numpy()
    expression = peak_by_spot.T.to_numpy(dtype=np.float32, copy=True)
    coords = coords_table[["x", "y"]].to_numpy(dtype=np.float32, copy=True)
    if expression.shape[0] != coords.shape[0]:
        raise ValueError(
            "Prepared TEMI expression and coordinate row counts differ: "
            f"{expression.shape[0]} != {coords.shape[0]}"
        )
    metadata = metadata.rename(columns={"nnsvg_peak_id": "gpytorch_feature_id"})
    summary = {"input_dir": str(input_dir), "source": "prepared_sanity_csv"}
    return expression, coords, feature_names, metadata, summary


def select_compact_spots(coords: pd.DataFrame, n_spots: int) -> np.ndarray:
    """Select the rows nearest the coordinate-wise median, in source order."""
    if n_spots <= 0:
        raise ValueError("n_spots must be positive")
    if coords.empty:
        raise ValueError("coords must not be empty")
    missing = {"x", "y"}.difference(coords.columns)
    if missing:
        raise ValueError(f"coords is missing columns: {sorted(missing)}")
    n_selected = min(int(n_spots), len(coords))
    xy = coords[["x", "y"]].to_numpy(dtype=np.float64)
    center = np.median(xy, axis=0)
    distance_squared = np.sum((xy - center) ** 2, axis=1)
    selected = np.argpartition(distance_squared, n_selected - 1)[:n_selected]
    selected.sort()
    return selected.astype(np.int64, copy=False)


def select_peak_indices(
    metadata: pd.DataFrame,
    n_peaks: int,
    min_prevalence: float,
    selected_peak_indices: list[int] | None,
) -> np.ndarray:
    """Resolve an explicit panel or rank eligible peaks by total intensity."""
    if selected_peak_indices:
        selected = np.asarray(selected_peak_indices, dtype=np.int64)
        if np.unique(selected).size != selected.size:
            raise ValueError("selected_peak_indices contains duplicates")
        return selected
    if n_peaks <= 0:
        raise ValueError("n_peaks must be positive")
    required = {"peak_index", "relative_prevalence", "peak_total_intensity"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"peak_metadata is missing columns: {sorted(missing)}")
    eligible = metadata.loc[metadata["relative_prevalence"] >= min_prevalence]
    if eligible.empty:
        raise ValueError("No peaks pass --min-prevalence")
    return (
        eligible.sort_values("peak_total_intensity", ascending=False)["peak_index"]
        .head(int(n_peaks))
        .to_numpy(dtype=np.int64)
    )
