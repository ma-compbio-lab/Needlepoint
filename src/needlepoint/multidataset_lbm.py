"""Aligned peak panels and dataset-balanced V-SKI stores for joint TEMI LBM."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans

from .graph import SpatialGraphConfig, build_spatial_graph
from .vski_lbm import VSKIMapStore


@dataclass(frozen=True)
class FittedMapDataset:
    """One fitted TEMI map source participating in a common-panel analysis."""

    name: str
    vski_dir: Path
    selected_models: Path
    map_cache_dir: Path | None = None


def load_fitted_peak_table(
    vski_dir: Path,
    peak_metadata_csv: Path,
    selected_models: Path | None = None,
) -> pd.DataFrame:
    """Load fitted-map provenance and attach legacy peak-calling m/z metadata."""

    vski_dir = Path(vski_dir)
    if selected_models is None:
        table = pd.read_csv(vski_dir / "vski_gp_model_metadata.csv")
        table["model_chunk_path"] = table["map_chunk_file"].map(
            lambda value: str(vski_dir / str(value))
        )
    else:
        table = pd.read_csv(selected_models)
        if "selected" in table:
            values = table["selected"]
            mask = (
                values
                if pd.api.types.is_bool_dtype(values)
                else values.astype(str).str.lower().isin({"1", "true", "yes"})
            )
            table = table.loc[mask].copy()
    required = {"gene", "feature_position", "model_chunk_path", "map_column"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Fitted-map table lacks columns: {missing}")
    table = table.sort_values("feature_position", kind="stable").reset_index(drop=True)
    if "peak_index" not in table:
        parsed = table["gene"].astype(str).str.extract(r"(\d+)$")[0]
        if parsed.isna().any():
            raise ValueError("Cannot parse peak_index from fitted feature IDs")
        table["peak_index"] = parsed.astype(np.int64)
    metadata = pd.read_csv(peak_metadata_csv)
    if not {"peak_index", "peak_mz"}.issubset(metadata.columns):
        raise ValueError(f"{peak_metadata_csv} must contain peak_index and peak_mz")
    metadata = metadata[["peak_index", "peak_mz"]].drop_duplicates("peak_index")
    if "peak_mz" in table:
        table = table.rename(columns={"peak_mz": "fitted_peak_mz"})
    table = table.merge(metadata, on="peak_index", how="left", validate="many_to_one")
    if table["peak_mz"].isna().any():
        missing_ids = table.loc[table["peak_mz"].isna(), "peak_index"].head().tolist()
        raise ValueError(f"Legacy peak metadata is missing fitted peaks: {missing_ids}")
    table["source_feature_position"] = table["feature_position"].astype(np.int64)
    return table


def reciprocal_nearest_peak_pairs(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    tolerance_ppm: float = 5.0,
    mz_min: float = 500.0,
    mz_max: float = 1000.0,
) -> pd.DataFrame:
    """Return conservative one-to-one reciprocal-nearest legacy peak matches."""

    if tolerance_ppm <= 0.0 or mz_max <= mz_min:
        raise ValueError("Require positive tolerance_ppm and mz_max > mz_min")
    left_filtered = left.loc[left["peak_mz"].between(mz_min, mz_max)].copy()
    right_filtered = right.loc[right["peak_mz"].between(mz_min, mz_max)].copy()
    left_filtered = left_filtered.sort_values("peak_mz", kind="stable").reset_index(drop=True)
    right_filtered = right_filtered.sort_values("peak_mz", kind="stable").reset_index(drop=True)
    if left_filtered.empty or right_filtered.empty:
        raise ValueError("No fitted peaks remain in the requested m/z range")

    left_mz = left_filtered["peak_mz"].to_numpy(dtype=np.float64)
    right_mz = right_filtered["peak_mz"].to_numpy(dtype=np.float64)
    nearest_right = _nearest_sorted_indices(left_mz, right_mz)
    nearest_left = _nearest_sorted_indices(right_mz, left_mz)
    rows: list[dict] = []
    for left_row, right_row in enumerate(nearest_right):
        if nearest_left[right_row] != left_row:
            continue
        average_mz = 0.5 * (left_mz[left_row] + right_mz[right_row])
        delta_ppm = 1.0e6 * (right_mz[right_row] - left_mz[left_row]) / average_mz
        if abs(delta_ppm) > tolerance_ppm:
            continue
        left_record = left_filtered.iloc[left_row]
        right_record = right_filtered.iloc[right_row]
        pair_id = f"CP{len(rows):05d}"
        rows.append(
            {
                "canonical_peak_id": pair_id,
                "left_peak_index": int(left_record["peak_index"]),
                "left_gene": str(left_record["gene"]),
                "left_source_feature_position": int(left_record["source_feature_position"]),
                "left_peak_mz": float(left_record["peak_mz"]),
                "right_peak_index": int(right_record["peak_index"]),
                "right_gene": str(right_record["gene"]),
                "right_source_feature_position": int(right_record["source_feature_position"]),
                "right_peak_mz": float(right_record["peak_mz"]),
                "delta_mz": float(right_mz[right_row] - left_mz[left_row]),
                "delta_ppm": float(delta_ppm),
                "absolute_delta_ppm": float(abs(delta_ppm)),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No reciprocal-nearest fitted peak pairs satisfy the tolerance")
    return result


def aligned_selected_model_table(
    fitted: pd.DataFrame,
    crosswalk: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    """Subset and reorder fitted-map metadata to the canonical peak order."""

    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    index_column = f"{side}_peak_index"
    lookup = fitted.set_index("peak_index", drop=False)
    requested = crosswalk[index_column].astype(np.int64)
    missing = requested.loc[~requested.isin(lookup.index)]
    if not missing.empty:
        raise ValueError(f"Fitted table is missing matched peaks: {missing.head().tolist()}")
    output = lookup.loc[requested].reset_index(drop=True).copy()
    output["source_feature_position"] = output["feature_position"].astype(np.int64)
    output["feature_position"] = np.arange(output.shape[0], dtype=np.int64)
    output["canonical_peak_id"] = crosswalk["canonical_peak_id"].to_numpy()
    output["matched_peak_mz"] = crosswalk[f"{side}_peak_mz"].to_numpy(dtype=np.float64)
    output["matched_delta_ppm"] = crosswalk["delta_ppm"].to_numpy(dtype=np.float64)
    output["selected"] = True
    return output


def write_common_peak_panel(
    output_dir: Path,
    left: pd.DataFrame,
    right: pd.DataFrame,
    crosswalk: pd.DataFrame,
    *,
    left_name: str,
    right_name: str,
    tolerance_ppm: float,
    mz_min: float,
    mz_max: float,
) -> dict:
    """Write crosswalk, aligned selected-model tables, and provenance."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    left_selected = aligned_selected_model_table(left, crosswalk, "left")
    right_selected = aligned_selected_model_table(right, crosswalk, "right")
    crosswalk.to_csv(output_dir / "common_peak_crosswalk.csv", index=False)
    left_selected.to_csv(output_dir / f"{left_name}_common_selected_models.csv", index=False)
    right_selected.to_csv(output_dir / f"{right_name}_common_selected_models.csv", index=False)
    manifest = {
        "method": "reciprocal-nearest one-to-one matching of legacy called peaks",
        "left_name": left_name,
        "right_name": right_name,
        "tolerance_ppm": float(tolerance_ppm),
        "mz_range": [float(mz_min), float(mz_max)],
        "n_left_filtered_fitted_peaks": int(left["peak_mz"].between(mz_min, mz_max).sum()),
        "n_right_filtered_fitted_peaks": int(right["peak_mz"].between(mz_min, mz_max).sum()),
        "n_common_peak_pairs": int(crosswalk.shape[0]),
        "median_absolute_delta_ppm": float(crosswalk["absolute_delta_ppm"].median()),
        "maximum_absolute_delta_ppm": float(crosswalk["absolute_delta_ppm"].max()),
        "outputs": {
            "crosswalk": "common_peak_crosswalk.csv",
            "left_selected_models": f"{left_name}_common_selected_models.csv",
            "right_selected_models": f"{right_name}_common_selected_models.csv",
        },
        "interpretation": (
            "A common peak is a reciprocal-nearest m/z pair within tolerance. "
            "It is a correspondence between called peaks, not a metabolite identification."
        ),
    }
    (output_dir / "common_peak_panel_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


class MultiDatasetVSKIMapStore:
    """Concatenate aligned V-SKI spot maps while sharing one feature axis.

    Each child store performs its own per-feature z-scoring. Optional balanced
    observation weights make every dataset contribute equal total mass to the
    shared block sufficient statistics despite different spot counts.
    """

    def __init__(
        self,
        datasets: Sequence[FittedMapDataset],
        crosswalk_csv: Path,
        *,
        map_standardization: str = "zscore",
        use_posterior_variance: bool = True,
        balance_datasets: bool = True,
        max_spots_per_dataset: int | None = None,
        feature_block_size: int = 64,
    ) -> None:
        if len(datasets) < 2:
            raise ValueError("Joint LBM requires at least two datasets")
        self.dataset_specs = list(datasets)
        self.crosswalk_csv = Path(crosswalk_csv)
        crosswalk = pd.read_csv(self.crosswalk_csv)
        if "canonical_peak_id" not in crosswalk:
            raise ValueError("Crosswalk must contain canonical_peak_id")
        self.crosswalk = crosswalk.reset_index(drop=True)
        self.children = [
            ExistingCachedVSKIMapStore(
                spec.vski_dir,
                spec.selected_models,
                spec.map_cache_dir,
                max_spots=max_spots_per_dataset,
                use_posterior_variance=use_posterior_variance,
                feature_block_size=feature_block_size,
            )
            if spec.map_cache_dir is not None
            else VSKIMapStore(
                spec.vski_dir,
                max_spots=max_spots_per_dataset,
                selected_models_file=spec.selected_models,
                map_standardization=map_standardization,
                use_posterior_variance=use_posterior_variance,
            )
            for spec in self.dataset_specs
        ]
        expected_features = self.crosswalk.shape[0]
        if any(child.n_features != expected_features for child in self.children):
            observed = [child.n_features for child in self.children]
            raise ValueError(
                f"Aligned child feature counts {observed} do not match crosswalk {expected_features}"
            )
        for index, (spec, child) in enumerate(zip(self.dataset_specs, self.children)):
            side = "left" if index == 0 else "right" if index == 1 else None
            if side is None:
                continue
            expected_genes = self.crosswalk[f"{side}_gene"].astype(str).to_numpy()
            if not np.array_equal(child.genes.astype(str), expected_genes):
                raise ValueError(f"{spec.name} fitted-map order does not match the crosswalk")

        self.offsets = np.cumsum([0, *[child.n_spots for child in self.children]]).astype(np.int64)
        self.n_spots = int(self.offsets[-1])
        self.n_features = int(expected_features)
        self.genes = self.crosswalk["canonical_peak_id"].astype(str).to_numpy()
        self.feature_positions = np.arange(self.n_features, dtype=np.int64)
        self.metadata = self.crosswalk.copy()
        self.metadata.insert(0, "gene", self.genes)
        self.metadata.insert(1, "feature_position", self.feature_positions)
        self.map_standardization = str(map_standardization)
        self.uses_posterior_variance = bool(use_posterior_variance)
        self.balance_datasets = bool(balance_datasets)
        self.max_spots_per_dataset = max_spots_per_dataset
        self.feature_block_size = int(feature_block_size)
        self.spots = self._combined_spots()
        self.spot_weights = self._observation_weights()

    def _combined_spots(self) -> pd.DataFrame:
        frames = []
        display_offset = 0.0
        for dataset_index, (spec, child) in enumerate(zip(self.dataset_specs, self.children)):
            frame = child.spots.copy()
            frame["dataset"] = spec.name
            frame["dataset_index"] = dataset_index
            frame["dataset_map_row"] = np.arange(child.n_spots, dtype=np.int64)
            frame["native_x"] = frame["x"].to_numpy(dtype=np.float64)
            frame["native_y"] = frame["y"].to_numpy(dtype=np.float64)
            x = frame["native_x"].to_numpy(dtype=np.float64)
            y = frame["native_y"].to_numpy(dtype=np.float64)
            x_span = max(float(np.ptp(x)), 1.0)
            y_span = max(float(np.ptp(y)), 1.0)
            frame["x"] = x - x.min() + display_offset
            frame["y"] = y - y.min()
            display_offset += x_span + 0.08 * max(x_span, y_span)
            frames.append(frame)
        spots = pd.concat(frames, ignore_index=True)
        spots["joint_map_row"] = np.arange(spots.shape[0], dtype=np.int64)
        return spots

    def _observation_weights(self) -> np.ndarray:
        if not self.balance_datasets:
            return np.ones(self.n_spots, dtype=np.float64)
        target_mass = self.n_spots / len(self.children)
        return np.concatenate(
            [
                np.full(child.n_spots, target_mass / child.n_spots, dtype=np.float64)
                for child in self.children
            ]
        )

    def iter_blocks(
        self,
        *,
        include_variance: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]:
        for dataset_index, child in enumerate(self.children):
            offset = int(self.offsets[dataset_index])
            for rows, mean, variance, positions, _ in child.iter_blocks(
                include_variance=include_variance
            ):
                yield rows + offset, mean, variance, positions, self.genes[positions]

    def load_sampled_peak_maps(
        self,
        n_spots: int,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        per_dataset = max(1, int(math.ceil(n_spots / len(self.children))))
        maps = []
        rows = []
        for index, child in enumerate(self.children):
            child_maps, child_rows = child.load_sampled_peak_maps(
                min(per_dataset, child.n_spots), random_state + index
            )
            maps.append(child_maps)
            rows.append(child_rows + self.offsets[index])
        combined_maps = np.concatenate(maps, axis=1)
        combined_rows = np.concatenate(rows)
        if combined_rows.size > n_spots > 0:
            keep = np.linspace(0, combined_rows.size - 1, n_spots, dtype=np.int64)
            combined_maps = combined_maps[:, keep]
            combined_rows = combined_rows[keep]
        return combined_maps, combined_rows

    def build_block_diagonal_graph(
        self,
        n_neighbors: int,
        *,
        construction: str = "knn",
        radius: float | None = None,
        distance_bandwidth: float | None = None,
        graph_model: str | None = None,
        return_audit: bool = False,
    ):
        config = SpatialGraphConfig(
            construction=construction,
            n_neighbors=n_neighbors,
            radius=radius,
            distance_bandwidth=distance_bandwidth,
            weighted=True,
            row_normalized=False,
        )
        built = [
            build_spatial_graph(
                child.spots[["x", "y"]].to_numpy(dtype=np.float32),
                config=config,
                graph_model=graph_model,
            )
            for child in self.children
        ]
        graph = sparse.block_diag(
            [item[0] for item in built], format="csr", dtype=np.float64
        )
        if return_audit:
            return graph, [item[1] for item in built]
        return graph

    def dataset_manifest(self) -> dict:
        return {
            "datasets": [
                {
                    **{
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in asdict(spec).items()
                    },
                    "n_spots": int(child.n_spots),
                    "total_observation_weight": float(
                        self.spot_weights[self.offsets[index] : self.offsets[index + 1]].sum()
                    ),
                }
                for index, (spec, child) in enumerate(zip(self.dataset_specs, self.children))
            ],
            "n_joint_spots": self.n_spots,
            "n_common_peaks": self.n_features,
            "map_standardization": self.map_standardization,
            "dataset_balanced_observation_weights": self.balance_datasets,
            "max_spots_per_dataset": self.max_spots_per_dataset,
            "feature_block_size": self.feature_block_size,
            "cross_dataset_graph_edges": 0,
        }


def consensus_peak_initialization(
    crosswalk_csv: Path,
    left_prototype_dir: Path,
    right_prototype_dir: Path,
    *,
    n_modules: int,
    random_state: int = 0,
    smoothing: float = 0.05,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Cluster concatenated soft prototype profiles into a shared A^(0)."""

    if n_modules <= 1 or not 0.0 <= smoothing < 1.0:
        raise ValueError("Require n_modules > 1 and smoothing in [0, 1)")
    crosswalk = pd.read_csv(crosswalk_csv)
    profiles = []
    for side, directory in (("left", left_prototype_dir), ("right", right_prototype_dir)):
        assignments = pd.read_csv(Path(directory) / "prototype_assignments.csv")
        gamma = np.load(Path(directory) / "prototype_responsibilities.npy")
        lookup = pd.Series(
            np.arange(assignments.shape[0], dtype=np.int64),
            index=assignments["gene"].astype(str),
        )
        genes = crosswalk[f"{side}_gene"].astype(str)
        missing = genes.loc[~genes.isin(lookup.index)]
        if not missing.empty:
            raise ValueError(f"{side} prototype fit is missing genes: {missing.head().tolist()}")
        block = np.asarray(gamma[lookup.loc[genes].to_numpy(dtype=np.int64)], dtype=np.float64)
        block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1.0e-12)
        profiles.append(block)
    embedding = np.concatenate(profiles, axis=1)
    labels = KMeans(n_clusters=n_modules, n_init=50, random_state=random_state).fit_predict(embedding)
    responsibilities = np.full(
        (crosswalk.shape[0], n_modules), smoothing / n_modules, dtype=np.float32
    )
    responsibilities[np.arange(crosswalk.shape[0]), labels] += 1.0 - smoothing
    table = crosswalk.copy()
    table["consensus_peak_module"] = labels
    table["consensus_initial_probability"] = responsibilities.max(axis=1)
    return responsibilities, table


class ExistingCachedVSKIMapStore:
    """Read a validated existing ``CachedSpatialMapStore`` without source I/O."""

    def __init__(
        self,
        vski_dir: Path,
        selected_models: Path,
        cache_dir: Path,
        *,
        max_spots: int | None = None,
        use_posterior_variance: bool = True,
        feature_block_size: int = 64,
    ) -> None:
        self.vski_dir = Path(vski_dir)
        self.selected_models_file = Path(selected_models)
        self.cache_dir = Path(cache_dir)
        self.feature_block_size = int(feature_block_size)
        self.map_standardization = "zscore"
        spots = pd.read_csv(self.vski_dir / "vski_gp_map_spots.csv.gz")
        count = spots.shape[0] if max_spots is None else min(int(max_spots), spots.shape[0])
        if count <= 0:
            raise ValueError("max_spots must select at least one row")
        self.spot_rows = np.arange(count, dtype=np.int64)
        self.spots = spots.iloc[self.spot_rows].reset_index(drop=True).copy()
        metadata = pd.read_csv(self.selected_models_file)
        if "selected" in metadata:
            values = metadata["selected"]
            mask = (
                values
                if pd.api.types.is_bool_dtype(values)
                else values.astype(str).str.lower().isin({"1", "true", "yes"})
            )
            metadata = metadata.loc[mask].copy()
        self.metadata = metadata.sort_values("feature_position", kind="stable").reset_index(drop=True)
        self.genes = self.metadata["gene"].astype(str).to_numpy()
        self.uses_posterior_variance = bool(use_posterior_variance)
        self.mean_path = self.cache_dir / "lbm_map_mean.npy"
        self.variance_path = self.cache_dir / "lbm_map_variance.npy"
        manifest_path = self.cache_dir / "lbm_map_cache_manifest.json"
        for path in (self.mean_path, self.variance_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("map_standardization") != "zscore":
            raise ValueError(f"Expected a z-scored LBM map cache in {self.cache_dir}")
        if int(manifest.get("n_features", -1)) != self.metadata.shape[0]:
            raise ValueError("Cached feature count does not match aligned selected-model table")
        metadata_columns = [
            column
            for column in ["gene", "feature_position", "model_chunk_path", "map_column"]
            if column in self.metadata
        ]
        metadata_text = self.metadata[metadata_columns].astype(str).to_csv(index=False)
        if manifest.get("metadata_sha256") != hashlib.sha256(metadata_text.encode()).hexdigest():
            raise ValueError("Cached feature provenance does not match selected-model table")
        coordinates = np.ascontiguousarray(
            spots[["x", "y"]].to_numpy(dtype=np.float64)
        )
        if manifest.get("coordinate_sha256") != hashlib.sha256(coordinates.tobytes()).hexdigest():
            raise ValueError("Cached spot coordinates do not match the V-SKI spot table")
        self._mean = np.load(self.mean_path, mmap_mode="r")
        self._variance = np.load(self.variance_path, mmap_mode="r")
        expected_full_shape = (int(manifest["n_spots"]), self.metadata.shape[0])
        if self._mean.shape != expected_full_shape or self._variance.shape != expected_full_shape:
            raise ValueError("Cached map arrays have an unexpected shape")

    @property
    def n_spots(self) -> int:
        return int(self.spot_rows.size)

    @property
    def n_features(self) -> int:
        return int(self.metadata.shape[0])

    def iter_blocks(
        self,
        *,
        include_variance: bool = True,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]]:
        rows = np.arange(self.n_spots, dtype=np.int64)
        for start in range(0, self.n_features, self.feature_block_size):
            stop = min(start + self.feature_block_size, self.n_features)
            positions = np.arange(start, stop, dtype=np.int64)
            mean = np.asarray(self._mean[: self.n_spots, start:stop], dtype=np.float32)
            variance = (
                np.asarray(self._variance[: self.n_spots, start:stop], dtype=np.float32)
                if include_variance and self.uses_posterior_variance
                else None
            )
            yield rows, mean, variance, positions, self.genes[positions]

    def load_sampled_peak_maps(
        self,
        n_spots: int,
        random_state: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(random_state)
        count = min(max(int(n_spots), 1), self.n_spots)
        rows = np.sort(rng.choice(self.n_spots, size=count, replace=False))
        return np.asarray(self._mean[rows], dtype=np.float32).T, rows


def _nearest_sorted_indices(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if reference.size == 1:
        return np.zeros(query.size, dtype=np.int64)
    insertion = np.searchsorted(reference, query)
    insertion = np.clip(insertion, 1, reference.size - 1)
    left = insertion - 1
    right = insertion
    return np.where(
        np.abs(reference[left] - query) <= np.abs(reference[right] - query),
        left,
        right,
    ).astype(np.int64)
