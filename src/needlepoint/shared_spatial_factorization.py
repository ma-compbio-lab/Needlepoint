"""Shared nonnegative spatial programs over feature-specific GP posterior maps.

The feature-specific additive GP remains the denoising model. This module lazily
reconstructs its posterior mean on the original normalized feature scale, then
fits a low-rank bank of nonnegative spatial programs. The shared programs use a
batched structured grid variational GP; the complete spot-by-feature matrix is
never placed on GPU.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from scipy.special import ndtr

from needlepoint.artifacts import (
    map_file_fingerprints,
    require_production_gp_artifacts,
)


NormalizationMode = Literal["absolute", "shape"]
FactorLink = Literal["softplus", "exp"]
NonnegativeProjection = Literal["error", "rectified_gaussian"]


@dataclass
class SharedFactorSettings:
    """Configuration for the shared spatial factorization."""

    n_factors: int = 20
    grid_size: int = 128
    kernel: str = "matern32_product"
    factor_link: FactorLink = "softplus"
    normalization_mode: NormalizationMode = "shape"
    amplitude_statistic: str = "mean"
    numerical_floor_tolerance: float = 1.0e-6
    n_epochs: int = 10
    steps_per_epoch: int = 0
    location_batch_size: int = 8192
    feature_batch_size: int = 256
    n_factor_samples: int = 1
    export_factor_samples: int = 4
    link_quadrature_points: int = 12
    nmf_sample_spots: int = 20_000
    nmf_max_iter: int = 100
    nmf_shrinkage: float = 0.0
    initialization_activity_floor: float = 1.0e-6
    initialization_neighbors: int = 1
    initialization_grid_smoothing_sigma: float = 0.0
    normalize_initial_factor_mean: bool = False
    initial_length_scale: float | None = None
    minimum_length_scale: float | None = None
    maximum_length_scale: float | None = None
    learning_rate: float = 0.01
    variational_learning_rate: float = 0.01
    hyperparameter_learning_rate: float = 0.02
    loading_l1: float = 0.0
    diversity_weight: float = 0.0
    sigma_min: float = 1.0e-4
    variance_min: float = 1.0e-6
    variance_max: float = 1.0e3
    validation_fraction: float = 0.01
    checkpoint_every: int = 1
    map_chunk_size_spots: int = 65_536
    residual_feature_chunk_size: int = 32
    dtype: str = "float32"
    device: str | None = None
    random_state: int = 0

    def validate(self) -> None:
        if self.n_factors <= 1 or self.grid_size <= 2:
            raise ValueError("n_factors must exceed one and grid_size must exceed two")
        if self.kernel not in {"matern32_product", "rbf"}:
            raise ValueError("kernel must be matern32_product or rbf")
        if self.factor_link not in {"softplus", "exp"}:
            raise ValueError("factor_link must be softplus or exp")
        if self.normalization_mode not in {"absolute", "shape"}:
            raise ValueError("normalization_mode must be absolute or shape")
        if self.amplitude_statistic not in {"mean", "sum"}:
            raise ValueError("amplitude_statistic must be mean or sum")
        positive = {
            "n_epochs": self.n_epochs,
            "location_batch_size": self.location_batch_size,
            "feature_batch_size": self.feature_batch_size,
            "n_factor_samples": self.n_factor_samples,
            "link_quadrature_points": self.link_quadrature_points,
            "nmf_sample_spots": self.nmf_sample_spots,
        }
        if any(int(value) <= 0 for value in positive.values()):
            raise ValueError(f"positive settings required: {positive}")
        if not 0.0 <= self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be in [0, 0.5)")
        if not 0.0 <= self.nmf_shrinkage < 1.0:
            raise ValueError("nmf_shrinkage must be in [0, 1)")
        if self.initialization_activity_floor <= 0.0:
            raise ValueError("initialization_activity_floor must be positive")
        if self.initialization_neighbors <= 0:
            raise ValueError("initialization_neighbors must be positive")
        if self.initialization_grid_smoothing_sigma < 0.0:
            raise ValueError("initialization_grid_smoothing_sigma must be nonnegative")
        if self.initial_length_scale is not None and self.initial_length_scale <= 0.0:
            raise ValueError("initial_length_scale must be positive")
        if self.minimum_length_scale is not None and self.minimum_length_scale <= 0.0:
            raise ValueError("minimum_length_scale must be positive")
        if self.maximum_length_scale is not None and self.maximum_length_scale <= 0.0:
            raise ValueError("maximum_length_scale must be positive")
        if (
            self.minimum_length_scale is not None
            and self.maximum_length_scale is not None
            and self.maximum_length_scale <= self.minimum_length_scale
        ):
            raise ValueError("maximum_length_scale must exceed minimum_length_scale")
        if self.variance_min <= 0.0 or self.variance_max <= self.variance_min:
            raise ValueError("invalid effective-variance bounds")


@dataclass
class SharedFactorFitResult:
    output_dir: Path
    loadings: np.ndarray
    normalized_loadings: np.ndarray
    feature_mixtures: np.ndarray
    baseline: np.ndarray
    residual_sigma: np.ndarray
    factor_map_sums: np.ndarray
    factor_length_scales: np.ndarray
    factor_output_scales: np.ndarray
    history: pd.DataFrame


class GPPosteriorFactorStore:
    """Memory-mapped nonnegative GP posterior means and variances.

    The source posterior maps are standardized. They are reconstructed as
    ``standardization_mean + standardization_std * posterior_mean``. Total
    posterior variance is rescaled by ``standardization_std**2``. Legacy
    factorized global/local exports carry ``V_global + V_local``. Opt-in frozen
    coupled exports include the signed component cross-covariance in the saved
    total, ``V_global + V_local + 2 Cov(global, local)``; this store consumes
    the declared total in either case.
    """

    CACHE_VERSION = 2

    def __init__(
        self,
        reference_vski_dir: Path,
        cache_dir: Path,
        *,
        selected_models: Path | None = None,
        normalization_mode: NormalizationMode = "shape",
        amplitude_statistic: str = "mean",
        numerical_floor_tolerance: float = 1.0e-6,
        modality: str = "temi_msi",
        normalization_method: str = "upstream_gp_preprocessing",
        max_spots: int | None = None,
        max_features: int | None = None,
        nonnegative_projection: NonnegativeProjection = "error",
        require_production_fit: bool = False,
        allow_nonproduction_fit: bool = False,
        allow_legacy_full_coverage: bool = False,
        materialize_cache: bool = False,
    ) -> None:
        self.reference_vski_dir = Path(reference_vski_dir)
        self.cache_dir = Path(cache_dir)
        self.selected_models = None if selected_models is None else Path(selected_models)
        self.normalization_mode = str(normalization_mode)
        self.amplitude_statistic = str(amplitude_statistic)
        self.numerical_floor_tolerance = float(numerical_floor_tolerance)
        self.modality = str(modality)
        self.normalization_method = str(normalization_method)
        self.nonnegative_projection = str(nonnegative_projection)
        self.require_production_fit = bool(require_production_fit)
        self.allow_nonproduction_fit = bool(allow_nonproduction_fit)
        self.allow_legacy_full_coverage = bool(allow_legacy_full_coverage)
        if self.normalization_mode not in {"absolute", "shape"}:
            raise ValueError("normalization_mode must be absolute or shape")
        if self.amplitude_statistic not in {"mean", "sum"}:
            raise ValueError("amplitude_statistic must be mean or sum")
        if self.numerical_floor_tolerance < 0.0:
            raise ValueError("numerical_floor_tolerance must be nonnegative")
        if self.nonnegative_projection not in {"error", "rectified_gaussian"}:
            raise ValueError("nonnegative_projection must be error or rectified_gaussian")

        source_spots = self._load_spots(self.reference_vski_dir)
        if max_spots is not None and max_spots <= 0:
            raise ValueError("max_spots must be positive when provided")
        if max_features is not None and max_features <= 0:
            raise ValueError("max_features must be positive when provided")
        self.source_spot_rows = spatially_balanced_sample(
            source_spots[["x", "y"]].to_numpy(dtype=np.float64),
            source_spots.shape[0] if max_spots is None else min(max_spots, source_spots.shape[0]),
        )
        self.spots = source_spots.iloc[self.source_spot_rows].reset_index(drop=True)
        self.metadata = self._load_metadata()
        source_dirs = {Path(path).parent for path in self.metadata["model_chunk_path"]}
        self.gp_artifact_audits = require_production_gp_artifacts(
            source_dirs,
            allow_nonproduction=(
                self.allow_nonproduction_fit or not self.require_production_fit
            ),
            allow_legacy_full_coverage=self.allow_legacy_full_coverage,
        )
        if max_features is not None:
            self.metadata = self.metadata.iloc[: min(max_features, self.metadata.shape[0])].copy()
            self.metadata["source_feature_position"] = self.metadata.get(
                "source_feature_position", self.metadata["feature_position"]
            )
            self.metadata["feature_position"] = np.arange(
                self.metadata.shape[0], dtype=np.int64
            )
        self.n_spots = int(self.spots.shape[0])
        self.n_features = int(self.metadata.shape[0])
        if self.n_features == 0:
            raise ValueError("No GP features were selected")
        self.feature_ids = self.metadata["gene"].astype(str).to_numpy()
        self._validate_source_spots()

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mean_path = self.cache_dir / "factor_input_mean.npy"
        self.variance_path = self.cache_dir / "factor_input_variance.npy"
        self.manifest_path = self.cache_dir / "factor_input_manifest.json"
        self.amplitude_path = self.cache_dir / "feature_amplitudes.csv"
        self._mean = None
        self._variance = None
        self.amplitudes = np.ones(self.n_features, dtype=np.float64)
        self.validation_report: dict = {}
        self.materialize_cache = bool(materialize_cache)
        self._streaming = None

    def prepare(self, *, rebuild: bool = False) -> dict:
        if not self.materialize_cache:
            from needlepoint.streaming_factor_input import StreamingFactorInput
            self._streaming = StreamingFactorInput(self)
            return self._streaming.prepare(rebuild=rebuild)
        signature = self._source_signature()
        if not rebuild and self._cache_is_compatible(signature):
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            amplitudes = pd.read_csv(self.amplitude_path)
            if not np.array_equal(amplitudes["gene"].astype(str), self.feature_ids):
                raise ValueError("Cached feature-amplitude order does not match source")
            self.amplitudes = amplitudes["feature_amplitude"].to_numpy(dtype=np.float64)
            self.validation_report = manifest["validation"]
            self._open_cache()
            return manifest

        started = time.time()
        token = f"partial-{os.getpid()}"
        mean_temporary = self.cache_dir / f".{self.mean_path.stem}.{token}.npy"
        variance_temporary = self.cache_dir / f".{self.variance_path.stem}.{token}.npy"
        amplitude_temporary = self.cache_dir / f".{self.amplitude_path.name}.{token}"
        mean = np.lib.format.open_memmap(
            mean_temporary,
            mode="w+",
            dtype=np.float32,
            shape=(self.n_spots, self.n_features),
        )
        variance = np.lib.format.open_memmap(
            variance_temporary,
            mode="w+",
            dtype=np.float32,
            shape=(self.n_spots, self.n_features),
        )
        feature_sums = np.zeros(self.n_features, dtype=np.float64)
        feature_min = np.full(self.n_features, np.inf, dtype=np.float64)
        negative_counts = np.zeros(self.n_features, dtype=np.int64)
        most_negative = 0.0
        tiny_negative_total = 0

        for source_path, rows in self.metadata.groupby("model_chunk_path", sort=False):
            positions = rows.index.to_numpy(dtype=np.int64)
            columns = rows["map_column"].to_numpy(dtype=np.int64)
            with np.load(Path(source_path), allow_pickle=True) as npz:
                standardized_mean = np.asarray(
                    npz["posterior_mean"][self.source_spot_rows][:, columns],
                    dtype=np.float32,
                )
                standardized_variance = np.asarray(
                    npz["posterior_variance"][self.source_spot_rows][:, columns],
                    dtype=np.float32,
                )
            location = rows["standardization_mean"].to_numpy(dtype=np.float64)
            scale = rows["standardization_std"].to_numpy(dtype=np.float64)
            reconstructed = standardized_mean.astype(np.float64) * scale[None, :] + location[None, :]
            reconstructed_variance = (
                standardized_variance.astype(np.float64) * np.square(scale)[None, :]
            )
            if not np.isfinite(reconstructed).all() or not np.isfinite(reconstructed_variance).all():
                raise ValueError(f"Non-finite GP posterior values in {source_path}")
            chunk_min = reconstructed.min(axis=0)
            feature_min[positions] = chunk_min
            negative = reconstructed < 0.0
            counts = negative.sum(axis=0, dtype=np.int64)
            negative_counts[positions] = counts
            if negative.any():
                most_negative = min(most_negative, float(reconstructed[negative].min()))
                materially_negative = (
                    float(reconstructed[negative].min()) < -self.numerical_floor_tolerance
                )
                if materially_negative and self.nonnegative_projection == "error":
                    raise ValueError(
                        "Materially negative reconstructed GP values detected: "
                        f"minimum={float(reconstructed[negative].min()):.6g}, "
                        f"tolerance={self.numerical_floor_tolerance:.6g}, source={source_path}"
                    )
                if not materially_negative:
                    tiny_negative_total += int(negative.sum())
            reconstructed, reconstructed_variance = project_posterior_nonnegative(
                reconstructed, reconstructed_variance, rows, self.nonnegative_projection,
                self.numerical_floor_tolerance)
            if (reconstructed_variance < 0.0).any():
                raise ValueError(f"Negative GP posterior variance in {source_path}")
            mean[:, positions] = reconstructed.astype(np.float32)
            variance[:, positions] = reconstructed_variance.astype(np.float32)
            feature_sums[positions] = reconstructed.sum(axis=0, dtype=np.float64)

        if self.normalization_mode == "shape":
            denominator = (
                feature_sums
                if self.amplitude_statistic == "sum"
                else feature_sums / max(self.n_spots, 1)
            )
            if (denominator <= 0.0).any():
                genes = self.feature_ids[denominator <= 0.0][:5]
                raise ValueError(f"Non-positive feature amplitudes, e.g. {genes.tolist()}")
            self.amplitudes = denominator
            for start in range(0, self.n_features, 64):
                stop = min(start + 64, self.n_features)
                amplitude = denominator[start:stop]
                mean[:, start:stop] /= amplitude[None, :]
                variance[:, start:stop] /= np.square(amplitude)[None, :]
        else:
            self.amplitudes = np.ones(self.n_features, dtype=np.float64)

        mean.flush()
        variance.flush()
        amplitude_table = pd.DataFrame(
            {
                "gene": self.feature_ids,
                "feature_position": np.arange(self.n_features, dtype=np.int64),
                "feature_amplitude": self.amplitudes,
                "reconstructed_minimum": feature_min,
                "negative_entry_count_before_floor": negative_counts,
            }
        )
        amplitude_table.to_csv(amplitude_temporary, index=False)
        self.validation_report = {
            "n_spots": self.n_spots,
            "n_features": self.n_features,
            "finite_values": True,
            "posterior_variance_nonnegative": True,
            "negative_entry_count_before_floor": int(negative_counts.sum()),
            "features_with_negative_entries": int(np.sum(negative_counts > 0)),
            "tiny_negative_entries_floored": int(tiny_negative_total),
            "nonnegative_projection": self.nonnegative_projection,
            "most_negative_entry": float(most_negative),
            "numerical_floor_tolerance": self.numerical_floor_tolerance,
            "total_variance_assumption": (
                "producer-declared total moments on the metadata field scale; "
                "this reader never recombines component variances"
            ),
        }
        manifest = {
            "cache_version": self.CACHE_VERSION,
            "source_signature": signature,
            "reference_vski_dir": str(self.reference_vski_dir),
            "selected_models": None if self.selected_models is None else str(self.selected_models),
            "shape": [self.n_spots, self.n_features],
            "normalization_mode": self.normalization_mode,
            "amplitude_statistic": self.amplitude_statistic,
            "nonnegative_projection": self.nonnegative_projection,
            "mean_file": self.mean_path.name,
            "variance_file": self.variance_path.name,
            "amplitude_file": self.amplitude_path.name,
            "validation": self.validation_report,
            "gp_artifact_audits": [
                audit.to_dict() for audit in self.gp_artifact_audits
            ],
            "elapsed_seconds": float(time.time() - started),
        }
        manifest_temporary = self.cache_dir / f".{self.manifest_path.name}.{token}"
        manifest_temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if self.manifest_path.exists():
            os.replace(
                self.manifest_path,
                self.manifest_path.with_name(self.manifest_path.name + ".previous"),
            )
        os.replace(mean_temporary, self.mean_path)
        os.replace(variance_temporary, self.variance_path)
        os.replace(amplitude_temporary, self.amplitude_path)
        os.replace(manifest_temporary, self.manifest_path)
        self._mean = np.load(self.mean_path, mmap_mode="r")
        self._variance = np.load(self.variance_path, mmap_mode="r")
        return manifest

    def read_block(
        self, spot_rows: np.ndarray, feature_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.materialize_cache:
            if self._streaming is None:
                self.prepare()
            return self._streaming.read_block(spot_rows, feature_positions)
        if self._mean is None or self._variance is None:
            self._open_cache()
        rows = np.asarray(spot_rows, dtype=np.int64)
        features = np.asarray(feature_positions, dtype=np.int64)
        return (
            np.asarray(self._mean[np.ix_(rows, features)], dtype=np.float32),
            np.asarray(self._variance[np.ix_(rows, features)], dtype=np.float32),
        )

    def read_all_features(self, spot_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.read_block(spot_rows, np.arange(self.n_features, dtype=np.int64))

    def _open_cache(self) -> None:
        if not self.mean_path.is_file() or not self.variance_path.is_file():
            raise FileNotFoundError("Factor input cache has not been prepared")
        self._mean = np.load(self.mean_path, mmap_mode="r")
        self._variance = np.load(self.variance_path, mmap_mode="r")
        expected = (self.n_spots, self.n_features)
        if self._mean.shape != expected or self._variance.shape != expected:
            raise ValueError("Cached factor input shape does not match source")

    def _load_metadata(self) -> pd.DataFrame:
        if self.selected_models is None:
            metadata_path = self.reference_vski_dir / "vski_gp_model_metadata.csv"
            metadata = pd.read_csv(metadata_path)
            metadata["model_chunk_path"] = metadata["map_chunk_file"].map(
                lambda value: str(self.reference_vski_dir / str(value))
            )
        else:
            metadata = pd.read_csv(self.selected_models)
            if "selected" in metadata:
                values = metadata["selected"]
                selected = (
                    values
                    if pd.api.types.is_bool_dtype(values)
                    else values.astype(str).str.lower().isin({"1", "true", "yes"})
                )
                metadata = metadata.loc[selected].copy()
            if "model_chunk_path" not in metadata:
                raise ValueError("Selected-model table lacks model_chunk_path")
            metadata = self._enrich_selected_metadata(metadata)
        required = {
            "gene",
            "feature_position",
            "model_chunk_path",
            "map_column",
            "standardization_mean",
            "standardization_std",
        }
        missing = sorted(required.difference(metadata.columns))
        if missing:
            raise ValueError(f"GP metadata lacks columns: {missing}")
        metadata = metadata.sort_values("feature_position", kind="mergesort").reset_index(drop=True)
        if metadata["gene"].astype(str).duplicated().any():
            raise ValueError("Feature IDs are not unique")
        if not np.array_equal(
            metadata["feature_position"].to_numpy(dtype=np.int64),
            np.arange(metadata.shape[0], dtype=np.int64),
        ):
            metadata["source_feature_position"] = metadata["feature_position"]
            metadata["feature_position"] = np.arange(metadata.shape[0], dtype=np.int64)
        metadata["model_chunk_path"] = metadata["model_chunk_path"].map(str)
        missing_paths = [path for path in metadata["model_chunk_path"].unique() if not Path(path).is_file()]
        if missing_paths:
            raise FileNotFoundError(missing_paths[0])
        metadata["modality"] = metadata.get("modality", self.modality)
        metadata["normalization_method"] = metadata.get(
            "normalization_method", self.normalization_method
        )
        return metadata

    def _enrich_selected_metadata(self, selected: pd.DataFrame) -> pd.DataFrame:
        needed = {
            "standardization_mean",
            "standardization_std",
            "joint_global_length_scale",
            "joint_local_length_scale",
            "joint_global_outputscale",
            "joint_local_outputscale",
            "joint_noise_variance",
        }
        missing = [column for column in needed if column not in selected]
        if not missing:
            return selected
        source_dirs = sorted({Path(path).parent for path in selected["model_chunk_path"].astype(str)})
        source_tables = []
        for source_dir in source_dirs:
            table = pd.read_csv(source_dir / "vski_gp_model_metadata.csv")
            table["source_map_dir"] = str(source_dir)
            source_tables.append(table)
        sources = pd.concat(source_tables, ignore_index=True)
        selected = selected.copy()
        selected["source_map_dir"] = selected["model_chunk_path"].map(lambda value: str(Path(value).parent))
        columns = ["source_map_dir", "gene", *sorted(needed)]
        columns = [column for column in columns if column in sources]
        selected = selected.merge(
            sources[columns].drop_duplicates(["source_map_dir", "gene"]),
            on=["source_map_dir", "gene"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_source"),
        )
        for column in needed:
            source_column = f"{column}_source"
            if source_column in selected:
                selected[column] = selected.get(column, np.nan)
                selected[column] = selected[column].fillna(selected[source_column])
                selected = selected.drop(columns=source_column)
        return selected.drop(columns="source_map_dir")

    @staticmethod
    def _load_spots(vski_dir: Path) -> pd.DataFrame:
        path = Path(vski_dir) / "vski_gp_map_spots.csv.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
        spots = pd.read_csv(path)
        required = {"x", "y"}
        if not required.issubset(spots.columns):
            raise ValueError(f"Spot table lacks {sorted(required.difference(spots.columns))}")
        if not np.isfinite(spots[["x", "y"]].to_numpy(dtype=np.float64)).all():
            raise ValueError("Spot coordinates contain non-finite values")
        return spots.reset_index(drop=True)

    def _validate_source_spots(self) -> None:
        reference = self.spots[[column for column in ["spot_index", "x", "y"] if column in self.spots]]
        source_dirs = sorted({Path(path).parent for path in self.metadata["model_chunk_path"]})
        for source_dir in source_dirs:
            if source_dir.resolve() == self.reference_vski_dir.resolve():
                continue
            other = self._load_spots(source_dir).iloc[self.source_spot_rows].reset_index(drop=True)
            columns = list(reference.columns)
            if other.shape[0] != reference.shape[0] or not np.array_equal(
                other[columns].to_numpy(), reference[columns].to_numpy()
            ):
                raise ValueError(f"Spot ordering differs for selected source {source_dir}")

    def _source_signature(self) -> str:
        columns = [
            "gene",
            "model_chunk_path",
            "map_column",
            "standardization_mean",
            "standardization_std",
        ]
        payload = {
            "cache_version": self.CACHE_VERSION,
            "features": self.metadata[columns].to_dict(orient="records"),
            "field_contract": self.metadata[[c for c in ('posterior_nonnegative',
                'posterior_map_scale', 'field_units', 'observation_model') if c in self.metadata]].to_dict(orient='records'),
            "spot_coordinates_sha256": hashlib.sha256(
                np.ascontiguousarray(
                    self.spots[["x", "y"]].to_numpy(dtype=np.float64)
                ).view(np.uint8)
            ).hexdigest(),
            "source_spot_rows_sha256": hashlib.sha256(
                np.ascontiguousarray(self.source_spot_rows).view(np.uint8)
            ).hexdigest(),
            "normalization_mode": self.normalization_mode,
            "amplitude_statistic": self.amplitude_statistic,
            "numerical_floor_tolerance": self.numerical_floor_tolerance,
            "nonnegative_projection": self.nonnegative_projection,
            "source_map_files": map_file_fingerprints(
                self.metadata["model_chunk_path"].astype(str).unique()
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _cache_is_compatible(self, signature: str) -> bool:
        if not all(path.is_file() for path in [self.mean_path, self.variance_path, self.manifest_path, self.amplitude_path]):
            return False
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return (
            manifest.get("cache_version") == self.CACHE_VERSION
            and manifest.get("storage_mode") != "streaming-v1"
            and manifest.get("source_signature") == signature
            and tuple(manifest.get("shape", [])) == (self.n_spots, self.n_features)
        )


def spatially_balanced_sample(coords: np.ndarray, n_samples: int) -> np.ndarray:
    """Select deterministic, spatially balanced rows using a regular tile grid."""
    coords = np.asarray(coords, dtype=np.float64)
    n_spots = coords.shape[0]
    if n_samples <= 0 or n_samples >= n_spots:
        return np.arange(n_spots, dtype=np.int64)
    n_side = max(1, int(math.ceil(math.sqrt(n_samples))))
    bounds_min = coords.min(axis=0)
    spans = np.maximum(coords.max(axis=0) - bounds_min, 1.0e-12)
    bins = np.floor((coords - bounds_min) / spans * n_side).astype(np.int64)
    bins = np.clip(bins, 0, n_side - 1)
    tile = bins[:, 0] * n_side + bins[:, 1]
    centers = bounds_min + (bins + 0.5) / n_side * spans
    distance = np.square(coords - centers).sum(axis=1)
    order = np.lexsort((distance, tile))
    _, first = np.unique(tile[order], return_index=True)
    selected = order[first]
    if selected.size < n_samples:
        remaining = np.setdiff1d(np.arange(n_spots), selected, assume_unique=False)
        stride = max(1, int(math.floor(remaining.size / max(n_samples - selected.size, 1))))
        selected = np.concatenate([selected, remaining[::stride][: n_samples - selected.size]])
    return np.sort(selected[:n_samples].astype(np.int64))


def project_posterior_nonnegative(mean, variance, metadata, mode, tolerance):
    """Preserve declared positive-field moments; rectify only Gaussian fields."""
    positive = (metadata['posterior_nonnegative'].astype(str).str.lower().isin(['true', '1']).to_numpy()
                if 'posterior_nonnegative' in metadata else np.zeros(mean.shape[1], dtype=bool))
    if positive.any() and (mean[:, positive] < -tolerance).any():
        raise ValueError('Declared nonnegative posterior has negative means')
    mean = mean.copy()
    variance = variance.copy()
    if mode == 'rectified_gaussian' and (~positive).any():
        mean[:, ~positive], variance[:, ~positive] = _rectified_gaussian_moments(
            mean[:, ~positive], variance[:, ~positive])
    mean = np.maximum(mean, 0)
    return mean, variance


def _rectified_gaussian_moments(
    mean: np.ndarray, variance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Moments of max(Z, 0) for Z following an elementwise Gaussian posterior."""
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.maximum(np.asarray(variance, dtype=np.float64), 0.0)
    standard_deviation = np.sqrt(variance)
    positive_sd = standard_deviation > 1.0e-12
    alpha = np.zeros_like(mean)
    alpha[positive_sd] = mean[positive_sd] / standard_deviation[positive_sd]
    cdf = ndtr(alpha)
    density = np.exp(-0.5 * np.square(alpha)) / math.sqrt(2.0 * math.pi)
    projected_mean = mean * cdf + standard_deviation * density
    second_moment = (
        (np.square(mean) + variance) * cdf
        + mean * standard_deviation * density
    )
    projected_variance = np.maximum(second_moment - np.square(projected_mean), 0.0)
    if np.any(~positive_sd):
        projected_mean[~positive_sd] = np.maximum(mean[~positive_sd], 0.0)
        projected_variance[~positive_sd] = 0.0
    return projected_mean, projected_variance


def normalize_factor_identifiability(
    factor_maps: np.ndarray, loadings: np.ndarray, epsilon: float = 1.0e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize factor-map columns and rescale loadings without changing Lambda."""
    factor_maps = np.asarray(factor_maps, dtype=np.float64)
    loadings = np.asarray(loadings, dtype=np.float64)
    sums = factor_maps.sum(axis=0) + float(epsilon)
    normalized_maps = factor_maps / sums[None, :]
    normalized_loadings = loadings * sums[None, :]
    mixtures = normalized_loadings / (
        normalized_loadings.sum(axis=1, keepdims=True) + float(epsilon)
    )
    return normalized_maps, normalized_loadings, mixtures


def scaled_gaussian_log_likelihood(
    target,
    reconstruction,
    effective_variance,
    *,
    n_spots_total: int,
    n_features_total: int,
    reconstruction_variance=0.0,
):
    """Unbiased expected Gaussian log likelihood for a Cartesian minibatch."""
    expected_square_error = (
        (target - reconstruction).square() + reconstruction_variance
    )
    log_likelihood = -0.5 * (
        torch_log_two_pi(target)
        + effective_variance.log()
        + expected_square_error / effective_variance
    )
    scale = (float(n_spots_total) / target.shape[-2]) * (
        float(n_features_total) / target.shape[-1]
    )
    return log_likelihood.sum(dim=(-2, -1)) * scale


def torch_log_two_pi(reference):
    return reference.new_tensor(math.log(2.0 * math.pi))


def _require_torch_gpytorch():
    try:
        import gpytorch
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("shared spatial factorization requires torch and gpytorch") from exc
    return torch, gpytorch


def _build_shared_grid_gp(
    *,
    n_factors: int,
    grid_size: int,
    grid_bounds,
    kernel_name: str,
    min_length_scale: float,
    max_length_scale: float,
    prior_jitter: float = 1.0e-3,
):
    torch, gpytorch = _require_torch_gpytorch()
    batch_shape = torch.Size([int(n_factors)])
    n_inducing = int(grid_size) ** len(grid_bounds)
    variational_dist = gpytorch.variational.MeanFieldVariationalDistribution(
        n_inducing,
        batch_shape=batch_shape,
    )

    class ProductMatern32Kernel(gpytorch.kernels.Kernel):
        has_lengthscale = True

        def forward(self, x1, x2, diag=False, **params):
            sqrt_three = math.sqrt(3.0)
            if diag:
                differences = (x1 - x2).abs()
                scaled = differences.unsqueeze(0) / self.lengthscale
                return ((1.0 + sqrt_three * scaled) * torch.exp(-sqrt_three * scaled)).prod(dim=-1)
            differences = (x1[:, None, :] - x2[None, :, :]).abs().unsqueeze(0)
            scaled = differences / self.lengthscale.unsqueeze(-2)
            return ((1.0 + sqrt_three * scaled) * torch.exp(-sqrt_three * scaled)).prod(dim=-1)

    class StructuredSharedStrategy(gpytorch.variational.GridInterpolationVariationalStrategy):
        def kl_divergence(self):
            variational = self._variational_distribution
            return _structured_product_grid_kl(
                torch=torch,
                variational_mean=variational.variational_mean,
                variational_variance=variational.variational_stddev.square(),
                grid=self.grid,
                lengthscale=self.model.covar_module.base_kernel.lengthscale,
                outputscale=self.model.covar_module.outputscale,
                jitter=float(prior_jitter),
                kernel_name=kernel_name,
            )

    class SharedGridGP(gpytorch.models.ApproximateGP):
        def __init__(self):
            strategy = StructuredSharedStrategy(
                self,
                grid_size=int(grid_size),
                grid_bounds=grid_bounds,
                variational_distribution=variational_dist,
            )
            super().__init__(strategy)
            self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
            constraint = gpytorch.constraints.Interval(
                float(min_length_scale), float(max_length_scale)
            )
            if kernel_name == "matern32_product":
                base = ProductMatern32Kernel(
                    batch_shape=batch_shape,
                    lengthscale_constraint=constraint,
                )
            elif kernel_name == "rbf":
                base = gpytorch.kernels.RBFKernel(
                    batch_shape=batch_shape,
                    lengthscale_constraint=constraint,
                )
            else:
                raise ValueError(kernel_name)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                base,
                batch_shape=batch_shape,
                outputscale_constraint=gpytorch.constraints.GreaterThan(1.0e-6),
            )

        def forward(self, x):
            return gpytorch.distributions.MultivariateNormal(
                self.mean_module(x), self.covar_module(x)
            )

    return SharedGridGP()


def _structured_product_grid_kl(
    *,
    torch,
    variational_mean,
    variational_variance,
    grid,
    lengthscale,
    outputscale,
    jitter: float,
    kernel_name: str,
):
    """Exact mean-field KL for separable RBF or product Matérn-3/2 grids."""
    n_factors, n_inducing = variational_mean.shape
    grid_size = int(grid.shape[0])
    if grid.ndim != 2 or grid.shape[1] != 2 or n_inducing != grid_size**2:
        raise ValueError("structured shared-factor KL requires a two-dimensional Cartesian grid")
    output_dtype = variational_mean.dtype
    work_dtype = torch.float64 if output_dtype == torch.float32 else output_dtype
    grid = grid.to(dtype=work_dtype)
    lengthscale = lengthscale.reshape(n_factors, -1).to(dtype=work_dtype)
    if lengthscale.shape[1] != 1:
        lengthscale = lengthscale.mean(dim=1, keepdim=True)
    outputscale = outputscale.reshape(n_factors).to(dtype=work_dtype)
    mean = variational_mean.to(dtype=work_dtype)
    variance = variational_variance.to(dtype=work_dtype).clamp_min(1.0e-12)
    factors = []
    for dimension in range(2):
        distance = (grid[:, dimension, None] - grid[None, :, dimension]).abs()
        scaled = distance[None, :, :] / lengthscale[:, :, None]
        if kernel_name == "matern32_product":
            root = math.sqrt(3.0) * scaled
            factor = (1.0 + root) * torch.exp(-root)
        elif kernel_name == "rbf":
            factor = torch.exp(-0.5 * scaled.square())
        else:
            raise ValueError(kernel_name)
        factors.append(factor)
    # GridInterpolationVariationalStrategy defines its prior as
    # outputscale * kron(Ky, Kx) + jitter * I. Diagonalizing the one-dimensional
    # factors keeps this exact without materializing the full grid covariance.
    kx = 0.5 * (factors[0] + factors[0].transpose(-1, -2))
    ky = 0.5 * (factors[1] + factors[1].transpose(-1, -2))
    eigenvalues_x, eigenvectors_x = torch.linalg.eigh(kx)
    eigenvalues_y, eigenvectors_y = torch.linalg.eigh(ky)
    prior_eigenvalues = (
        outputscale[:, None, None]
        * eigenvalues_y[:, :, None]
        * eigenvalues_x[:, None, :]
        + float(jitter)
    ).clamp_min(1.0e-12)
    mean_grid = mean.reshape(n_factors, grid_size, grid_size)
    rotated_mean = torch.matmul(
        eigenvectors_y.transpose(-1, -2),
        torch.matmul(mean_grid, eigenvectors_x),
    )
    mean_quad = (rotated_mean.square() / prior_eigenvalues).sum(dim=(-2, -1))
    inverse_eigenvalues = prior_eigenvalues.reciprocal()
    inverse_diagonal = torch.matmul(
        eigenvectors_y.square(),
        torch.matmul(inverse_eigenvalues, eigenvectors_x.square().transpose(-1, -2)),
    )
    trace = (
        variance.reshape(n_factors, grid_size, grid_size) * inverse_diagonal
    ).sum(dim=(-2, -1))
    prior_logdet = prior_eigenvalues.log().sum(dim=(-2, -1))
    q_logdet = variance.log().sum(dim=-1)
    kl = 0.5 * (trace + mean_quad - n_inducing + prior_logdet - q_logdet)
    return kl.to(dtype=output_dtype)


def _stable_batched_cholesky(torch, matrix):
    """Cholesky with bounded diagonal escalation for backend roundoff."""
    try:
        return torch.linalg.cholesky(matrix)
    except RuntimeError:
        factors = []
        identity = torch.eye(
            matrix.shape[-1], dtype=matrix.dtype, device=matrix.device
        )
        for factor_matrix in matrix:
            last_error = None
            for extra_jitter in (0.0, 1.0e-8, 1.0e-6, 1.0e-4):
                try:
                    factor = torch.linalg.cholesky(
                        factor_matrix + extra_jitter * identity
                    )
                    factors.append(factor)
                    break
                except RuntimeError as error:  # pragma: no cover - backend specific
                    last_error = error
            else:  # pragma: no cover - indicates corrupt kernel parameters
                raise RuntimeError("Structured grid Cholesky failed") from last_error
        return torch.stack(factors)


def _softplus_inverse_numpy(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 1.0e-8)
    return np.where(values > 20.0, values, np.log(np.expm1(values)))


def _positive_link(torch, values, name: str):
    if name == "softplus":
        return torch.nn.functional.softplus(values)
    if name == "exp":
        return values.clamp(max=20.0).exp()
    raise ValueError(name)


def _linked_normal_moments(
    torch,
    mean,
    variance,
    link: str,
    *,
    quadrature_points: int = 12,
    exp_clip: float = 20.0,
):
    """Deterministic first two moments of a positive link of a normal variable."""
    variance = variance.clamp_min(0.0)
    if link == "exp":
        standard_deviation = variance.sqrt()
        safe_deviation = standard_deviation.clamp_min(1.0e-12)
        alpha = (float(exp_clip) - mean) / safe_deviation

        def clipped_lognormal_moment(order: int):
            order_value = float(order)
            log_below = (
                order_value * mean
                + 0.5 * order_value**2 * variance
                + torch.special.log_ndtr(alpha - order_value * standard_deviation)
            )
            log_above = (
                order_value * float(exp_clip)
                + torch.special.log_ndtr(-alpha)
            )
            moment = torch.logaddexp(log_below, log_above).clamp_max(
                order_value * float(exp_clip)
            ).exp()
            deterministic = mean.clamp_max(float(exp_clip)).mul(order_value).exp()
            return torch.where(standard_deviation > 1.0e-10, moment, deterministic)

        linked_mean = clipped_lognormal_moment(1)
        second_moment = clipped_lognormal_moment(2)
    elif link == "softplus":
        if int(quadrature_points) <= 0:
            raise ValueError("quadrature_points must be positive")
        nodes_np, weights_np = np.polynomial.hermite.hermgauss(int(quadrature_points))
        nodes = torch.as_tensor(nodes_np, dtype=mean.dtype, device=mean.device)
        weights = torch.as_tensor(weights_np, dtype=mean.dtype, device=mean.device)
        scale = (2.0 * variance).sqrt()
        linked_mean = torch.zeros_like(mean)
        second_moment = torch.zeros_like(mean)
        for node, weight in zip(nodes.unbind(), weights.unbind()):
            value = torch.nn.functional.softplus(mean + scale * node)
            normalized_weight = weight / math.sqrt(math.pi)
            linked_mean = linked_mean + normalized_weight * value
            second_moment = second_moment + normalized_weight * value.square()
    else:
        raise ValueError(link)
    linked_variance = (second_moment - linked_mean.square()).clamp_min(0.0)
    return linked_mean, linked_variance


def _linked_reconstruction_moments(
    torch,
    factor_mean,
    factor_variance,
    loadings,
    baseline,
):
    """Mean and variance of b + H A' under independent factor posteriors."""
    reconstruction_mean = torch.einsum("rb,fr->bf", factor_mean, loadings)
    reconstruction_mean = reconstruction_mean + baseline[None, :]
    reconstruction_variance = torch.einsum(
        "rb,fr->bf", factor_variance, loadings.square()
    )
    return reconstruction_mean, reconstruction_variance


def _sample_feature_positions(
    rng: np.random.Generator,
    n_features: int,
    batch_size: int,
) -> np.ndarray:
    """Uniform fixed-size feature sample for an unbiased Cartesian minibatch."""
    sample_size = min(int(batch_size), int(n_features))
    if sample_size == n_features:
        return np.arange(n_features, dtype=np.int64)
    return np.sort(
        rng.choice(n_features, size=sample_size, replace=False).astype(np.int64)
    )


def _estimate_spacing_and_extent(coords: np.ndarray) -> tuple[float, float]:
    coords = np.asarray(coords, dtype=np.float64)
    sample = coords[spatially_balanced_sample(coords, min(coords.shape[0], 20_000))]
    # Query representative points against the full coordinate set. Building a
    # tree only on the sample overestimates spacing on dense regular slides.
    tree = cKDTree(coords)
    distances, _ = tree.query(sample, k=2)
    spacing = float(np.median(distances[:, 1]))
    extent = float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))
    return max(spacing, 1.0e-6), max(extent, spacing * 2.0)


def _initial_factor_scales(metadata: pd.DataFrame, n_factors: int, lower: float, upper: float) -> np.ndarray:
    values = []
    for column in ["joint_local_length_scale", "joint_global_length_scale", "learned_length_scale"]:
        if column in metadata:
            candidate = metadata[column].to_numpy(dtype=np.float64)
            values.append(candidate[np.isfinite(candidate) & (candidate > 0.0)])
    if values:
        pool = np.concatenate(values)
        quantiles = np.linspace(0.05, 0.95, int(n_factors))
        result = np.quantile(pool, quantiles)
    else:
        result = np.geomspace(lower, upper, int(n_factors))
    return np.clip(result, lower * 1.001, upper * 0.999).astype(np.float32)


def _run_nmf_initialization(
    store: GPPosteriorFactorStore,
    settings: SharedFactorSettings,
    training_rows: np.ndarray,
) -> dict[str, np.ndarray | float]:
    from sklearn.decomposition import MiniBatchNMF

    available_coords = store.spots.iloc[training_rows][["x", "y"]].to_numpy(dtype=np.float64)
    local_rows = spatially_balanced_sample(
        available_coords, min(settings.nmf_sample_spots, training_rows.size)
    )
    rows = training_rows[local_rows]
    matrix, _ = store.read_all_features(rows)
    baseline = np.quantile(matrix, 0.05, axis=0).astype(np.float32)
    centered = np.maximum(matrix - baseline[None, :], 0.0)
    model = MiniBatchNMF(
        n_components=settings.n_factors,
        init="nndsvda",
        max_iter=settings.nmf_max_iter,
        batch_size=min(1024, centered.shape[0]),
        random_state=settings.random_state,
    )
    location_factors = model.fit_transform(centered).astype(np.float32)
    loadings = model.components_.T.astype(np.float32)
    if settings.nmf_shrinkage > 0.0:
        shrinkage = float(settings.nmf_shrinkage)
        loadings = (
            (1.0 - shrinkage) * loadings
            + shrinkage * loadings.sum(axis=0, keepdims=True) / loadings.shape[0]
        )
        location_factors = (
            (1.0 - shrinkage) * location_factors
            + shrinkage
            * location_factors.sum(axis=1, keepdims=True)
            / location_factors.shape[1]
        )
    factor_mean = np.maximum(location_factors.mean(axis=0), 1.0e-6)
    location_factors /= factor_mean[None, :]
    loadings *= factor_mean[None, :]
    reconstruction = baseline[None, :] + location_factors @ loadings.T
    residual_sigma = np.maximum(
        np.sqrt(np.mean(np.square(matrix - reconstruction), axis=0)),
        settings.sigma_min,
    ).astype(np.float32)
    return {
        "sample_rows": rows,
        "sample_coords": store.spots.iloc[rows][["x", "y"]].to_numpy(dtype=np.float32),
        "location_factors": location_factors,
        "loadings": loadings,
        "baseline": baseline,
        "residual_sigma": residual_sigma,
        "nmf_reconstruction_rmse": float(np.sqrt(np.mean(np.square(matrix - reconstruction)))),
    }


def _initialize_grid_factor_values(
    *,
    sample_coords: np.ndarray,
    sample_factors: np.ndarray,
    inducing_coords: np.ndarray,
    grid_size: int,
    neighbors: int,
    activity_floor: float,
    smoothing_sigma: float,
    normalize_mean: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Transfer sampled NMF factors to the inducing grid without hard zeros."""
    n_neighbors = min(int(neighbors), sample_coords.shape[0])
    distances, indices = cKDTree(sample_coords).query(inducing_coords, k=n_neighbors)
    if n_neighbors == 1:
        grid_values = sample_factors[np.asarray(indices, dtype=np.int64)]
    else:
        distances = np.asarray(distances, dtype=np.float64)
        indices = np.asarray(indices, dtype=np.int64)
        bandwidth = np.maximum(distances[:, -1:], 1.0e-8)
        weights = np.exp(-0.5 * np.square(distances / bandwidth))
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-12)
        grid_values = np.einsum(
            "ij,ijr->ir", weights, sample_factors[indices], optimize=True
        )
    grid_values = np.maximum(grid_values, float(activity_floor))
    if smoothing_sigma > 0.0:
        n_factors = grid_values.shape[1]
        grid_values = gaussian_filter(
            grid_values.reshape(int(grid_size), int(grid_size), n_factors),
            sigma=(float(smoothing_sigma), float(smoothing_sigma), 0.0),
            mode="nearest",
        ).reshape(-1, n_factors)
        grid_values = np.maximum(grid_values, float(activity_floor))
    factor_means = np.ones(grid_values.shape[1], dtype=np.float64)
    if normalize_mean:
        factor_means = np.maximum(
            grid_values.mean(axis=0, dtype=np.float64), float(activity_floor)
        )
        grid_values = grid_values / factor_means[None, :]
    return grid_values.astype(np.float32), factor_means.astype(np.float32)


def fit_shared_spatial_factorization(
    store: GPPosteriorFactorStore,
    output_dir: Path,
    settings: SharedFactorSettings | None = None,
    *,
    resume_checkpoint: Path | None = None,
) -> SharedFactorFitResult:
    """Fit and export shared nonnegative spatial programs."""
    torch, gpytorch = _require_torch_gpytorch()
    settings = settings or SharedFactorSettings()
    settings.validate()
    if settings.n_factors > min(store.n_spots, store.n_features):
        raise ValueError("n_factors exceeds the input rank bound")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest = store.prepare()
    started = time.time()
    device = torch.device(
        settings.device
        if settings.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = torch.float64 if settings.dtype == "float64" else torch.float32
    torch.manual_seed(settings.random_state)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(settings.random_state)
        torch.cuda.reset_peak_memory_stats(device)
    rng = np.random.default_rng(settings.random_state)
    coords = store.spots[["x", "y"]].to_numpy(dtype=np.float32)
    n_validation = int(round(settings.validation_fraction * store.n_spots))
    validation_rows = spatially_balanced_sample(coords, n_validation) if n_validation else np.empty(0, dtype=np.int64)
    training_rows = np.setdiff1d(
        np.arange(store.n_spots, dtype=np.int64), validation_rows, assume_unique=True
    )
    initialization = _run_nmf_initialization(store, settings, training_rows)
    spacing, extent = _estimate_spacing_and_extent(coords)
    axis_spans = np.ptp(coords.astype(np.float64), axis=0)
    inducing_axis_spacing = float(
        np.max(axis_spans) / max(settings.grid_size - 3, 1)
    )
    inferred_min_length_scale = max(
        spacing * 0.5, inducing_axis_spacing * 0.5, 1.0e-4
    )
    min_length_scale = (
        inferred_min_length_scale
        if settings.minimum_length_scale is None
        else float(settings.minimum_length_scale)
    )
    max_length_scale = (
        max(min_length_scale * 2.0, extent)
        if settings.maximum_length_scale is None
        else float(settings.maximum_length_scale)
    )
    if max_length_scale <= min_length_scale:
        raise ValueError(
            "The effective maximum length scale must exceed the effective minimum"
        )
    bounds = [
        (float(coords[:, dimension].min()), float(coords[:, dimension].max()))
        for dimension in range(coords.shape[1])
    ]
    gp = _build_shared_grid_gp(
        n_factors=settings.n_factors,
        grid_size=settings.grid_size,
        grid_bounds=bounds,
        kernel_name=settings.kernel,
        min_length_scale=min_length_scale,
        max_length_scale=max_length_scale,
    ).to(device=device, dtype=dtype)

    sample_coords = np.asarray(initialization["sample_coords"], dtype=np.float32)
    sample_factors = np.asarray(initialization["location_factors"], dtype=np.float32)
    inducing = gp.variational_strategy.inducing_points.detach().cpu().numpy()
    grid_factor_values, grid_factor_means = _initialize_grid_factor_values(
        sample_coords=sample_coords,
        sample_factors=sample_factors,
        inducing_coords=inducing,
        grid_size=settings.grid_size,
        neighbors=settings.initialization_neighbors,
        activity_floor=settings.initialization_activity_floor,
        smoothing_sigma=settings.initialization_grid_smoothing_sigma,
        normalize_mean=settings.normalize_initial_factor_mean,
    )
    initialization["loadings"] = (
        np.asarray(initialization["loadings"], dtype=np.float32)
        * grid_factor_means[None, :]
    )

    class FactorParameters(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.raw_loadings = torch.nn.Parameter(
                torch.as_tensor(
                    _softplus_inverse_numpy(initialization["loadings"]),
                    dtype=dtype,
                    device=device,
                )
            )
            self.raw_baseline = torch.nn.Parameter(
                torch.as_tensor(
                    _softplus_inverse_numpy(initialization["baseline"]),
                    dtype=dtype,
                    device=device,
                )
            )
            sigma_offset = np.maximum(
                np.asarray(initialization["residual_sigma"]) - settings.sigma_min,
                1.0e-8,
            )
            self.raw_sigma = torch.nn.Parameter(
                torch.as_tensor(
                    _softplus_inverse_numpy(sigma_offset),
                    dtype=dtype,
                    device=device,
                )
            )

        def values(self):
            return (
                torch.nn.functional.softplus(self.raw_loadings),
                torch.nn.functional.softplus(self.raw_baseline),
                torch.nn.functional.softplus(self.raw_sigma) + settings.sigma_min,
            )

    parameters = FactorParameters()
    initial_scales = _initial_factor_scales(
        store.metadata, settings.n_factors, min_length_scale, max_length_scale
    )
    if settings.initial_length_scale is not None:
        initial_scales = np.full(
            settings.n_factors,
            np.clip(
                float(settings.initial_length_scale),
                min_length_scale * 1.001,
                max_length_scale * 0.999,
            ),
            dtype=np.float32,
        )
    initialization_summary = {
        "nmf_reconstruction_rmse": float(initialization["nmf_reconstruction_rmse"]),
        "nmf_shrinkage": float(settings.nmf_shrinkage),
        "grid_transfer_neighbors": int(settings.initialization_neighbors),
        "grid_smoothing_sigma_cells": float(
            settings.initialization_grid_smoothing_sigma
        ),
        "activity_floor": float(settings.initialization_activity_floor),
        "normalized_initial_factor_mean": bool(settings.normalize_initial_factor_mean),
        "grid_factor_value_quantiles": {
            str(quantile): float(np.quantile(grid_factor_values, quantile))
            for quantile in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
        "grid_factor_means_absorbed_into_loadings": grid_factor_means.astype(
            float
        ).tolist(),
        "initial_length_scales": initial_scales.astype(float).tolist(),
    }
    (output_dir / "shared_factor_initialization_summary.json").write_text(
        json.dumps(initialization_summary, indent=2), encoding="utf-8"
    )
    variational = gp.variational_strategy._variational_distribution
    with torch.no_grad():
        variational.variational_mean.copy_(
            torch.as_tensor(
                _softplus_inverse_numpy(grid_factor_values.T), dtype=dtype, device=device
            )
        )
        variational._variational_stddev.fill_(0.1)
        gp.variational_strategy.variational_params_initialized.fill_(1)
        gp.covar_module.base_kernel.lengthscale = torch.as_tensor(
            initial_scales, dtype=dtype, device=device
        ).view(-1, 1, 1)
        gp.covar_module.outputscale = torch.ones(
            settings.n_factors, dtype=dtype, device=device
        )

    variational_parameters = list(gp.variational_parameters())
    variational_ids = {id(value) for value in variational_parameters}
    hyperparameters = [value for value in gp.parameters() if id(value) not in variational_ids]
    optimizer = torch.optim.Adam(
        [
            {"params": variational_parameters, "lr": settings.variational_learning_rate},
            {"params": hyperparameters, "lr": settings.hyperparameter_learning_rate},
            {"params": parameters.parameters(), "lr": settings.learning_rate},
        ]
    )
    history: list[dict] = []
    first_epoch = 1
    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        if checkpoint["feature_ids"] != store.feature_ids.tolist():
            raise ValueError("Checkpoint feature order differs from current input")
        gp.load_state_dict(checkpoint["gp_state"])
        parameters.load_state_dict(checkpoint["factor_parameter_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        history = list(checkpoint["history"])
        first_epoch = int(checkpoint["epoch"]) + 1
        rng.bit_generator.state = checkpoint["numpy_rng_state"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng_state"]]
            )

    coords_t_all = torch.as_tensor(coords, dtype=dtype, device=device)
    steps_per_epoch = (
        settings.steps_per_epoch
        if settings.steps_per_epoch > 0
        else int(math.ceil(training_rows.size / settings.location_batch_size))
    )
    sampled_features_per_step = min(settings.feature_batch_size, store.n_features)
    for epoch in range(first_epoch, settings.n_epochs + 1):
        epoch_started = time.time()
        gp.train()
        parameters.train()
        totals = {"elbo": 0.0, "likelihood": 0.0, "kl": 0.0, "l1": 0.0, "diversity": 0.0}
        for step in range(steps_per_epoch):
            replace = training_rows.size < settings.location_batch_size
            rows = rng.choice(
                training_rows,
                size=min(settings.location_batch_size, training_rows.size),
                replace=replace,
            ).astype(np.int64)
            feature_positions = _sample_feature_positions(
                rng, store.n_features, settings.feature_batch_size
            )
            target_np, variance_np = store.read_block(rows, feature_positions)
            target = torch.as_tensor(target_np, dtype=dtype, device=device)
            inherited_variance = torch.as_tensor(variance_np, dtype=dtype, device=device)
            posterior = gp(coords_t_all[rows])
            factor_mean, factor_variance = _linked_normal_moments(
                torch,
                posterior.mean,
                posterior.variance,
                settings.factor_link,
                quadrature_points=settings.link_quadrature_points,
            )
            loadings, baseline, sigma = parameters.values()
            selected_loadings = loadings[feature_positions]
            reconstruction, reconstruction_variance = _linked_reconstruction_moments(
                torch,
                factor_mean,
                factor_variance,
                selected_loadings,
                baseline[feature_positions],
            )
            effective_variance = (
                inherited_variance + sigma[feature_positions].square()[None, :]
            ).clamp(settings.variance_min, settings.variance_max)
            likelihood = scaled_gaussian_log_likelihood(
                target,
                reconstruction,
                effective_variance,
                n_spots_total=training_rows.size,
                n_features_total=store.n_features,
                reconstruction_variance=reconstruction_variance,
            )
            kl = gp.variational_strategy.kl_divergence().sum()
            l1 = settings.loading_l1 * store.n_spots * store.n_features * loadings.mean()
            diversity = target.new_tensor(0.0)
            if settings.diversity_weight > 0.0:
                centered = factor_mean - factor_mean.mean(dim=1, keepdim=True)
                normalized = centered / centered.norm(dim=1, keepdim=True).clamp_min(1.0e-8)
                correlation = normalized @ normalized.transpose(0, 1)
                off_diagonal = correlation - torch.eye(
                    settings.n_factors, dtype=dtype, device=device
                )
                diversity = (
                    settings.diversity_weight
                    * store.n_spots
                    * store.n_features
                    * off_diagonal.square().mean()
                )
            elbo = likelihood - kl - l1 - diversity
            loss = -elbo / (store.n_spots * store.n_features)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(gp.parameters()) + list(parameters.parameters()), 10.0
            )
            optimizer.step()
            totals["elbo"] += float(elbo.detach().cpu())
            totals["likelihood"] += float(likelihood.detach().cpu())
            totals["kl"] += float(kl.detach().cpu())
            totals["l1"] += float(l1.detach().cpu())
            totals["diversity"] += float(diversity.detach().cpu())

        validation = _factor_validation_metrics(
            torch,
            gp,
            parameters,
            store,
            coords_t_all,
            validation_rows,
            settings,
            device,
            dtype,
        )
        record = {
            "epoch": epoch,
            "steps": steps_per_epoch,
            "effective_entry_passes": float(
                epoch
                * steps_per_epoch
                * min(settings.location_batch_size, training_rows.size)
                * sampled_features_per_step
                / max(store.n_spots * store.n_features, 1)
            ),
            "elbo": totals["elbo"] / steps_per_epoch,
            "likelihood_term": totals["likelihood"] / steps_per_epoch,
            "kl_term": totals["kl"] / steps_per_epoch,
            "absolute_likelihood_to_kl_ratio": abs(totals["likelihood"])
            / max(abs(totals["kl"]), 1.0e-12),
            "loading_l1_term": totals["l1"] / steps_per_epoch,
            "factor_diversity_term": totals["diversity"] / steps_per_epoch,
            "validation_rmse": validation["rmse"],
            "validation_weighted_nll": validation["weighted_nll"],
            "epoch_seconds": float(time.time() - epoch_started),
            "learned_length_scale_min": float(gp.covar_module.base_kernel.lengthscale.min().detach().cpu()),
            "learned_length_scale_median": float(gp.covar_module.base_kernel.lengthscale.median().detach().cpu()),
            "learned_length_scale_max": float(gp.covar_module.base_kernel.lengthscale.max().detach().cpu()),
        }
        history.append(record)
        pd.DataFrame(history).to_csv(
            output_dir / "shared_factor_training_history.csv", index=False
        )
        print(
            f"shared-factor epoch {epoch:03d}/{settings.n_epochs} "
            f"elbo={record['elbo']:.3e} val_rmse={record['validation_rmse']:.5f} "
            f"seconds={record['epoch_seconds']:.1f}",
            flush=True,
        )
        if settings.checkpoint_every > 0 and epoch % settings.checkpoint_every == 0:
            _save_factor_checkpoint(
                torch,
                output_dir / "checkpoint_latest.pt",
                epoch,
                gp,
                parameters,
                optimizer,
                history,
                settings,
                store,
                rng,
                device,
            )

    factor_maps, factor_variances, map_index = _export_factor_maps(
        torch,
        gp,
        coords_t_all,
        output_dir,
        settings,
    )
    loadings_t, baseline_t, sigma_t = parameters.values()
    loadings = loadings_t.detach().cpu().numpy().astype(np.float64)
    baseline = baseline_t.detach().cpu().numpy().astype(np.float64)
    sigma = sigma_t.detach().cpu().numpy().astype(np.float64)
    normalized_maps, normalized_loadings, mixtures = normalize_factor_identifiability(
        factor_maps, loadings
    )
    _write_normalized_factor_map_chunks(
        output_dir,
        map_index,
        normalized_maps.astype(np.float32),
    )
    factor_sums = factor_maps.sum(axis=0, dtype=np.float64)
    residual_summary = _write_factor_outputs(
        store,
        output_dir,
        settings,
        loadings,
        normalized_loadings,
        mixtures,
        baseline,
        sigma,
        factor_maps,
        factor_variances,
        factor_sums,
        gp,
        history,
        cache_manifest,
        spacing,
        extent,
        inducing_axis_spacing,
        min_length_scale,
        max_length_scale,
        started,
        device,
        torch,
    )
    del residual_summary
    return SharedFactorFitResult(
        output_dir=output_dir,
        loadings=loadings.astype(np.float32),
        normalized_loadings=normalized_loadings.astype(np.float32),
        feature_mixtures=mixtures.astype(np.float32),
        baseline=baseline.astype(np.float32),
        residual_sigma=sigma.astype(np.float32),
        factor_map_sums=factor_sums,
        factor_length_scales=gp.covar_module.base_kernel.lengthscale.detach().cpu().reshape(-1).numpy(),
        factor_output_scales=gp.covar_module.outputscale.detach().cpu().reshape(-1).numpy(),
        history=pd.DataFrame(history),
    )


def _factor_validation_metrics(
    torch,
    gp,
    parameters,
    store,
    coords_t_all,
    validation_rows,
    settings,
    device,
    dtype,
) -> dict[str, float]:
    if validation_rows.size == 0:
        return {"rmse": float("nan"), "weighted_nll": float("nan")}
    rows = validation_rows[: min(validation_rows.size, settings.location_batch_size)]
    feature_positions = np.arange(store.n_features, dtype=np.int64)
    target_np, variance_np = store.read_block(rows, feature_positions)
    target = torch.as_tensor(target_np, dtype=dtype, device=device)
    inherited_variance = torch.as_tensor(variance_np, dtype=dtype, device=device)
    gp.eval()
    parameters.eval()
    with torch.no_grad():
        posterior = gp(coords_t_all[rows])
        factor_mean, factor_variance = _linked_normal_moments(
            torch,
            posterior.mean,
            posterior.variance,
            settings.factor_link,
            quadrature_points=settings.link_quadrature_points,
        )
        loadings, baseline, sigma = parameters.values()
        reconstruction, reconstruction_variance = _linked_reconstruction_moments(
            torch, factor_mean, factor_variance, loadings, baseline
        )
        effective_variance = (
            inherited_variance + sigma.square()[None, :]
        ).clamp(settings.variance_min, settings.variance_max)
        rmse = torch.sqrt(torch.mean((target - reconstruction).square()))
        weighted_nll = 0.5 * torch.mean(
            torch_log_two_pi(target)
            + effective_variance.log()
            + (
                (target - reconstruction).square() + reconstruction_variance
            ) / effective_variance
        )
    return {"rmse": float(rmse.cpu()), "weighted_nll": float(weighted_nll.cpu())}


def _save_factor_checkpoint(
    torch,
    path,
    epoch,
    gp,
    parameters,
    optimizer,
    history,
    settings,
    store,
    rng,
    device,
) -> None:
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "gp_state": gp.state_dict(),
        "factor_parameter_state": parameters.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history": history,
        "settings": asdict(settings),
        "posterior_link_moments": (
            "deterministic analytic moments for clipped exp; deterministic "
            "Gauss-Hermite moments for softplus"
        ),
        "feature_ids": store.feature_ids.tolist(),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _export_factor_maps(torch, gp, coords_t_all, output_dir, settings):
    map_dir = output_dir / "shared_factor_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    n_spots = coords_t_all.shape[0]
    means = np.empty((n_spots, settings.n_factors), dtype=np.float32)
    variances = np.empty_like(means)
    rows = []
    gp.eval()
    with torch.no_grad():
        for chunk_id, start in enumerate(range(0, n_spots, settings.map_chunk_size_spots)):
            stop = min(start + settings.map_chunk_size_spots, n_spots)
            posterior = gp(coords_t_all[start:stop])
            mean_t, variance_t = _linked_normal_moments(
                torch,
                posterior.mean,
                posterior.variance,
                settings.factor_link,
                quadrature_points=settings.link_quadrature_points,
            )
            mean = mean_t.transpose(0, 1).cpu().numpy().astype(np.float32)
            variance = variance_t.transpose(0, 1).cpu().numpy().astype(np.float32)
            means[start:stop] = mean
            variances[start:stop] = variance
            file_name = f"shared_factor_maps_chunk_{chunk_id:04d}.npz"
            np.savez_compressed(map_dir / file_name, posterior_mean=mean, posterior_variance=variance)
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "file": file_name,
                    "spot_start": start,
                    "spot_stop": stop,
                    "n_spots": stop - start,
                    "n_factors": settings.n_factors,
                }
            )
    index = pd.DataFrame(rows)
    index.to_csv(map_dir / "shared_factor_map_chunks.csv", index=False)
    return means, variances, index


def _write_normalized_factor_map_chunks(output_dir, map_index, normalized_maps):
    map_dir = output_dir / "shared_factor_maps"
    for row in map_index.itertuples(index=False):
        path = map_dir / row.file
        with np.load(path) as npz:
            mean = np.asarray(npz["posterior_mean"], dtype=np.float32)
            variance = np.asarray(npz["posterior_variance"], dtype=np.float32)
        np.savez_compressed(
            path,
            posterior_mean=mean,
            posterior_variance=variance,
            normalized_mean=normalized_maps[row.spot_start : row.spot_stop],
        )


def _write_factor_outputs(
    store,
    output_dir,
    settings,
    loadings,
    normalized_loadings,
    mixtures,
    baseline,
    sigma,
    factor_maps,
    factor_variances,
    factor_sums,
    gp,
    history,
    cache_manifest,
    spacing,
    extent,
    inducing_axis_spacing,
    min_length_scale,
    max_length_scale,
    started,
    device,
    torch,
):
    store.spots.to_csv(output_dir / "shared_factor_spots.csv.gz", index=False)
    residual_rows = []
    for start in range(0, store.n_features, settings.residual_feature_chunk_size):
        stop = min(start + settings.residual_feature_chunk_size, store.n_features)
        features = np.arange(start, stop, dtype=np.int64)
        sum_residual = np.zeros(stop - start, dtype=np.float64)
        sum_square = np.zeros(stop - start, dtype=np.float64)
        sum_abs = np.zeros(stop - start, dtype=np.float64)
        for spot_start in range(0, store.n_spots, settings.map_chunk_size_spots):
            spot_stop = min(spot_start + settings.map_chunk_size_spots, store.n_spots)
            rows = np.arange(spot_start, spot_stop, dtype=np.int64)
            target, _ = store.read_block(rows, features)
            reconstruction = baseline[features][None, :] + factor_maps[rows] @ loadings[features].T
            residual = target - reconstruction
            sum_residual += residual.sum(axis=0, dtype=np.float64)
            sum_square += np.square(residual).sum(axis=0, dtype=np.float64)
            sum_abs += np.abs(residual).sum(axis=0, dtype=np.float64)
        for offset, position in enumerate(features):
            residual_rows.append(
                {
                    "gene": store.feature_ids[position],
                    "feature_position": int(position),
                    "residual_mean": float(sum_residual[offset] / store.n_spots),
                    "residual_rmse": float(math.sqrt(sum_square[offset] / store.n_spots)),
                    "residual_mean_absolute": float(sum_abs[offset] / store.n_spots),
                }
            )
    residual_summary = pd.DataFrame(residual_rows)
    residual_summary.to_csv(output_dir / "feature_specific_spatial_residual_summary.csv", index=False)
    loading_table = store.metadata.copy()
    loading_table["feature_amplitude"] = store.amplitudes
    loading_table["baseline"] = baseline
    loading_table["feature_specific_sigma"] = sigma
    for factor in range(settings.n_factors):
        loading_table[f"A_tilde_{factor:02d}"] = normalized_loadings[:, factor]
        loading_table[f"Pi_{factor:02d}"] = mixtures[:, factor]
    loading_table = loading_table.merge(
        residual_summary, on=["gene", "feature_position"], how="left", validate="one_to_one"
    )
    loading_table.to_parquet(output_dir / "shared_factor_loadings.parquet", index=False)
    loading_table.to_csv(output_dir / "shared_factor_loadings.csv.gz", index=False)

    global_fraction = np.full(store.n_features, np.nan, dtype=np.float64)
    if {"joint_global_outputscale", "joint_local_outputscale"}.issubset(store.metadata.columns):
        global_scale = store.metadata["joint_global_outputscale"].to_numpy(dtype=np.float64)
        local_scale = store.metadata["joint_local_outputscale"].to_numpy(dtype=np.float64)
        global_fraction = global_scale / np.maximum(global_scale + local_scale, 1.0e-12)
    inherited = np.nansum(loadings * global_fraction[:, None], axis=0) / np.maximum(
        np.nansum(loadings * np.isfinite(global_fraction)[:, None], axis=0), 1.0e-12
    )
    normalized_maps = factor_maps / np.maximum(factor_sums[None, :], 1.0e-12)
    factor_metadata = []
    length_scales = gp.covar_module.base_kernel.lengthscale.detach().cpu().reshape(-1).numpy()
    output_scales = gp.covar_module.outputscale.detach().cpu().reshape(-1).numpy()
    for factor in range(settings.n_factors):
        factor_metadata.append(
            {
                "factor": factor,
                "length_scale": float(length_scales[factor]),
                "output_scale": float(output_scales[factor]),
                "inherited_global_fraction": float(inherited[factor]),
                "factor_map_sum": float(factor_sums[factor]),
                "concentration_score": float(
                    store.n_spots * np.square(normalized_maps[:, factor]).sum()
                ),
                "spatial_extent_weighted_variance": float(
                    np.average(
                        np.square(
                            store.spots[["x", "y"]].to_numpy(dtype=np.float64)
                            - np.average(
                                store.spots[["x", "y"]].to_numpy(dtype=np.float64),
                                axis=0,
                                weights=factor_maps[:, factor],
                            )
                        ).sum(axis=1),
                        weights=factor_maps[:, factor],
                    )
                ),
            }
        )
    pd.DataFrame(factor_metadata).to_csv(output_dir / "shared_factor_summary.csv", index=False)
    peak_gpu_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    metadata = {
        "format_version": 1,
        "model": "unified structured-grid probabilistic nonnegative spatial factorization",
        "input": {
            "reference_vski_dir": str(store.reference_vski_dir),
            "selected_models": None if store.selected_models is None else str(store.selected_models),
            "n_spots": store.n_spots,
            "n_features": store.n_features,
            "feature_order_sha256": hashlib.sha256("\n".join(store.feature_ids).encode()).hexdigest(),
            "total_posterior_variance": (
                "saved posterior variance; joint maps use V_global + V_local under "
                "the factorized variational posterior because cross-covariance is unavailable"
            ),
        },
        "settings": asdict(settings),
        "kernel_note": (
            "matern32_product is a separable product of one-dimensional Matern-3/2 "
            "kernels, enabling exact Kronecker mean-field KL; it is not the isotropic "
            "Euclidean Matern-3/2 covariance"
            if settings.kernel == "matern32_product"
            else "separable RBF grid kernel"
        ),
        "coordinates": {
            "units": "inherited_from_feature_gp_run",
            "median_nearest_neighbor_spacing": float(spacing),
            "coordinate_extent": float(extent),
            "maximum_inducing_axis_spacing": float(inducing_axis_spacing),
            "minimum_length_scale": float(min_length_scale),
            "maximum_length_scale": float(max_length_scale),
            "minimum_length_scale_rule": (
                "explicit CLI/settings override"
                if settings.minimum_length_scale is not None
                else "max(0.5 * nearest-neighbor spacing, "
                "0.5 * maximum inducing-axis spacing)"
            ),
        },
        "cache": cache_manifest,
        "runtime": {
            "elapsed_seconds": float(time.time() - started),
            "peak_gpu_memory_bytes": peak_gpu_memory,
            "peak_cpu_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "device": str(device),
            "grid_points_per_factor": int(settings.grid_size**2),
            "dense_gpu_likelihood_shape": [
                settings.location_batch_size,
                settings.feature_batch_size,
                settings.n_factors,
            ],
        },
        "outputs": {
            "loadings_parquet": "shared_factor_loadings.parquet",
            "loadings_csv": "shared_factor_loadings.csv.gz",
            "factor_maps": "shared_factor_maps/",
            "spots": "shared_factor_spots.csv.gz",
            "factor_summary": "shared_factor_summary.csv",
            "training_history": "shared_factor_training_history.csv",
            "initialization_summary": "shared_factor_initialization_summary.json",
            "residual_summary": "feature_specific_spatial_residual_summary.csv",
            "lazy_residual_definition": "D = cached_factor_input_mean - (baseline + H @ A.T)",
            "checkpoint": "checkpoint_latest.pt",
        },
    }
    (output_dir / "shared_factor_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(output_dir / "shared_factor_training_history.csv", index=False)
    return residual_summary
