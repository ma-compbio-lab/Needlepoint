"""Blockwise reconstruction of posterior moments for prototype consumers."""
import json
import os
import time

import numpy as np
import pandas as pd

from needlepoint.artifacts import atomic_write_json
from needlepoint.posterior_blocks import PosteriorBlocks


class StreamingFactorInput:
    block_size = 8192

    def __init__(self, store):
        self.store = store
        self.reader = PosteriorBlocks()

    def reconstruct(self, path, table, rows):
        from needlepoint.shared_spatial_factorization import project_posterior_nonnegative
        s = self.store
        columns = table.map_column.to_numpy(dtype=np.int64)
        mean = self.reader.read(path, 'posterior_mean', s.source_spot_rows[rows], columns)
        variance = self.reader.read(path, 'posterior_variance', s.source_spot_rows[rows], columns)
        scale = table.standardization_std.to_numpy(dtype=np.float64)
        mean = mean.astype(np.float64) * scale + table.standardization_mean.to_numpy(dtype=np.float64)
        variance = variance.astype(np.float64) * scale**2
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError(f'Non-finite GP posterior values in {path}')
        if (variance < 0).any():
            raise ValueError(f'Negative GP posterior variance in {path}')
        minimum = mean.min(axis=0)
        counts = (mean < 0).sum(axis=0)
        if minimum.min() < -s.numerical_floor_tolerance and s.nonnegative_projection == 'error':
            raise ValueError(f'Materially negative reconstructed GP values detected in {path}')
        mean, variance = project_posterior_nonnegative(mean, variance, table,
            s.nonnegative_projection, s.numerical_floor_tolerance)
        return mean, variance, minimum, counts

    def prepare(self, *, rebuild=False):
        s = self.store
        signature = s._source_signature()
        if not rebuild and s.manifest_path.is_file() and s.amplitude_path.is_file():
            manifest = json.loads(s.manifest_path.read_text())
            if manifest.get('storage_mode') == 'streaming-v1' and manifest.get('source_signature') == signature:
                table = pd.read_csv(s.amplitude_path)
                if not np.array_equal(table.gene.astype(str), s.feature_ids):
                    raise ValueError('Cached feature-amplitude order does not match source')
                s.amplitudes = table.feature_amplitude.to_numpy(dtype=np.float64)
                if not np.isfinite(s.amplitudes).all() or (s.amplitudes <= 0).any():
                    raise ValueError('Invalid saved feature amplitudes')
                s.validation_report = manifest['validation']
                return manifest
        started = time.time()
        sums = np.zeros(s.n_features, dtype=np.float64)
        minima = np.full(s.n_features, np.inf)
        counts = np.zeros(s.n_features, dtype=np.int64)
        tiny = 0
        for path, table in s.metadata.groupby('model_chunk_path', sort=False):
            positions = table.index.to_numpy(dtype=np.int64)
            for start in range(0, s.n_spots, self.block_size):
                rows = np.arange(start, min(start + self.block_size, s.n_spots))
                mean, _, minimum, count = self.reconstruct(path, table, rows)
                sums[positions] += mean.sum(axis=0, dtype=np.float64)
                minima[positions] = np.minimum(minima[positions], minimum)
                counts[positions] += count
            if minima[positions].min() >= -s.numerical_floor_tolerance:
                tiny += int(counts[positions].sum())
        s.amplitudes = np.ones(s.n_features, dtype=np.float64)
        if s.normalization_mode == 'shape':
            s.amplitudes = sums if s.amplitude_statistic == 'sum' else sums / max(s.n_spots, 1)
            if (s.amplitudes <= 0).any():
                raise ValueError('Non-positive feature amplitudes')
        s.validation_report = dict(n_spots=s.n_spots, n_features=s.n_features,
            finite_values=True, posterior_variance_nonnegative=True,
            negative_entry_count_before_floor=int(counts.sum()),
            features_with_negative_entries=int((counts > 0).sum()),
            tiny_negative_entries_floored=tiny,
            nonnegative_projection=s.nonnegative_projection,
            most_negative_entry=float(min(0, minima.min())),
            numerical_floor_tolerance=s.numerical_floor_tolerance,
            total_variance_assumption='producer-declared total moments on the metadata field scale; this reader never recombines component variances')
        table = pd.DataFrame(dict(gene=s.feature_ids,
            feature_position=np.arange(s.n_features), feature_amplitude=s.amplitudes,
            reconstructed_minimum=minima, negative_entry_count_before_floor=counts))
        temporary = s.amplitude_path.with_name(f'.{s.amplitude_path.name}.{os.getpid()}.partial')
        table.to_csv(temporary, index=False)
        os.replace(temporary, s.amplitude_path)
        manifest = dict(cache_version=s.CACHE_VERSION, storage_mode='streaming-v1',
            source_signature=signature, reference_vski_dir=str(s.reference_vski_dir),
            selected_models=None if s.selected_models is None else str(s.selected_models),
            shape=[s.n_spots, s.n_features], normalization_mode=s.normalization_mode,
            amplitude_statistic=s.amplitude_statistic, nonnegative_projection=s.nonnegative_projection,
            mean_file=None, variance_file=None, amplitude_file=s.amplitude_path.name,
            validation=s.validation_report,
            gp_artifact_audits=[a.to_dict() for a in s.gp_artifact_audits],
            elapsed_seconds=time.time()-started)
        atomic_write_json(s.manifest_path, manifest)
        return manifest

    def read_block(self, rows, features):
        s = self.store
        rows = np.asarray(rows, dtype=np.int64)
        features = np.asarray(features, dtype=np.int64)
        mean = np.empty((len(rows), len(features)), dtype=np.float32)
        variance = np.empty_like(mean)
        if not len(rows) or not len(features):
            return mean, variance
        table = s.metadata.iloc[features].copy().reset_index(drop=True)
        for path, group in table.groupby('model_chunk_path', sort=False):
            positions = group.index.to_numpy(dtype=np.int64)
            for start in range(0, len(rows), self.block_size):
                stop = min(start + self.block_size, len(rows))
                m, v, _, _ = self.reconstruct(path, group, rows[start:stop])
                # Match the previous FP32 cache's rounding before normalization.
                amplitudes = s.amplitudes[features[positions]]
                mean[start:stop, positions] = m.astype(np.float32) / amplitudes
                variance[start:stop, positions] = v.astype(np.float32) / amplitudes**2
        return mean, variance
