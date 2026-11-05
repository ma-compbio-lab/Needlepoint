"""Execution-resource planning without altering scientific resolution/support."""
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Resources:
    device: str = 'auto'
    cpu_threads: int = 1
    memory_fraction: float = 0.7
    feature_chunk_size: int | None = None
    batch_size: int | None = None
    local_kl_batch_size: int = 1024

    def resolve(self, *, n_spots, n_features, grid_size):
        """Return a conservative starting allocation, not a memory guarantee.

        Auto selects a visible CUDA device or CPU; CUDA ordinals obey the
        scheduler's CUDA_VISIBLE_DEVICES mask. Explicit CUDA never falls back.
        Estimates are heuristics; users can override batch/chunk sizes.
        """
        if self.cpu_threads < 1 or self.local_kl_batch_size < 1 or not 0 < self.memory_fraction <= 0.9:
            raise ValueError('Positive threads/KL batch and 0 < memory_fraction <= 0.9 required')
        if int(self.cpu_threads) != self.cpu_threads or int(self.local_kl_batch_size) != self.local_kl_batch_size:
            raise ValueError('Threads and KL batch must be integers')
        if min(n_spots, n_features) < 1 or grid_size < 4:
            raise ValueError('Positive dimensions and grid_size >= 4 required')
        device = self.device
        free = None
        if device == 'auto' or device.startswith('cuda'):
            try:
                import torch
            except ImportError:
                if device != 'auto': raise ImportError('CUDA execution requires needlepoint[gp]')
                device = 'cpu'
            else:
                if device == 'auto': device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                if device.startswith('cuda'):
                    if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable; choose cpu explicitly')
                    index = torch.device(device).index
                    index = 0 if index is None else index
                    if index >= torch.cuda.device_count(): raise ValueError('CUDA ordinal is outside the visible device allocation')
                    device = f'cuda:{index}'
                    free, _ = torch.cuda.mem_get_info(index)
        if device != 'cpu' and not device.startswith('cuda:'):
            raise ValueError('Supported devices: auto, cpu, cuda, cuda:<visible ordinal>')
        for value in (self.feature_chunk_size, self.batch_size):
            if value is not None and (value < 1 or int(value) != value):
                raise ValueError('Batch/chunk overrides must be positive integers')
        batch = min(n_spots, self.batch_size or (8192 if free else 1024))
        # Account for structured grid factors, local variational state and
        # active likelihood tensors. This is deliberately a planning heuristic.
        per_feature = 4*n_spots*4 + 8*grid_size**2*12 + 4*batch*64
        available = int(free*self.memory_fraction) if free else 512*1024**2
        chunk = min(n_features, self.feature_chunk_size or max(1, min(64 if free else 8, available//max(1, per_feature))))
        estimate = per_feature*chunk
        return {'device': device, 'cpu_threads': self.cpu_threads,
                'feature_chunk_size': int(chunk), 'batch_size': int(batch),
                'local_kl_batch_size': self.local_kl_batch_size,
                'visible_cuda_mask': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'free_cuda_bytes_at_planning': free,
                'estimated_working_bytes': estimate,
                'estimate_is_guarantee': False,
                'warnings': ['Estimated memory exceeds planning budget; lower batch/chunk overrides'] if estimate > available else []}
