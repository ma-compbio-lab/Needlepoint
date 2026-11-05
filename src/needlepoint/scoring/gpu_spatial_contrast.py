"""CUDA FFT primitives for multiscale spatial-contrast scoring.

This module deliberately contains only the regular-grid smoothing hot path.
Sparse H5AD loading, count normalization, candidate selection, permutation
planning, FDR calibration, and reporting remain in the generic scorer.  The
separation keeps the established CPU implementation available as an exact
reference and makes CUDA use explicit in run manifests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass
class CUDAFFTDiagnostics:
    """Runtime and memory diagnostics accumulated across scored blocks."""

    device: str
    device_name: str
    torch_version: str
    cuda_version: str | None
    reduction_accumulator: str
    n_blocks: int = 0
    n_profiles: int = 0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TorchFFTGridScorer:
    """Batched CUDA implementation of masked regular-grid Gaussian smoothing.

    The CPU scorer constructs the padded grid, Gaussian frequency kernels, and
    mask-normalization denominators.  Reusing those arrays here ensures that
    CPU and CUDA differ only in the FFT/reduction implementation, not in grid
    geometry or kernel construction.
    """

    def __init__(
        self,
        geometry,
        cpu_smoother,
        *,
        device: str = "cuda",
        reduction_accumulator: str = "float64",
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("PyTorch is required for grid-fft-cuda") from exc

        if reduction_accumulator not in {"float32", "float64"}:
            raise ValueError("reduction_accumulator must be float32 or float64")
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(f"grid-fft-cuda requires a CUDA device, got {resolved}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to PyTorch")
        # Resolving the index now makes provenance unambiguous when the caller
        # uses the generic ``cuda`` alias.
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())

        self.torch = torch
        self.device = resolved
        self.reduction_dtype = (
            torch.float64 if reduction_accumulator == "float64" else torch.float32
        )
        self.padded_shape = tuple(map(int, cpu_smoother.padded_shape))
        self.offset_y = int(cpu_smoother.offset_y)
        self.offset_x = int(cpu_smoother.offset_x)

        y = np.asarray(geometry.y_index, dtype=np.int64) + self.offset_y
        x = np.asarray(geometry.x_index, dtype=np.int64) + self.offset_x
        self.y_index = torch.as_tensor(y, dtype=torch.long, device=resolved)
        self.x_index = torch.as_tensor(x, dtype=torch.long, device=resolved)
        self.kernels = {
            float(key): torch.as_tensor(
                np.asarray(value, dtype=np.float32),
                dtype=torch.float32,
                device=resolved,
            )
            for key, value in cpu_smoother.kernels.items()
        }
        # Only denominators at occupied bins are required.  This avoids keeping
        # seven full cropped grids on the GPU and removes a large crop/divide
        # temporary from every inverse transform.
        occupied_y = np.asarray(geometry.y_index, dtype=np.int64)
        occupied_x = np.asarray(geometry.x_index, dtype=np.int64)
        self.denominators_at_spots = {
            float(key): torch.as_tensor(
                np.asarray(value[occupied_y, occupied_x], dtype=np.float32),
                dtype=torch.float32,
                device=resolved,
            )
            for key, value in cpu_smoother.denominators.items()
        }
        properties = torch.cuda.get_device_properties(resolved)
        self.diagnostics = CUDAFFTDiagnostics(
            device=str(resolved),
            device_name=str(properties.name),
            torch_version=str(torch.__version__),
            cuda_version=None if torch.version.cuda is None else str(torch.version.cuda),
            reduction_accumulator=reduction_accumulator,
        )

    def score_scale_metrics(
        self,
        block: np.ndarray,
        bandwidths: Iterable[float],
        *,
        tss: np.ndarray,
        variance: np.ndarray,
    ) -> dict[float, tuple[np.ndarray, np.ndarray]]:
        """Return CPU-compatible ``(r2, contrast)`` arrays for every bandwidth."""

        torch = self.torch
        values = np.asarray(block, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("block must have shape (n_spots, n_features)")
        n_spots, n_features = map(int, values.shape)
        if n_spots != int(self.y_index.numel()):
            raise ValueError("block spot count does not match CUDA grid geometry")
        if np.asarray(tss).shape != (n_features,) or np.asarray(variance).shape != (n_features,):
            raise ValueError("tss and variance must have one value per feature")

        torch.cuda.reset_peak_memory_stats(self.device)
        block_gpu = torch.as_tensor(
            np.ascontiguousarray(values.T),
            dtype=torch.float32,
            device=self.device,
        )
        canvas = torch.zeros(
            (n_features, *self.padded_shape),
            dtype=torch.float32,
            device=self.device,
        )
        canvas[:, self.y_index, self.x_index] = block_gpu
        block_fft = torch.fft.rfftn(canvas, dim=(-2, -1))
        del canvas

        tss_gpu = torch.as_tensor(
            np.asarray(tss, dtype=np.float64 if self.reduction_dtype == torch.float64 else np.float32),
            dtype=self.reduction_dtype,
            device=self.device,
        )
        variance_gpu = torch.as_tensor(
            np.asarray(
                variance,
                dtype=np.float64 if self.reduction_dtype == torch.float64 else np.float32,
            ),
            dtype=self.reduction_dtype,
            device=self.device,
        )
        output: dict[float, tuple[np.ndarray, np.ndarray]] = {}
        for raw_bandwidth in bandwidths:
            bandwidth = float(raw_bandwidth)
            if bandwidth not in self.kernels:
                raise KeyError(f"CUDA FFT kernel is absent for bandwidth {bandwidth:g}")
            smoothed_grid = torch.fft.irfftn(
                block_fft * self.kernels[bandwidth].unsqueeze(0),
                s=self.padded_shape,
                dim=(-2, -1),
            )
            smoothed = smoothed_grid[:, self.y_index, self.x_index]
            smoothed = smoothed / self.denominators_at_spots[bandwidth].unsqueeze(0)
            del smoothed_grid

            residual = block_gpu - smoothed
            sse = torch.sum(residual.square(), dim=1, dtype=self.reduction_dtype)
            r2 = torch.zeros(n_features, dtype=self.reduction_dtype, device=self.device)
            valid_tss = tss_gpu > 1.0e-12
            r2[valid_tss] = 1.0 - sse[valid_tss] / tss_gpu[valid_tss]
            r2.clamp_(0.0, 1.0)

            smooth_sum = torch.sum(smoothed, dim=1, dtype=self.reduction_dtype)
            smooth_sum_sq = torch.sum(smoothed.square(), dim=1, dtype=self.reduction_dtype)
            smooth_mean = smooth_sum / max(n_spots, 1)
            smooth_var = torch.clamp(
                smooth_sum_sq / max(n_spots, 1) - smooth_mean.square(),
                min=0.0,
            )
            contrast = torch.zeros(n_features, dtype=self.reduction_dtype, device=self.device)
            valid_var = variance_gpu > 1.0e-12
            contrast[valid_var] = smooth_var[valid_var] / variance_gpu[valid_var]
            contrast.clamp_(0.0, 1.0)
            output[bandwidth] = (
                r2.to(dtype=torch.float32).cpu().numpy(),
                contrast.to(dtype=torch.float32).cpu().numpy(),
            )
            del residual, sse, r2, smooth_sum, smooth_sum_sq, smooth_mean, smooth_var, contrast
            del smoothed

        torch.cuda.synchronize(self.device)
        self.diagnostics.n_blocks += 1
        self.diagnostics.n_profiles += n_features
        self.diagnostics.peak_allocated_bytes = max(
            self.diagnostics.peak_allocated_bytes,
            int(torch.cuda.max_memory_allocated(self.device)),
        )
        self.diagnostics.peak_reserved_bytes = max(
            self.diagnostics.peak_reserved_bytes,
            int(torch.cuda.max_memory_reserved(self.device)),
        )
        del block_gpu, block_fft, tss_gpu, variance_gpu
        return output

    def manifest(self) -> dict[str, Any]:
        """Return JSON-serializable backend provenance and peak memory use."""

        output = self.diagnostics.to_dict()
        output.update(
            {
                "implementation": "torch.fft.rfftn/irfftn batched regular-grid Gaussian smoothing",
                "padded_shape_yx": list(self.padded_shape),
                "n_observed_spots": int(self.y_index.numel()),
                "kernel_bandwidths_um": sorted(self.kernels),
            }
        )
        return output
