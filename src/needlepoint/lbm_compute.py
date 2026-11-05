"""Optional Torch acceleration for the dense streamed LBM updates."""

from __future__ import annotations

from typing import Iterator

import numpy as np


class TorchLBMCompute:
    """Run LBM block products and soft domain updates on a Torch device.

    Map blocks remain streamed by the existing store.  Only the current block
    and the O(NK)/O(NR) LBM state are resident on the selected device.
    """

    def __init__(self, *, device: str = "cuda", dtype: str = "float32") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("The torch LBM backend requires PyTorch") from exc
        if dtype not in {"float32", "float64"}:
            raise ValueError("LBM compute dtype must be float32 or float64")
        self.torch = torch
        self.device = torch.device(device)
        self.dtype_name = dtype
        self.dtype = torch.float64 if dtype == "float64" else torch.float32
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested for LBM but is unavailable")
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def audit(self) -> dict:
        peak = 0
        if self.device.type == "cuda":
            peak = int(self.torch.cuda.max_memory_allocated(self.device))
        return {
            "backend": "torch",
            "device": str(self.device),
            "dtype": self.dtype_name,
            "peak_device_memory_bytes": peak,
            "gpu_accelerated": bool(self.device.type == "cuda"),
            "hardcap_update_backend": "numpy",
            "graph_message_backend": "numpy",
        }

    def feature_domain_stats(
        self,
        blocks: Iterator,
        *,
        n_features: int,
        n_domains: int,
        B: np.ndarray,
        spot_weights: np.ndarray,
    ) -> dict[str, np.ndarray]:
        torch = self.torch
        FtB = torch.zeros((n_features, n_domains), dtype=self.dtype, device=self.device)
        F2tB = torch.zeros_like(FtB)
        UtB = torch.zeros_like(FtB)
        B_t = torch.as_tensor(B, dtype=self.dtype, device=self.device)
        weights_t = torch.as_tensor(spot_weights, dtype=self.dtype, device=self.device)
        for rows, mean, var, positions, _ in blocks:
            rows_t = torch.as_tensor(rows, dtype=torch.long, device=self.device)
            positions_t = torch.as_tensor(positions, dtype=torch.long, device=self.device)
            mean_t = torch.as_tensor(mean, dtype=self.dtype, device=self.device)
            B_block = B_t.index_select(0, rows_t) * weights_t.index_select(0, rows_t)[:, None]
            FtB.index_add_(0, positions_t, mean_t.T @ B_block)
            F2tB.index_add_(0, positions_t, mean_t.square().T @ B_block)
            if var is not None:
                var_t = torch.as_tensor(var, dtype=self.dtype, device=self.device)
                UtB.index_add_(0, positions_t, var_t.T @ B_block)
        return {
            "FtB": self._numpy64(FtB),
            "F2tB": self._numpy64(F2tB),
            "UtB": self._numpy64(UtB),
        }

    def spot_module_stats(
        self,
        blocks: Iterator,
        *,
        n_spots: int,
        n_modules: int,
        A: np.ndarray,
        compute_module_maps: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        FA = torch.zeros((n_spots, n_modules), dtype=self.dtype, device=self.device)
        F2A = torch.zeros_like(FA)
        UA = torch.zeros_like(FA)
        module_maps = torch.zeros_like(FA) if compute_module_maps else None
        A_t = torch.as_tensor(A, dtype=self.dtype, device=self.device)
        nA = torch.clamp(A_t.sum(dim=0), min=1.0e-8)
        A_norm = A_t / nA[None, :]
        for rows, mean, var, positions, _ in blocks:
            rows_t = torch.as_tensor(rows, dtype=torch.long, device=self.device)
            positions_t = torch.as_tensor(positions, dtype=torch.long, device=self.device)
            mean_t = torch.as_tensor(mean, dtype=self.dtype, device=self.device)
            A_block = A_t.index_select(0, positions_t)
            FA.index_add_(0, rows_t, mean_t @ A_block)
            F2A.index_add_(0, rows_t, mean_t.square() @ A_block)
            if var is not None:
                var_t = torch.as_tensor(var, dtype=self.dtype, device=self.device)
                UA.index_add_(0, rows_t, var_t @ A_block)
            if compute_module_maps:
                assert module_maps is not None
                module_maps.index_add_(
                    0,
                    rows_t,
                    mean_t @ A_norm.index_select(0, positions_t),
                )
        return (
            self._numpy64(FA),
            self._numpy64(F2A),
            self._numpy64(UA),
            (
                module_maps.detach().cpu().numpy().astype(np.float32, copy=False)
                if module_maps is not None
                else np.empty((0, 0), dtype=np.float32)
            ),
        )

    def spot_evidence(
        self,
        FA: np.ndarray,
        F2A: np.ndarray,
        nA: np.ndarray,
        mu: np.ndarray,
        var: np.ndarray,
    ) -> np.ndarray:
        torch = self.torch
        FA_t = torch.as_tensor(FA, dtype=self.dtype, device=self.device)
        F2A_t = torch.as_tensor(F2A, dtype=self.dtype, device=self.device)
        nA_t = torch.as_tensor(nA, dtype=self.dtype, device=self.device)
        mu_t = torch.as_tensor(mu, dtype=self.dtype, device=self.device)
        var_t = torch.clamp(
            torch.as_tensor(var, dtype=self.dtype, device=self.device), min=1.0e-12
        )
        inv_var = var_t.reciprocal()
        const = nA_t @ torch.log(var_t) + nA_t @ (mu_t.square() * inv_var)
        evidence = -0.5 * (
            F2A_t @ inv_var - 2.0 * (FA_t @ (mu_t * inv_var)) + const[None, :]
        )
        return self._numpy64(evidence)

    def row_softmax(self, logits: np.ndarray, *, output_float32: bool = False) -> np.ndarray:
        torch = self.torch
        values = torch.as_tensor(logits, dtype=self.dtype, device=self.device)
        values = values - values.max(dim=1, keepdim=True).values
        exponentials = torch.exp(torch.clamp(values, min=-80.0, max=80.0))
        denominator = exponentials.sum(dim=1, keepdim=True)
        probabilities = exponentials / torch.clamp(denominator, min=1.0e-30)
        result = probabilities.detach().cpu().numpy()
        return result.astype(np.float32 if output_float32 else np.float64, copy=False)

    def top_two_margin(self, logits: np.ndarray) -> np.ndarray:
        torch = self.torch
        values = torch.as_tensor(logits, dtype=self.dtype, device=self.device)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("Expected a nonempty two-dimensional logit matrix")
        if values.shape[1] == 1:
            return np.full(values.shape[0], np.inf, dtype=np.float64)
        top_two = torch.topk(values, k=2, dim=1, sorted=True).values
        return self._numpy64(top_two[:, 0] - top_two[:, 1])

    def capped_softmax(
        self,
        logits: np.ndarray,
        *,
        capacity_factor: float,
        temperature: float,
        step: float,
        max_iter: int,
        tol: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        torch = self.torch
        values = torch.as_tensor(logits, dtype=self.dtype, device=self.device)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError("Expected nonempty logits with shape n_spots x n_domains")
        if capacity_factor < 1.0 or temperature <= 0.0 or max_iter < 1:
            raise ValueError("Invalid capped-softmax settings")
        n_spots, n_domains = values.shape
        cap = torch.full(
            (n_domains,),
            float(capacity_factor) * n_spots / n_domains,
            dtype=self.dtype,
            device=self.device,
        )
        penalty = torch.zeros_like(cap)
        tol_abs = max(float(tol) * n_spots, 0.0)
        probabilities = None
        iterations = 0
        for iterations in range(1, int(max_iter) + 1):
            probabilities = self._torch_row_softmax((values - penalty[None, :]) / temperature)
            size = probabilities.sum(dim=0)
            violation = torch.clamp(size - cap, min=0.0)
            if float(violation.max().item()) <= tol_abs:
                break
            penalty = torch.clamp(
                penalty + float(step) * (size - cap) / torch.clamp(cap, min=1.0),
                min=0.0,
            )
        probabilities = self._torch_row_softmax((values - penalty[None, :]) / temperature)
        size = probabilities.sum(dim=0)
        violation = torch.clamp(size - cap, min=0.0)
        stats = {
            "capped_softmax_iterations": float(iterations),
            "capped_softmax_max_violation": float(violation.max().item()),
            "capped_softmax_mean_positive_violation": float(violation.mean().item()),
            "capped_softmax_active_penalties": float((penalty > 0.0).sum().item()),
            "capped_softmax_max_penalty": float(penalty.max().item()),
        }
        return probabilities.detach().cpu().numpy().astype(np.float32, copy=False), stats

    def _torch_row_softmax(self, values):
        torch = self.torch
        values = values - values.max(dim=1, keepdim=True).values
        exponentials = torch.exp(torch.clamp(values, min=-80.0, max=80.0))
        return exponentials / torch.clamp(exponentials.sum(dim=1, keepdim=True), min=1.0e-30)

    @staticmethod
    def _numpy64(value) -> np.ndarray:
        return value.detach().cpu().numpy().astype(np.float64, copy=False)
