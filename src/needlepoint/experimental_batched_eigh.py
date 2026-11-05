"""Opt-in real FP64 cuSOLVER batched Jacobi eigensolver experiment.

No global library preference or scientific default is modified. This uses the
cuSOLVER already loaded by Torch, not a separately installed CUDA library.
The public wrapper accepts symmetric matrices only; its derivative is the
symmetric-input eigendecomposition derivative, not a lower-triangle API.
"""
from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _library():
    # CUDA must have been initialized by the caller. Resolve the library the
    # actual process loaded instead of selecting an incompatible system CUDA.
    def loaded_paths():
        return {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                if '/libcusolver.so' in line}
    paths = loaded_paths()
    if not paths:
        # Torch lazily loads its CUDA linalg extension. A fixed 1x1 Cholesky
        # initializes that extension using Torch's own pinned dependencies.
        # This is a once-per-process loader, NOT an eigensolver fallback.
        import torch
        torch.linalg.cholesky_ex(torch.ones((1, 1), device='cuda', dtype=torch.float64))
        paths = loaded_paths()
    if len(paths) != 1:
        raise RuntimeError(f'Expected one loaded cuSOLVER library, found {sorted(paths)}')
    lib = ctypes.CDLL(paths.pop())
    pointer = ctypes.c_void_p
    integer = ctypes.c_int
    signatures = {
        'cusolverDnCreate': [ctypes.POINTER(pointer)],
        'cusolverDnDestroy': [pointer],
        'cusolverDnSetStream': [pointer, pointer],
        'cusolverDnCreateSyevjInfo': [ctypes.POINTER(pointer)],
        'cusolverDnDestroySyevjInfo': [pointer],
        'cusolverDnXsyevjSetTolerance': [pointer, ctypes.c_double],
        'cusolverDnXsyevjSetMaxSweeps': [pointer, integer],
        'cusolverDnDsyevjBatched_bufferSize': [pointer, integer, integer, integer,
            pointer, integer, pointer, ctypes.POINTER(integer), pointer, integer],
        'cusolverDnDsyevjBatched': [pointer, integer, integer, integer,
            pointer, integer, pointer, pointer, integer, pointer, pointer, integer],
    }
    for name, signature in signatures.items():
        function = getattr(lib, name)
        function.argtypes = signature
        function.restype = integer
    return lib


def _check(status, operation):
    if status != 0:
        raise RuntimeError(f'{operation} failed with cuSOLVER status {status}')


def jacobi_eigh_forward(matrix):
    """Checked GPU forward; no autograd. Input must be symmetric real FP64."""
    import torch
    if matrix.device.type != 'cuda' or matrix.dtype != torch.float64:
        raise ValueError('Batched Jacobi experiment requires CUDA float64 matrices')
    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError('Expected a batch of square matrices')
    batch, n, _ = matrix.shape
    if not (1 <= n <= 256) or batch < 1:
        raise ValueError('Experimental validation is restricted to 1 <= n <= 256')
    with torch.cuda.device(matrix.device):
        # cuSOLVER overwrites column-major input with eigenvector columns.
        vectors = matrix.mT.clone(memory_format=torch.contiguous_format).mT
        values = torch.empty((batch, n), dtype=matrix.dtype, device=matrix.device)
        info = torch.empty(batch, dtype=torch.int32, device=matrix.device)
        lib = _library()
        handle = ctypes.c_void_p()
        params = ctypes.c_void_p()
        try:
            _check(lib.cusolverDnCreate(ctypes.byref(handle)), 'create handle')
            _check(lib.cusolverDnSetStream(handle,
                torch.cuda.current_stream(matrix.device).cuda_stream), 'set stream')
            _check(lib.cusolverDnCreateSyevjInfo(ctypes.byref(params)), 'create params')
            _check(lib.cusolverDnXsyevjSetTolerance(params, 1e-14), 'set tolerance')
            _check(lib.cusolverDnXsyevjSetMaxSweeps(params, 100), 'set sweeps')
            size = ctypes.c_int()
            common = (handle, 1, 0, n, vectors.data_ptr(), n, values.data_ptr())
            _check(lib.cusolverDnDsyevjBatched_bufferSize(*common,
                ctypes.byref(size), params, batch), 'query workspace')
            workspace = torch.empty(size.value, dtype=matrix.dtype, device=matrix.device)
            _check(lib.cusolverDnDsyevjBatched(*common, workspace.data_ptr(),
                size.value, info.data_ptr(), params, batch), 'batched Jacobi')
            # Mandatory convergence check also ensures workspace lifetime.
            codes = info.cpu()
            if bool((codes != 0).any()):
                raise RuntimeError(f'Batched Jacobi nonconvergence: {codes.tolist()}')
            return values, vectors
        finally:
            if params.value:
                lib.cusolverDnDestroySyevjInfo(params)
            if handle.value:
                lib.cusolverDnDestroy(handle)


@lru_cache(maxsize=1)
def _autograd_function():
    import torch

    class SymmetricJacobiEigh(torch.autograd.Function):
        @staticmethod
        def forward(ctx, matrix):
            values, vectors = jacobi_eigh_forward(matrix)
            ctx.save_for_backward(values, vectors)
            return values, vectors

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, grad_values, grad_vectors):
            values, vectors = ctx.saved_tensors
            result = torch.diag_embed(grad_values)
            if grad_vectors is not None:
                projected = vectors.mT @ grad_vectors
                antisymmetric = (projected - projected.mT) * 0.5
                gaps = values.unsqueeze(-2) - values.unsqueeze(-1)
                gaps.diagonal(dim1=-2, dim2=-1).fill_(1.)
                # Do not silently regularize repeated-eigenvalue derivatives.
                # The caller must reject nonfinite gradients, as in Torch.
                result = result + antisymmetric / gaps
            return vectors @ result @ vectors.mT

    return SymmetricJacobiEigh


def jacobi_eigh(matrix):
    """First-order differentiable symmetric eigensolver (experimental)."""
    return _autograd_function().apply(matrix)
