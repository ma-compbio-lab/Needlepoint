"""Opt-in, all-observation SKI screening with deterministic grid-space LU.

This uses the validated determinant-lemma/Woodbury implementation. It replaces
the historical stochastic log determinant, not the SKI covariance or support.
Grid256 is intentionally not enabled by this grid128 production profile.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from needlepoint.experimental_direct_ski import (
    direct_observation_residual,
    direct_ski_reference,
)
from needlepoint.gp_initialization import (
    InitialLengthScaleScreenSettings,
    _ski_rbf_operator,
    _standardize_columns,
    _validate_screen_inputs,
    merge_initial_length_scale_screen_tables,
    resolve_initial_length_scales,
    scale_fraction_column,
    scale_loglik_column,
)

BACKEND = "ski-direct-lu-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _atomic_npz(path, **values):
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez(handle, **values)
    os.replace(temporary, path)


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _identity(expression, coords, genes, settings, residual_tolerance, provenance):
    arrays = ([expression.data, expression.indices, expression.indptr]
              if sparse.issparse(expression) else [expression])
    value = dict(backend=BACKEND, shape=list(expression.shape),
                 expression_sha256=[_array_hash(x) for x in arrays],
                 coordinate_sha256=_array_hash(coords), genes=list(map(str, genes)),
                 settings=vars(settings), residual_tolerance=residual_tolerance,
                 provenance=provenance or {})
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _load_checkpoint(path, identity):
    record_path = path.with_suffix(".json")
    if not record_path.exists():
        return None
    record = json.loads(record_path.read_text())
    if record.get("identity") != identity or record.get("sha256") != file_sha256(path):
        raise ValueError(f"Incompatible or corrupted screen checkpoint: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _save_checkpoint(path, identity, **values):
    _atomic_npz(path, **values)
    atomic_json(path.with_suffix(".json"), dict(identity=identity, sha256=file_sha256(path)))


def screen_direct_ski(expression, coords, genes, settings=None, *, checkpoint_dir,
                      residual_tolerance=1e-6, residual_policy="warn", dense_column_chunk=256,
                      max_workspace_gib=24., provenance=None):
    """Return a canonical initializer table, retaining every row and feature.

    The CPU builds exact sufficient statistics in bounded feature chunks;
    CUDA performs grid-kernel products, FP64 LU, solves and residual audits.
    A fingerprinted checkpoint is written after each of all scale/fraction
    candidates. Failed candidates never disappear from the selection grid.
    """
    import gpytorch
    import torch

    settings = settings or InitialLengthScaleScreenSettings()
    if not 4 <= settings.grid_size <= 128:
        raise ValueError("Direct-SKI production profile requires grid_size in [4, 128]")
    if settings.dtype != "float32":
        raise ValueError("Validated direct-SKI profile uses float32 kernel construction and FP64 solves")
    if residual_tolerance <= 0 or dense_column_chunk <= 0 or max_workspace_gib <= 0:
        raise ValueError("Positive residual tolerance, column chunk and memory budget required")
    if residual_policy not in ("warn", "error"):
        raise ValueError("residual_policy must be warn or error")
    coords = np.asarray(coords, dtype=np.float32)
    genes = np.asarray(genes).astype(str)
    expression = expression.tocsr() if sparse.issparse(expression) else np.asarray(expression)
    scales, generation = resolve_initial_length_scales(coords, settings)
    _validate_screen_inputs(expression, coords, genes, settings, scales=scales)
    if len(set(genes)) != len(genes) or not np.isfinite(coords).all():
        raise ValueError("Unique features and finite coordinates required")
    n, p = expression.shape
    fractions = np.asarray(settings.spatial_fractions, dtype=np.float64)
    device = torch.device(settings.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = _identity(expression, coords, genes, settings, residual_tolerance,
                         dict(provenance or {}, residual_policy=residual_policy))
    identity_path = root / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text())["identity"] != identity:
        raise ValueError("Screen checkpoint input/settings/source fingerprint changed")
    atomic_json(identity_path, dict(identity=identity, backend=BACKEND, n_spots=n, n_genes=p))
    started = time.monotonic()

    def progress(stage, **values):
        record = dict(stage=stage, backend=BACKEND, pid=os.getpid(),
                      updated_epoch=time.time(), elapsed_seconds=time.monotonic()-started, **values)
        atomic_json(root / "progress.json", record)
        print(json.dumps(record), flush=True)

    with torch.no_grad():
        progress("geometry")
        coords_t = torch.as_tensor(coords, device=device)
        first = _ski_rbf_operator(torch, gpytorch, coords_t, float(scales[0]), settings.grid_size).evaluate_kernel()
        indices = first.left_interp_indices.cpu().numpy()
        values = first.left_interp_values.cpu().numpy().astype(np.float64)
        m = first.base_linear_op.shape[-1]
        w = sparse.csr_matrix((values.ravel(), indices.ravel(),
                              np.arange(n+1, dtype=np.int64)*indices.shape[1]), shape=(n, m))
        w.eliminate_zeros()
        gram = (w.T @ w).tocsr()
        gram.eliminate_zeros()
        active = np.flatnonzero(gram.diagonal() > 0)
        ma = len(active)
        # Conservative live allocation estimate: grid products, LU workspaces,
        # all-feature RHS/residual buffers and interpolation state.
        estimate = (6*ma*ma*8 + 8*m*p*8 + n*16*16) / 1024**3
        if estimate > max_workspace_gib:
            raise MemoryError(f"Direct-SKI estimated workspace {estimate:.2f} GiB exceeds {max_workspace_gib:.2f}")
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            if estimate*1024**3 > free*.90:
                raise MemoryError("Insufficient free GPU memory for the direct-SKI workspace")
        progress("sufficient_statistics", active_grid_size=ma, estimated_workspace_gib=estimate)
        stats_path = root / "statistics.npz"
        sufficient = _load_checkpoint(stats_path, identity)
        if sufficient is None:
            h = np.empty((m, p), dtype=np.float64)
            q = np.empty(p, dtype=np.float64)
            means = np.empty(p, dtype=np.float32)
            stds = np.empty(p, dtype=np.float32)
            for start in range(0, p, settings.feature_chunk_size):
                stop = min(start+settings.feature_chunk_size, p)
                raw = expression[:, start:stop]
                raw = raw.toarray() if sparse.issparse(raw) else raw
                y, means[start:stop], stds[start:stop] = _standardize_columns(np.asarray(raw, dtype=np.float32))
                y64 = y.astype(np.float64)
                h[:, start:stop] = w.T @ y64
                q[start:stop] = np.square(y64).sum(0)
                progress("sufficient_statistics", completed_features=stop, total_features=p)
            sufficient = dict(projection=h, squared_norm=q, means=means, stds=stds)
            _save_checkpoint(stats_path, identity, **sufficient)
        h_np, q_np = sufficient["projection"], sufficient["squared_norm"]
        if h_np.shape != (m, p) or q_np.shape != (p,) or not np.isfinite(h_np).all() or not np.isfinite(q_np).all():
            raise ValueError("Invalid sufficient-statistic checkpoint")
        del w, values, indices
        gram_coo = gram.tocoo()
        g = torch.sparse_coo_tensor(np.vstack([gram_coo.row, gram_coo.col]), gram_coo.data,
                                   gram.shape, dtype=torch.float64, device=device).coalesce()
        h = torch.as_tensor(h_np, device=device)
        q = torch.as_tensor(q_np, device=device)
        active_t = torch.as_tensor(active, device=device)
        table = pd.DataFrame(dict(gene=genes, feature_position=np.arange(p),
            standardization_mean=sufficient["means"], standardization_std=sufficient["stds"],
            null_loglik=-.5*n*(np.log(2*np.pi*np.maximum(q_np/n, 1e-8))+1),
            backend=BACKEND, n_screen_spots=n))
        maximum_residual = 0.
        residual_warnings = []
        for si, scale in enumerate(scales):
            records = [_load_checkpoint(root/f"s{si:03d}_f{fi:03d}.npz", identity)
                       for fi in range(len(fractions))]
            if any(record is None for record in records):
                progress("kernel_grid_product", scale_index=si, total_scales=len(scales))
                op32 = _ski_rbf_operator(torch, gpytorch, coords_t, float(scale), settings.grid_size).evaluate_kernel()
                if (not torch.equal(op32.left_interp_indices, first.left_interp_indices)
                        or not torch.equal(op32.left_interp_values, first.left_interp_values)):
                    raise ValueError("Interpolation geometry changed across candidate scales")
                op = op32.to(dtype=torch.float64)
                kg = torch.empty((ma, ma), dtype=torch.float64, device=device)
                for start in range(0, ma, dense_column_chunk):
                    stop = min(start+dense_column_chunk, ma)
                    col = torch.as_tensor(gram[:, active[start:stop]].toarray(), device=device)
                    kg[:, start:stop] = op.base_linear_op.matmul(col)[active_t]
                kh = op.base_linear_op.matmul(h)[active_t]
            for fi, fraction in enumerate(fractions):
                path = root/f"s{si:03d}_f{fi:03d}.npz"
                record = records[fi]
                if record is None:
                    progress("candidate", scale_index=si, fraction_index=fi,
                             completed=si*len(fractions)+fi, total=len(scales)*len(fractions))
                    diagonal = 1-float(fraction)+settings.jitter
                    result = direct_ski_reference(kg, kh, h[active_t], q,
                        n_observations=n, fraction=float(fraction), diagonal=diagonal)
                    # Audit every feature against the original observation operator.
                    residuals = []
                    for start in range(0, p, settings.feature_chunk_size):
                        stop = min(start+settings.feature_chunk_size, p)
                        u = torch.zeros((m, stop-start), device=device, dtype=torch.float64)
                        u[active_t] = result["grid_solution"][:, start:stop]
                        residual = direct_observation_residual(op.base_linear_op.matmul,
                            lambda v: torch.sparse.mm(g, v), h[:, start:stop], q[start:stop], u,
                            fraction=float(fraction), diagonal=diagonal)
                        residuals.append(residual.cpu().numpy())
                    record = dict(loglik=(-.5*(result["inv_quad"]+result["logdet"]+n*math.log(2*math.pi))).cpu().numpy(),
                                  residual=np.concatenate(residuals),
                                  grid_residual=result["grid_relative_residual"].cpu().numpy(),
                                  logdet=np.asarray(float(result["logdet"])))
                    del result, u
                    records[fi] = record
                warning = _validate_candidate(record, p, residual_tolerance, residual_policy)
                if warning is not None:
                    warning.update(scale_index=si, fraction_index=fi,
                                   length_scale=float(scale), spatial_fraction=float(fraction))
                    residual_warnings.append(warning)
                    atomic_json(root / "residual_warnings.json", residual_warnings)
                    progress("residual_warning", **warning)
                if not path.with_suffix(".json").exists():
                    _save_checkpoint(path, identity, **record)
                maximum_residual = max(maximum_residual, float(record["residual"].max()))
            ll = np.stack([record["loglik"] for record in records])
            best = ll.argmax(0)  # Stable first-fraction tie policy.
            table[scale_loglik_column(float(scale))] = ll[best, np.arange(p)]
            table[scale_fraction_column(float(scale))] = fractions[best]
            if "kg" in locals():
                del kg, kh, op, op32, col
            progress("scale_complete", completed_scales=si+1, total_scales=len(scales),
                     max_original_operator_residual=maximum_residual)
        scores = merge_initial_length_scale_screen_tables([table], length_scales=scales,
            elapsed_seconds=time.monotonic()-started, device=str(device))
        summary = dict(backend=BACKEND, n_spots=n, n_features=p, device=str(device),
            length_scales=scales.tolist(), length_scale_generation=generation,
            spatial_fractions=fractions.tolist(), settings=vars(settings),
            n_candidates=len(scales)*len(fractions), active_grid_size=ma,
            max_original_operator_residual=maximum_residual, residual_tolerance=residual_tolerance,
            residual_policy=residual_policy, residual_warnings=residual_warnings,
            n_residual_warnings=len(residual_warnings),
            residual_semantics="relative for nonzero standardized targets; absolute for exact-zero targets",
            n_zero_standardized_targets=int(np.sum(q_np == 0)),
            all_observations_retained=True, kernel_construction_dtype="float32",
            solve_dtype="float64", logdet_method="determinant-lemma-pivoted-LU",
            historical_slq_equivalence=False, checkpoint_identity=identity,
            elapsed_seconds=time.monotonic()-started)
        progress("complete", **{k: summary[k] for k in ("n_candidates", "n_features")})
        return scores, summary


def _validate_candidate(record, n_features, tolerance, residual_policy="warn"):
    import warnings
    if residual_policy not in ("warn", "error"):
        raise ValueError("residual_policy must be warn or error")
    for key in ("loglik", "residual", "grid_residual"):
        values = np.asarray(record[key])
        if values.shape != (n_features,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid complete candidate {key}")
    if np.asarray(record["logdet"]).shape != () or not np.isfinite(record["logdet"]):
        raise ValueError("Invalid complete candidate logdet")
    if any(np.any(np.asarray(record[key]) < 0) for key in ("residual", "grid_residual")):
        raise ValueError("Invalid negative candidate residual")
    observation = float(np.max(record["residual"]))
    grid = float(np.max(record["grid_residual"]))
    if max(observation, grid) > tolerance:
        message = (f"Direct-SKI candidate exceeds residual tolerance: observation={observation:.9g}, "
                   f"grid={grid:.9g}, tolerance={tolerance:.9g}")
        if residual_policy == "error":
            raise FloatingPointError(message)
        warnings.warn(message + "; continuing with this candidate", RuntimeWarning, stacklevel=2)
        return dict(observation_residual=observation, grid_residual=grid,
                    tolerance=tolerance, policy=residual_policy)
    return None
