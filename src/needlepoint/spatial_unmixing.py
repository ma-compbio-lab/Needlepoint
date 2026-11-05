"""Joint Gaussian unmixing with NeedlePoint global/local priors on latent maps.

Unlike posterior-map factorization, every optimization step evaluates the raw
frame likelihood after mixing the latent fields. All data-dependent initializers
receive training observations only. This is an experimental microscopy model,
not a replacement for the feature-wise omics GP pipeline.
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import NMF
from threadpoolctl import threadpool_limits

from needlepoint.unmixing_constraints import (
    initialize_explicit_components, outside_nuisance_fraction, distinctiveness_penalty,
)

from needlepoint.gpytorch_global_local import (
    _BatchedCachedVNNGP,
    load_or_build_vnngp_geometry,
)
from needlepoint.gpytorch_joint_global_local import (
    _initialize_local_variational_distribution,
    _model_parameter_partition,
)
from needlepoint.gpytorch_vski import (
    _BatchedGridSVGP,
    _coordinate_grid_bounds,
    _require_gpytorch,
)


@dataclass
class SpatialUnmixingSettings:
    n_components: int = 4
    factor_link: str = "identity"
    grid_size: int = 24
    local_neighbors: int = 8
    global_length_um: float = 250.0
    global_min_length_um: float = 50.0
    local_length_um: float = 40.0
    local_min_length_um: float = 6.0
    local_max_length_um: float = 150.0
    coordinate_unit_um: float = 100.0
    epochs: int = 100
    batch_size: int = 1024
    local_kl_batch_size: int = 256
    learning_rate: float = 0.02
    hyperparameter_learning_rate: float = 0.003
    loading_learning_rate: float = 0.005
    noise_floor: float = 1.0e-5
    quadrature_points: int = 12
    gradient_clip: float = 50.0
    seed: int = 384
    device: str = "cpu"
    dtype: str = "float32"
    validation_every: int = 10
    geometry_cache_dir: str | None = None
    explicit_nuisance: bool = False
    anchor_map_weight: float = 10.0
    anchor_profile_freedom: float = 0.1
    minimum_distinct_fraction: float = 0.5
    distinctness_weight: float = 0.0

    def validate(self):
        for name in ("n_components", "epochs", "batch_size", "local_neighbors",
                     "local_kl_batch_size", "quadrature_points", "validation_every"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.grid_size < 5:
            raise ValueError("grid_size must be at least 5")
        if self.factor_link not in {"identity", "softplus"}:
            raise ValueError("factor_link must be identity or softplus")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        for name in ("global_length_um", "global_min_length_um", "local_length_um",
                     "local_min_length_um", "local_max_length_um", "coordinate_unit_um",
                     "learning_rate", "hyperparameter_learning_rate",
                     "loading_learning_rate", "noise_floor", "gradient_clip"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.global_length_um > self.global_min_length_um:
            raise ValueError("global initial length must exceed its minimum")
        if not self.local_min_length_um < self.local_length_um < self.local_max_length_um:
            raise ValueError("local initial length must lie strictly inside its bounds")
        if self.explicit_nuisance and (self.factor_link != "softplus" or self.n_components < 3):
            raise ValueError("Explicit nuisance requires positive tissue factors and at least three components")
        if not 0 <= self.anchor_profile_freedom < .5:
            raise ValueError("anchor_profile_freedom must be in [0, .5)")
        if not 0 < self.minimum_distinct_fraction < 1:
            raise ValueError("minimum_distinct_fraction must be strictly between zero and one")
        if not np.isfinite(self.anchor_map_weight) or self.anchor_map_weight < 0:
            raise ValueError("anchor_map_weight must be finite and nonnegative")
        if not np.isfinite(self.distinctness_weight) or self.distinctness_weight < 0:
            raise ValueError("distinctness_weight must be finite and nonnegative")
        if self.distinctness_weight and not self.explicit_nuisance:
            raise ValueError("Distinctiveness requires explicit nuisance roles")


def factor_moments(mean, variance, link="identity", quadrature_points=12):
    """Marginals of identity/softplus applied to a Gaussian latent field."""
    import torch
    if link == "identity":
        return mean, variance
    if link != "softplus":
        raise ValueError("Unknown factor link")
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
    nodes = torch.as_tensor(nodes, device=mean.device, dtype=mean.dtype)
    weights = torch.as_tensor(weights / np.sqrt(np.pi), device=mean.device, dtype=mean.dtype)
    values = torch.nn.functional.softplus(
        mean[..., None] + (2 * variance.clamp_min(1e-12)).sqrt()[..., None] * nodes
    )
    first = (values * weights).sum(-1)
    # Centered second moment avoids cancellation for nearly deterministic maps.
    var = ((values - first[..., None]).square() * weights).sum(-1)
    return first, var


def mixed_gaussian_log_likelihood(target, factor_mean, factor_variance,
                                  loadings, offset, noise_variance):
    """Analytic E_q log likelihood; arrays use observations x factors/frames.

    Loadings are point estimates. Independent variational factor marginals
    imply Var(sum_k A_ik L_bk) = sum_k Var(A_ik) L_bk**2. The expectation is
    analytic for identity and quadrature-approximated for softplus factors.
    """
    prediction = factor_mean @ loadings.T + offset
    variance = factor_variance @ loadings.square().T
    return -0.5 * (
        np.log(2 * np.pi) + noise_variance.log()
        + ((target - prediction).square() + variance) / noise_variance
    )


def _inverse_softplus(value):
    import torch
    return value + torch.log(-torch.expm1(-value))


class SpatialUnmixingFit:
    """Fitted parameters and prediction with exact training-site marginals."""

    def loadings(self):
        import torch
        raw = (torch.nn.functional.softplus(self.raw_loadings)
               if self.settings.factor_link == "softplus" else self.raw_loadings)
        if self.settings.explicit_nuisance:
            raw = torch.cat([raw[:, :1], self.raw_loadings[:, 1:2], raw[:, 2:]], dim=1)
        normalized = raw / raw.norm(dim=0, keepdim=True).clamp_min(1e-8)
        if self.settings.explicit_nuisance:
            rho = self.settings.anchor_profile_freedom
            anchored = (1-rho)*self.anchor_profiles+rho*normalized[:, :2]
            anchored = anchored/anchored.norm(dim=0, keepdim=True).clamp_min(1e-8)
            normalized = torch.cat([anchored, normalized[:, 2:]], dim=1)
        return normalized

    def moments(self, mean, variance):
        import torch
        fm, fv = factor_moments(mean, variance, self.settings.factor_link,
                                self.settings.quadrature_points)
        if self.settings.explicit_nuisance:
            fm = torch.cat([fm[:, :1], mean[:, 1:2], fm[:, 2:]], dim=1)
            fv = torch.cat([fv[:, :1], variance[:, 1:2], fv[:, 2:]], dim=1)
        return fm, fv

    def audit_distinctiveness(self):
        import torch
        if not self.settings.explicit_nuisance:
            return {}
        pred = self.predict(self.training_coords)
        fm = torch.as_tensor(pred["factor_mean"], device=self.raw_loadings.device,
                              dtype=self.raw_loadings.dtype)
        with torch.no_grad():
            raw = outside_nuisance_fraction(fm, self.loadings())
            structural = outside_nuisance_fraction(fm, self.loadings(), center_maps=True)
            deviation = ((fm[:, :2]-self.anchor_maps)/self.anchor_map_scale).square().mean(0).sqrt()
        return dict(raw_fraction=raw.cpu().tolist(), structural_fraction=structural.cpu().tolist(),
                    minimum_fraction=float(torch.minimum(raw, structural).min()),
                    anchor_normalized_rmse=deviation.cpu().tolist(),
                    n_audited_training_sites=len(self.training_coords))

    def noise(self):
        import torch
        return torch.nn.functional.softplus(self.raw_noise) + self.settings.noise_floor

    def _offset(self, nuisance):
        return self.intercept + nuisance @ self.nuisance_loadings

    def predict(self, coords, nuisance=None, batch_size=2048):
        import torch
        coords = np.asarray(coords, dtype=np.float32)
        if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
            raise ValueError("prediction coords must be finite N x 2")
        if len(coords) == 0 or batch_size <= 0:
            raise ValueError("prediction coordinates and batch size must be nonempty/positive")
        h = self._prepare_nuisance(nuisance, len(coords))
        device, dtype = self.raw_loadings.device, self.raw_loadings.dtype
        coords_t = torch.as_tensor((coords-self.origin)/self.settings.coordinate_unit_um,
                                   dtype=dtype, device=device)
        h_t = torch.as_tensor(h, dtype=dtype, device=device)
        self.global_model.eval()
        self.local_model.eval()
        strategy = self.local_model.variational_strategy
        # Only training sites are inducing points; held-out values never enter q.
        _, nearest = self.training_tree.query(coords)
        exact = np.all(self.training_coords[nearest] == coords, axis=1)
        if not exact.all() and not getattr(self, "nn_initialized", False):
            strategy.nn_util.set_nn_idx(self.coords_t.float())
            self.nn_initialized = True
        arrays = {key: [] for key in ("factor_mean", "factor_variance", "prediction",
                                       "predictive_variance", "nuisance_prediction")}
        with torch.no_grad():
            for start in range(0, len(coords), batch_size):
                end = min(start+batch_size, len(coords))
                x = coords_t[start:end]
                global_q = self.global_model(x)
                lm = torch.empty_like(global_q.mean)
                lv = torch.empty_like(global_q.variance)
                same = exact[start:end]
                rows = np.flatnonzero(same)
                if len(rows):
                    idx = torch.as_tensor(nearest[start:end][same], device=device)
                    m, v = strategy.training_marginals(idx)
                    lm[:, rows], lv[:, rows] = m, v
                rows = np.flatnonzero(~same)
                if len(rows):
                    local_q = self.local_model(x[rows])
                    lm[:, rows], lv[:, rows] = local_q.mean, local_q.variance
                fm, fv = self.moments((global_q.mean+lm).T, (global_q.variance+lv).T)
                loadings = self.loadings()
                offset = self._offset(h_t[start:end])
                values = dict(factor_mean=fm, factor_variance=fv,
                              prediction=(fm @ loadings.T + offset)*self.scale,
                              predictive_variance=(fv @ loadings.square().T+self.noise())
                                                   * self.scale**2,
                              nuisance_prediction=offset*self.scale)
                for key, value in values.items():
                    arrays[key].append(value.cpu().numpy())
        result = {key: np.concatenate(parts) for key, parts in arrays.items()}
        if not all(np.isfinite(x).all() for x in result.values()):
            raise FloatingPointError("Non-finite posterior prediction")
        return result

    def _prepare_nuisance(self, nuisance, n):
        if nuisance is None:
            nuisance = np.empty((n, 0), dtype=np.float32)
        h = np.asarray(nuisance, dtype=np.float32)
        if h.shape != (n, len(self.nuisance_center)) or not np.isfinite(h).all():
            raise ValueError("nuisance shape/values do not match training")
        return (h-self.nuisance_center)/self.nuisance_scale

    def _state(self):
        return dict(global_model=copy.deepcopy(self.global_model.state_dict()),
                    local_model=copy.deepcopy(self.local_model.state_dict()),
                    parameters=[p.detach().clone() for p in self.parameters])

    def _restore(self, state):
        import torch
        self.global_model.load_state_dict(state["global_model"])
        self.local_model.load_state_dict(state["local_model"])
        with torch.no_grad():
            for p, value in zip(self.parameters, state["parameters"]):
                p.copy_(value)

    def save(self, path):
        import torch
        torch.save(dict(settings=asdict(self.settings), state=self._state(),
                        scale=self.scale, origin=self.origin,
                        training_coords=self.training_coords,
                        nuisance_center=self.nuisance_center,
                        nuisance_scale=self.nuisance_scale,
                        geometry_audit=self.geometry_audit,
                        anchor_profiles=getattr(self, "anchor_profiles", None),
                        anchor_maps=getattr(self, "anchor_maps", None),
                        anchor_map_scale=getattr(self, "anchor_map_scale", None),
                        history=self.history, summary=self.summary), Path(path))


def fit_spatial_unmixing(values, coords, *, settings=None, nuisance=None,
                         support_coords=None, validation=None, progress=None):
    """Fit raw observations jointly, with optional held-out model selection.

    ``validation`` is (values, coordinates, nuisance). It is used only to choose
    an epoch by RMSE. ``support_coords`` sets interpolation bounds using geometry
    only. One shared scalar rescales intensities; frames are not standardized.
    """
    torch, gpytorch = _require_gpytorch()
    s = settings or SpatialUnmixingSettings()
    s.validate()
    y = np.asarray(values, dtype=np.float32)
    xy = np.asarray(coords, dtype=np.float32)
    if y.ndim != 2 or xy.shape != (len(y), 2):
        raise ValueError("values must be N x B and coords N x 2")
    if not np.isfinite(y).all() or not np.isfinite(xy).all():
        raise ValueError("training values/coords must be finite")
    if min(y.shape) <= s.n_components or len(y) <= s.local_neighbors:
        raise ValueError("Too few observations/frames for components or neighbors")
    if np.unique(xy, axis=0).shape[0] != len(xy):
        raise ValueError("training coordinates must be unique")
    support = xy if support_coords is None else np.asarray(support_coords, dtype=np.float32)
    if support.ndim != 2 or support.shape[1] != 2 or not np.isfinite(support).all():
        raise ValueError("support coordinates must be finite N x 2")
    if len(support) == 0 or (xy.min(0)<support.min(0)).any() or (xy.max(0)>support.max(0)).any():
        raise ValueError("support bounds must include training coordinates")
    h = np.empty((len(y), 0), np.float32) if nuisance is None else np.asarray(nuisance, np.float32)
    if h.ndim != 2 or h.shape[0] != len(y) or not np.isfinite(h).all():
        raise ValueError("nuisance must be a finite N x P matrix")
    if s.explicit_nuisance and h.shape[1]:
        raise ValueError("Explicit artifact field replaces the separate nuisance covariates")
    torch.manual_seed(s.seed)
    rng = np.random.default_rng(s.seed)
    started = time.monotonic()
    fit = SpatialUnmixingFit()
    fit.settings = s
    fit.scale = float(np.sqrt(np.mean((y-y.mean(0))**2, dtype=np.float64)))
    if fit.scale < 1e-8:
        raise ValueError("Data have no variable signal")
    fit.origin = support.min(0)
    fit.nuisance_center = h.mean(0)
    fit.nuisance_scale = np.maximum(h.std(0), 1e-5)
    h = (h-fit.nuisance_center)/fit.nuisance_scale
    normalized = y/fit.scale
    # Regression is an initializer only; intercept/nuisance coefficients remain
    # learnable in the coupled likelihood, and no nuisance is hard-subtracted.
    design = np.column_stack([np.ones(len(y)), h])
    coefficient = np.linalg.lstsq(design, normalized, rcond=1e-6)[0].astype(np.float32)
    residual = normalized-design@coefficient
    if s.explicit_nuisance:
        factors, loadings, latent, intercept = initialize_explicit_components(normalized, s.n_components, s.seed)
        coefficient[0] = intercept
    elif s.factor_link == "identity":
        _, _, vt = np.linalg.svd(residual, full_matrices=False)
        loadings = vt[:s.n_components].T.copy()
        factors = residual @ loadings
        latent = factors.copy()
    else:
        # Clipping occurs only in this approximate NMF initialization. The
        # Gaussian likelihood always receives all unchanged signed observations.
        coefficient[0] = np.quantile(normalized, .01, axis=0)
        residual = normalized-design@coefficient
        nmf = NMF(n_components=s.n_components, init="nndsvdar", max_iter=300,
                  tol=1e-4, random_state=s.seed)
        # Load sklearn before the Torch GP thread pools are created, and limit
        # this small-rank initializer to one thread. Some local BLAS/OpenMP
        # combinations stall when loaded after a completed multithreaded GP fit.
        with threadpool_limits(limits=1):
            factors = nmf.fit_transform(np.maximum(residual, 0))
        loadings = nmf.components_.T
        norm = np.linalg.norm(loadings, axis=0).clip(1e-8)
        loadings = loadings/norm
        factors = factors*norm
        factors = np.maximum(factors, .02)
        latent = factors+np.log(-np.expm1(-factors))
    fit.initial_loadings = loadings.copy()
    noise_init = np.maximum(np.mean((normalized-design@coefficient-factors@loadings.T)**2, axis=0),
                            s.noise_floor*10).astype(np.float32)
    order, dag, fit.geometry_audit = load_or_build_vnngp_geometry(
        xy, s.local_neighbors, cache_dir=s.geometry_cache_dir,
        cache_mode="readwrite" if s.geometry_cache_dir else "off")
    fit.training_coords = xy[order]
    fit.training_tree = cKDTree(fit.training_coords)
    device, dtype = torch.device(s.device), getattr(torch, s.dtype)
    def tensor(x):
        return torch.as_tensor(x, device=device, dtype=dtype)
    x = tensor((xy[order]-fit.origin)/s.coordinate_unit_um)
    fit.coords_t = x
    yt, ht = tensor(normalized[order]), tensor(h[order])
    if s.explicit_nuisance:
        fit.anchor_profiles = tensor(loadings[:, :2].copy())
        fit.anchor_maps = tensor(factors[order, :2].copy())
        fit.anchor_map_scale = tensor(np.maximum(factors[:, :2].std(0), .05))
    fit.global_model = _BatchedGridSVGP.build(
        gpytorch, torch, n_features=s.n_components, grid_size=s.grid_size,
        grid_bounds=_coordinate_grid_bounds((support-fit.origin)/s.coordinate_unit_um),
        variational_distribution="meanfield", kl_backend="structured_kron",
        min_length_scale=s.global_min_length_um/s.coordinate_unit_um).to(device=device, dtype=dtype)
    fit.local_model = _BatchedCachedVNNGP.build(
        gpytorch, torch, x, torch.as_tensor(dag, device=device), n_features=s.n_components,
        n_neighbors=s.local_neighbors, training_batch_size=s.batch_size,
        min_length_scale=s.local_min_length_um/s.coordinate_unit_um,
        max_length_scale=s.local_max_length_um/s.coordinate_unit_um).to(device=device, dtype=dtype)
    g, l = fit.global_model, fit.local_model
    g.covar_module.base_kernel.lengthscale = s.global_length_um/s.coordinate_unit_um
    l.covar_module.base_kernel.lengthscale = s.local_length_um/s.coordinate_unit_um
    latent_variance = np.maximum(latent.var(0), .05)
    g.covar_module.outputscale = tensor(latent_variance*.7)
    l.covar_module.outputscale = tensor(latent_variance*.3)
    g.mean_module.constant = tensor(latent.mean(0))
    # Initialize global grid from training-only local averages, then assign the
    # remaining latent contrast to local q. Both are subsequently jointly fit.
    with torch.no_grad():
        g(x[:2])  # initialize GPyTorch variational buffers before setting q
        inducing = g.variational_strategy.inducing_points.cpu().numpy()
        d, ids = cKDTree(x.cpu().numpy()).query(inducing, k=min(32,len(x)))
        weights = np.exp(-.5*(d/(s.global_length_um/s.coordinate_unit_um))**2)
        weights /= weights.sum(1, keepdims=True).clip(1e-30)
        initial_global = (latent[order][ids]*weights[..., None]).sum(1)
        qg = g.variational_strategy._variational_distribution
        qg.variational_mean.copy_(tensor(initial_global.T))
        qg._variational_stddev.fill_(.03)
        g.variational_strategy._clear_cache()
        global_values = []
        for start in range(0,len(x),s.batch_size):
            global_values.append(g(x[start:start+s.batch_size]).mean.T.cpu().numpy())
        global_values = np.concatenate(global_values)
        _initialize_local_variational_distribution(l)
        ql = l.variational_strategy._variational_distribution
        ql.variational_mean.copy_(tensor((latent[order]-global_values).T))
        ql._variational_stddev.fill_(.03)
    initial = tensor(loadings)
    raw_loadings = _inverse_softplus(initial.clamp_min(1e-5)) if s.factor_link=="softplus" else initial
    if s.explicit_nuisance:
        raw_loadings[:, 1] = initial[:, 1]
    fit.raw_loadings = torch.nn.Parameter(raw_loadings)
    fit.intercept = torch.nn.Parameter(tensor(coefficient[0]))
    fit.nuisance_loadings = torch.nn.Parameter(tensor(coefficient[1:]))
    fit.raw_noise = torch.nn.Parameter(_inverse_softplus(tensor(noise_init)-s.noise_floor))
    fit.parameters = [fit.raw_loadings, fit.intercept, fit.nuisance_loadings, fit.raw_noise]
    gv, gh = _model_parameter_partition(g)
    lv, lh = _model_parameter_partition(l)
    optimizer = torch.optim.Adam([
        dict(params=gv+lv, lr=s.learning_rate),
        dict(params=gh+lh, lr=s.hyperparameter_learning_rate),
        dict(params=fit.parameters, lr=s.loading_learning_rate)])
    parameters = list(g.parameters())+list(l.parameters())+fit.parameters
    generator = torch.Generator(device=device).manual_seed(s.seed+918)
    fit.history = []
    best, best_state, best_epoch = np.inf, None, None
    for epoch in range(1,s.epochs+1):
        g.train(); l.train()
        permutation = rng.permutation(len(y))
        sums = np.zeros(6)
        epoch_start = time.monotonic()
        for start in range(0,len(y),s.batch_size):
            idx = torch.as_tensor(permutation[start:start+s.batch_size], device=device)
            optimizer.zero_grad(set_to_none=True)
            global_q = g(x[idx])
            lm, lv = l.variational_strategy.training_marginals(idx)
            fm, fv = fit.moments((global_q.mean+lm).T, (global_q.variance+lv).T)
            ell = mixed_gaussian_log_likelihood(yt[idx], fm, fv, fit.loadings(),
                                                fit._offset(ht[idx]), fit.noise()).sum(-1).mean()
            klg = g.variational_strategy.kl_divergence().sum()
            kll = l.variational_strategy.local_kl(mode="stochastic",
                        batch_size=s.local_kl_batch_size, generator=generator).sum()
            loss = -ell+(klg+kll)/len(y)
            anchor_penalty = torch.zeros((), device=device, dtype=dtype)
            distinct_penalty = torch.zeros((), device=device, dtype=dtype)
            if s.explicit_nuisance:
                anchor_penalty = ((fm[:, :2]-fit.anchor_maps[idx])/fit.anchor_map_scale).square().mean(0).sum()
                if s.distinctness_weight:
                    distinct_penalty, _, _ = distinctiveness_penalty(fm, fit.loadings(), s.minimum_distinct_fraction)
                loss = loss+s.anchor_map_weight*anchor_penalty+s.distinctness_weight*distinct_penalty
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite unmixing objective at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, s.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            sums += np.array([loss.item(),ell.item(),klg.item()/len(y),kll.item()/len(y),
                              anchor_penalty.item(), distinct_penalty.item()])*len(idx)
        row = dict(epoch=epoch, objective=float(sums[0]/len(y)),
                   negative_elbo=float((-sums[1]+sums[2]+sums[3])/len(y)),
                   expected_log_likelihood=float(sums[1]/len(y)),
                   global_kl_per_pixel=float(sums[2]/len(y)),
                   local_kl_per_pixel=float(sums[3]/len(y)),
                   anchor_penalty=float(sums[4]/len(y)), distinctness_penalty=float(sums[5]/len(y)),
                   epoch_seconds=time.monotonic()-epoch_start)
        if validation is not None and (epoch % s.validation_every == 0 or epoch==1 or epoch==s.epochs):
            vy, vx, vh = validation
            vp = fit.predict(vx, vh)
            row["validation_rmse"] = float(np.sqrt(np.mean((vp["prediction"]-vy)**2)))
            audit = fit.audit_distinctiveness()
            row["distinctiveness"] = audit
            feasible = not s.distinctness_weight or audit["minimum_fraction"] >= s.minimum_distinct_fraction-1e-4
            row["constraint_feasible"] = bool(feasible)
            if feasible and row["validation_rmse"] < best:
                best, best_epoch = row["validation_rmse"], epoch
                best_state = fit._state()
        fit.history.append(row)
        if progress:
            progress(row)
    if best_state is not None:
        fit._restore(best_state)
    audit = fit.audit_distinctiveness()
    if s.distinctness_weight and (audit["minimum_fraction"] < s.minimum_distinct_fraction-1e-4
                                 or (validation is not None and best_state is None)):
        raise RuntimeError(f"No feasible distinct-component fit: {audit}")
    fit.summary = dict(settings=asdict(s), n_training=len(y), n_frames=y.shape[1],
                       selected_epoch=best_epoch or s.epochs,
                       selection=("validation RMSE among constraint-feasible checkpoints" if s.distinctness_weight
                                  else "validation RMSE") if best_state else "last epoch",
                       distinctiveness=audit,
                       component_roles=(["white_matter_anchor", "boundary_artifact_anchor"]+
                                        [f"tissue_{i+1}" for i in range(2,s.n_components)])
                                       if s.explicit_nuisance else None,
                       constraint_scope="all training posterior-mean maps; raw and intercept-removed patterns",
                       anchor_source="training-only PC1/PC2 working interpretation" if s.explicit_nuisance else None,
                       training_seconds=time.monotonic()-started,
                       intensity_scale=fit.scale,
                       global_length_um=(g.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
                                         .ravel()*s.coordinate_unit_um).tolist(),
                       local_length_um=(l.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
                                        .ravel()*s.coordinate_unit_um).tolist(),
                       noise_sd_counts=(fit.noise().detach().cpu().numpy()**.5*fit.scale).tolist(),
                       fit_purpose="evaluation" if validation is not None else "experimental",
                       downstream_ready=False,
                       convergence_claim=False,
                       uncertainty="factorized variational maps; point-estimate loadings, nuisance and kernels")
    return fit
