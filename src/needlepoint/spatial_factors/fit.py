"""Minibatched observation-level inference, exports, and checkpoint replay."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
import torch

from ..artifacts import sha256_file
from ..gpytorch_joint_global_local import JointGlobalLocalGPSettings, _validate_joint_settings
from .likelihoods import make_observation
from .model import GlobalLocalFactorModel, _finite, _selection, factor_moments
from .settings import SpatialFactorSettings


class AlignedArray:
    """Broadcast metadata without allocating an N x J copy.

    A one-dimensional array always indexes observations; feature-only metadata
    must have shape (1, J). Both location and feature gathers stay batch sized.
    """

    def __init__(self, value, shape, name, *, nonnegative=False, boolean=False):
        self.base = np.asarray(value)
        if boolean and self.base.dtype != bool:
            raise ValueError(f"{name} must be explicitly boolean")
        if not boolean and (not np.isfinite(self.base).all()
                            or (nonnegative and np.any(self.base < 0))):
            raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
        if self.base.ndim == 1:
            self.base = self.base[:, None]
        try:
            self.value = np.broadcast_to(self.base, shape)
        except ValueError as exc:
            raise ValueError(f"{name} must broadcast to {shape}; vectors index observations") from exc

    def block(self, rows, features):
        columns = np.arange(self.value.shape[1])[features]
        return self.value[np.asarray(rows)[:, None], columns]

    def tensor(self, rows, features, model):
        dtype = torch.bool if self.base.dtype == bool else model.dtype
        return torch.tensor(self.block(rows, features).copy(), dtype=dtype, device=model.device)


def _block(values, rows, features):
    if sparse.issparse(values):
        return values[rows, features].toarray()
    return values[np.asarray(rows)[:, None], np.arange(values.shape[1])[features]]


def _fingerprint(values, coordinates, names, arrays):
    digest = hashlib.sha256()
    sequence = ([values.data, values.indices, values.indptr] if sparse.issparse(values) else [values])
    sequence += [coordinates] + [a.base for a in arrays if a is not None]
    digest.update(json.dumps(dict(shape=values.shape, features=names.tolist()), sort_keys=True).encode())
    for array in sequence:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _global_initial_values(model, latent):
    """Initialize the broad field at its kernel scale, before local residuals.

    Nearest-site sampling aliases sparse NMF zeros into isolated grid pits.
    Cubic interpolation turns those pits into repeated footprints. Smooth only
    the initial Gaussian grid mean; subsequent residual initialization preserves
    the supplied factor log means at every training observation.
    """
    points = model.global_model.variational_strategy.inducing_points.detach().cpu().numpy()
    _, nearest = cKDTree(model.coordinates.detach().cpu().numpy()).query(points, k=1)
    values = latent[nearest, :model.n_spatial].copy()
    if model.settings.global_initialization == "kernel_smooth":
        axes = model.global_model.variational_strategy.grid.detach().cpu().numpy()
        size = len(axes)
        # Grid inducing values and the structured KL use x-fast ordering.
        if not (np.allclose(points[:, 0].reshape(size, size), axes[None, :, 0])
                and np.allclose(points[:, 1].reshape(size, size), axes[:, None, 1])):
            raise RuntimeError("Unexpected global inducing-grid order")
        dx, dy = axes[1] - axes[0]
        length = float(model.global_model.covar_module.base_kernel.lengthscale.detach().flatten()[0])
        values = gaussian_filter(values.reshape(size, size, model.n_spatial),
            sigma=(length / dy, length / dx, 0), mode="nearest").reshape(-1, model.n_spatial)
    return values


def _initialize(model, values, exposure):
    """NMF/SVD initializer only: the likelihood always sees original values."""
    from sklearn.decomposition import NMF, TruncatedSVD

    config = model.settings
    rng = np.random.default_rng(config.random_state)
    n, j, k = model.n_training, model.n_features, model.n_factors
    sample_rows = np.sort(rng.choice(n, min(n, config.initialization_spots), replace=False))
    sample = _block(values, sample_rows, slice(None)).astype(float)
    ex = exposure.block(sample_rows, slice(None))
    sample = np.divide(sample, ex, out=np.zeros_like(sample), where=ex > 0)
    positive = config.nonnegative
    use_decomposition = config.initialization == "nmf" and np.any((np.maximum(sample, 0) if positive else sample) != 0)
    if use_decomposition and (positive or (k <= min(sample.shape) and j > 1)):
        if positive:
            decomposition = NMF(n_components=k, init="nndsvda" if k <= min(sample.shape) else "random",
                max_iter=config.initialization_iterations, random_state=config.random_state)
            decomposition.fit(np.maximum(sample, 0))
        else:
            decomposition = TruncatedSVD(n_components=k, random_state=config.random_state).fit(sample)
        weights = decomposition.components_.T.copy()
        activity = np.empty((n, k))
        for start in range(0, n, model.gp_settings.batch_size_spots):
            rows = np.arange(start, min(n, start + model.gp_settings.batch_size_spots))
            block = _block(values, rows, slice(None)).astype(float)
            ex = exposure.block(rows, slice(None))
            block = np.divide(block, ex, out=np.zeros_like(block), where=ex > 0)
            activity[rows] = decomposition.transform(np.maximum(block, 0) if positive else block)
        if positive:
            # Normalize initial factor mean to one without changing its product.
            scale = np.maximum(activity.mean(0), 1e-4)
            activity /= scale
            weights *= scale
            activity = np.maximum(activity, 1e-4)
            weights = np.maximum(weights, 1e-5)
        # Assign the most spatially autocorrelated initial factors to the GP.
        coords = model.coordinates.detach().cpu().numpy()
        _, neighbors = cKDTree(coords).query(coords, k=min(7, n))
        centered = activity - activity.mean(0)
        score = (centered * centered[neighbors[:, 1:]].mean(1)).sum(0) / np.maximum(
            (centered**2).sum(0), 1e-12)
        order = np.argsort(-score, kind="stable")
        activity, weights = activity[:, order], weights[:, order]
    else:
        activity = np.exp(rng.normal(0, 0.1, (n, k))) if positive else rng.normal(0, 0.1, (n, k))
        if positive:
            weights = np.maximum(sample.mean(0), 0.01)[:, None] / k
            weights = np.broadcast_to(weights, (j, k)).copy() * rng.uniform(0.8, 1.2, (j, k))
        else:
            weights = rng.normal(0, max(float(sample.std()), 0.1) / np.sqrt(k), (j, k))
    latent = np.log(activity) if positive else activity
    with torch.no_grad():
        raw = weights + np.log(-np.expm1(-weights)) if positive else weights
        model.raw_loadings.copy_(torch.tensor(raw, dtype=model.dtype, device=model.device))
        model.nonspatial_mean.copy_(torch.tensor(latent[:, model.n_spatial:], dtype=model.dtype, device=model.device))
        if config.learn_nonspatial_prior:
            model.nonspatial_prior_mean.copy_(model.nonspatial_mean.mean(0))
        gq = model.global_model.variational_strategy._variational_distribution
        gq.variational_mean.copy_(torch.tensor(_global_initial_values(model, latent).T,
                                              dtype=model.dtype, device=model.device))
        model.global_model.mean_module.constant.copy_(torch.tensor(latent[:, :model.n_spatial].mean(0),
                                                                  dtype=model.dtype, device=model.device))
        model.global_model.train()
        global_mean = []
        for x in model.coordinates.split(model.gp_settings.eval_batch_size_spots):
            global_mean.append(model.global_model(x).mean.T)
        residual = torch.tensor(latent[:, :model.n_spatial], dtype=model.dtype, device=model.device) - torch.cat(global_mean)
        lq = model.local_model.variational_strategy._variational_distribution
        lq.variational_mean.copy_(residual[model.order].T)
        model.global_model.train()


@torch.no_grad()
def _factor_arrays(model):
    model.eval()
    arrays = {key: [] for key in ("factors", "factor_variance", "log_factor_mean", "log_factor_variance",
                                "global_mean", "global_variance", "local_mean", "local_variance")}
    for indices in torch.arange(model.n_training, device=model.device).split(model.gp_settings.eval_batch_size_spots):
        mean, variance, parts = model.training_moments(indices)
        fm, fv = factor_moments(mean, variance, model.settings.nonnegative)
        for key, value in zip(arrays, (fm, fv, mean, variance, *parts)):
            arrays[key].append(_finite(key, value).cpu().numpy())
    return {key: np.concatenate(value) for key, value in arrays.items()}


@dataclass
class SpatialFactorResult:
    model: GlobalLocalFactorModel
    feature_names: np.ndarray
    history: pd.DataFrame
    arrays: dict
    metadata: dict
    exposure: AlignedArray
    threshold: AlignedArray | None = None

    @property
    def factors(self):
        return self.arrays["factors"]

    @property
    def factor_variance(self):
        return self.arrays["factor_variance"]

    @property
    def coordinates(self):
        return self.model.native_coordinates.detach().cpu().numpy()

    @property
    def loadings(self):
        return self.model.loadings().detach().cpu().numpy()

    @property
    def feature_mixtures(self):
        if not self.model.settings.nonnegative:
            raise ValueError("Simplex interpretation requires nonnegative factors")
        contribution = self.loadings * self.factors.sum(0)
        return contribution / contribution.sum(1, keepdims=True)

    @property
    def feature_spatial_importance(self):
        return self.feature_mixtures[:, :self.model.n_spatial].sum(1)

    @property
    def observation_spatial_importance(self):
        if not self.model.settings.nonnegative:
            raise ValueError("Spatial importance requires nonnegative factors")
        contribution = self.factors * self.loadings.sum(0)
        return contribution[:, :self.model.n_spatial].sum(1) / contribution.sum(1)

    def predict(self, coordinates=None, **kwargs):
        # Known training exposure/threshold follows the same exact row/feature
        # selection; novel coordinates require explicitly supplied new metadata.
        if coordinates is None:
            rows = _selection(kwargs.get("indices"), self.model.n_training, "indices")
            features = _selection(kwargs.get("features"), self.model.n_features, "features")
            if "exposure" not in kwargs:
                kwargs["exposure"] = self.exposure.block(rows, features)
            if "threshold" not in kwargs and self.threshold is not None:
                kwargs["threshold"] = self.threshold.block(rows, features)
        return self.model.predict(coordinates, **kwargs)

    def save(self, output_dir):
        """Write a new, self-contained experimental artifact; refuse overwrite."""
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "factors.npz", **self.arrays, loadings=self.loadings,
            coordinates=self.coordinates, feature_names=self.feature_names,
            exposure=self.exposure.base,
            **({} if self.threshold is None else {"threshold": self.threshold.base}))
        self.history.to_csv(output / "history.csv", index=False)
        # Save only tensors, never the dynamically defined GPyTorch classes.
        torch.save({key: value.detach().cpu() for key, value in self.model.state_dict().items()},
                   output / "model.pt")
        metadata = dict(self.metadata, settings=asdict(self.model.settings),
            gp_settings=asdict(self.model.gp_settings), observation=self.model.observation.config())
        metadata["files"] = {name: sha256_file(output / name)
                             for name in ("factors.npz", "history.csv", "model.pt")}
        (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return output


def fit_spatial_factors(values, coordinates=None, *, likelihood="poisson", settings=None,
                        gp_settings=None, feature_names=None, exposure=1.0,
                        exposure_semantics="unit", is_censored=None, threshold=None,
                        measurement_scale=None, output_dir=None, epoch_callback=None):
    """Fit shared spatial/nonspatial factors directly to a matrix or SpatialData.

    No normalization, standardization, feature filtering, or row removal occurs.
    Counts must be raw integers, with known exposure supplied explicitly. A
    Tobit fit additionally requires measurement_scale, flags, and thresholds.
    Features are evaluated in bounded blocks, with shared sample gradients
    accumulated before a single backward pass through the GP factor graph.
    ``epoch_callback(model, record)`` optionally observes each completed epoch
    for progress logging or tensor checkpointing; it must not mutate the model.
    """
    from ..inputs import SpatialData

    settings = settings or SpatialFactorSettings()
    gp_settings = gp_settings or JointGlobalLocalGPSettings()
    if epoch_callback is not None and not callable(epoch_callback):
        raise ValueError("epoch_callback must be callable")
    _validate_joint_settings(gp_settings)
    measurement = None
    if isinstance(values, SpatialData):
        if coordinates is not None:
            raise ValueError("SpatialData already provides coordinates")
        if feature_names is not None and not np.array_equal(np.asarray(feature_names, dtype=str), values.genes):
            raise ValueError("SpatialData already provides feature_names")
        measurement = values.measurement
        coordinates, feature_names, values = values.coords, values.genes, values.counts
    if coordinates is None:
        raise ValueError("coordinates are required for a matrix input")
    values = values.tocsr(copy=True) if sparse.issparse(values) else np.asarray(values)
    if sparse.issparse(values):
        values.sum_duplicates()
        values.sort_indices()
    if values.ndim != 2 or min(values.shape) < 1:
        raise ValueError("values must be a nonempty N x J matrix")
    n, j = values.shape
    coordinates = np.asarray(coordinates, float)
    if coordinates.shape != (n, 2) or not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be a finite N x 2 matrix matching observations")
    names = np.asarray([f"feature_{i}" for i in range(j)] if feature_names is None else feature_names, dtype=str)
    if names.shape != (j,) or len(set(names)) != j or np.any(names == ""):
        raise ValueError("feature_names must be nonempty, unique and match matrix columns")
    observation = make_observation(likelihood, j)
    if measurement == "continuous" and observation.name in {"poisson", "negative-binomial"}:
        raise ValueError("Count likelihoods require SpatialData measurement='counts'")
    exp_array = AlignedArray(exposure, (n, j), "exposure", nonnegative=True)
    if not isinstance(exposure_semantics, str) or not exposure_semantics.strip():
        raise ValueError("exposure_semantics must be a nonempty description")
    if (exposure_semantics == "unit" or not observation.uses_exposure) and np.any(exp_array.base != 1):
        raise ValueError("Nonunit exposure requires a count likelihood and explicit exposure_semantics")
    flags = None if is_censored is None else AlignedArray(is_censored, (n, j), "is_censored", boolean=True)
    thresholds = None if threshold is None else AlignedArray(threshold, (n, j), "threshold")
    if observation.requires_censoring:
        if flags is None or thresholds is None or not isinstance(measurement_scale, str) or not measurement_scale.strip():
            raise ValueError("Tobit requires is_censored, threshold and explicit measurement_scale")
    elif flags is not None or thresholds is not None:
        raise ValueError("Censoring metadata requires a censoring likelihood")
    # Validate before constructing GP geometry or starting optimization.
    for start in range(0, n, gp_settings.batch_size_spots):
        rows = np.arange(start, min(n, start + gp_settings.batch_size_spots))
        for first in range(0, j, settings.feature_batch_size):
            features = slice(first, min(j, first + settings.feature_batch_size))
            observation.validate(_block(values, rows, features), exposure=exp_array.block(rows, features),
                is_censored=None if flags is None else flags.block(rows, features),
                threshold=None if thresholds is None else thresholds.block(rows, features))
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    # fork_rng avoids changing the caller's Torch random stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(settings.random_state)
        model = GlobalLocalFactorModel(coordinates, j, settings, gp_settings, observation)
    _initialize(model, values, exp_array)
    kernel_ids = {id(p) for component in (model.global_model, model.local_model)
                  for p in component.covar_module.parameters()}
    general, kernel = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (kernel if id(parameter) in kernel_ids else general).append(parameter)
    optimizer = torch.optim.Adam([{"params": general, "lr": settings.learning_rate},
                                  {"params": kernel, "lr": settings.hyperparameter_learning_rate}])
    generator = torch.Generator(device=model.device).manual_seed(settings.random_state)
    rng = np.random.default_rng(settings.random_state)
    history = []
    parameters = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(settings.epochs):
        model.train()
        permutation = rng.permutation(n)
        totals = np.zeros(6)
        for start in range(0, n, gp_settings.batch_size_spots):
            rows = permutation[start:start + gp_settings.batch_size_spots]
            indices = torch.tensor(rows, dtype=torch.long, device=model.device)
            optimizer.zero_grad(set_to_none=True)
            mean, variance, _ = model.training_moments(indices)
            samples = model.sample_factors(mean, variance, settings.n_samples, generator)
            sample_leaf = samples.detach().requires_grad_()
            negative_likelihood = 0.0
            for first in range(0, j, settings.feature_batch_size):
                features = slice(first, min(j, first + settings.feature_batch_size))
                predictor = _finite("mixture predictor", sample_leaf @ model.loadings(features).T)
                target = torch.tensor(_block(values, rows, features), dtype=model.dtype, device=model.device)
                log_prob = observation.log_prob(target, predictor, features=features,
                    exposure=exp_array.tensor(rows, features, model),
                    is_censored=None if flags is None else flags.tensor(rows, features, model),
                    threshold=None if thresholds is None else thresholds.tensor(rows, features, model))
                if log_prob.shape != predictor.shape:
                    raise ValueError("FactorObservation.log_prob must return [sample, location, feature]")
                loss = -_finite("observation log likelihood", log_prob).mean(0).sum() / len(rows)
                loss.backward()
                negative_likelihood += float(loss.detach())
            global_kl, local_kl, h_kl = model.kl_terms(indices, generator)
            # Global and local KLs count every spatial latent once, regardless
            # of feature count/chunking. Nonspatial KL is observation indexed.
            penalty = ((global_kl + local_kl) / n + h_kl / len(rows)
                       + settings.loading_l1 * model.loadings().sum() / n)
            _finite("KL/regularization", penalty)
            torch.autograd.backward((samples, penalty), (sample_leaf.grad, torch.ones_like(penalty)))
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            for parameter in parameters:
                _finite("model parameter", parameter)
            totals += len(rows) * np.asarray([negative_likelihood + float(penalty.detach()),
                -negative_likelihood, float(global_kl.detach()), float(local_kl.detach()),
                float(h_kl.detach()) / len(rows), float(norm)])
        history.append(dict(epoch=epoch + 1, negative_elbo=totals[0] / n,
            expected_log_likelihood=totals[1] / n, global_kl=totals[2] / n,
            local_kl=totals[3] / n, nonspatial_kl_per_observation=totals[4] / n,
            gradient_norm=totals[5] / n))
        if epoch_callback is not None:
            epoch_callback(model, dict(history[-1]))
    metadata = dict(schema_version=1, method="global-local-spatial-factor-vi", experimental=True,
        posterior="meanfield_across_factors_and_global_local_nonspatial",
        training="fixed_epoch_cap", convergence_verified=False,
        exposure_semantics=exposure_semantics, measurement_scale=measurement_scale,
        likelihood=observation.name, n_observations=n, n_features=j,
        input_sha256=_fingerprint(values, coordinates, names, [exp_array, flags, thresholds]),
        geometry=model.geometry_metadata,
        references=["https://arxiv.org/abs/2110.06122", "https://github.com/willtownes/nsf-paper"],
        limitations=["Point-estimated loadings and likelihood parameters",
                     "Mean-field variational uncertainty is not calibrated",
                     "No automatic selection of factor count or convergence certification"])
    result = SpatialFactorResult(model, names, pd.DataFrame(history), _factor_arrays(model),
                                 metadata, exp_array, thresholds)
    if output_dir is not None:
        result.save(output_dir)
    return result


def load_spatial_factors(output_dir, *, device="cpu", likelihood=None):
    """Verify file digests and rebuild a saved factor model without raw data."""
    root = Path(output_dir)
    metadata = json.loads((root / "manifest.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("method") != "global-local-spatial-factor-vi":
        raise ValueError("Unsupported spatial factor artifact")
    required = ("factors.npz", "history.csv", "model.pt")
    for name in required:
        if metadata.get("files", {}).get(name) != sha256_file(root / name):
            raise ValueError(f"Artifact checksum mismatch: {name}")
    settings = SpatialFactorSettings.from_saved(metadata["settings"])
    gp_settings = replace(JointGlobalLocalGPSettings(**metadata["gp_settings"]), device=device,
                          geometry_cache_dir=None, geometry_cache_mode="off")
    with np.load(root / "factors.npz", allow_pickle=False) as saved:
        coordinates, names = saved["coordinates"], saved["feature_names"]
        exposure = AlignedArray(saved["exposure"], (len(coordinates), len(names)), "exposure", nonnegative=True)
        threshold = None if "threshold" not in saved else AlignedArray(
            saved["threshold"], (len(coordinates), len(names)), "threshold")
        arrays = {key: saved[key] for key in ("factors", "factor_variance", "log_factor_mean", "log_factor_variance",
                    "global_mean", "global_variance", "local_mean", "local_variance")}
    observation = make_observation(metadata["observation"] if likelihood is None else likelihood, len(names))
    with torch.random.fork_rng(devices=[]):
        model = GlobalLocalFactorModel(coordinates, len(names), settings, gp_settings, observation)
    model.load_state_dict(torch.load(root / "model.pt", map_location=model.device, weights_only=True))
    model.eval()
    return SpatialFactorResult(model, names, pd.read_csv(root / "history.csv"), arrays, metadata, exposure, threshold)
