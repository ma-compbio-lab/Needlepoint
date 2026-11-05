"""Training settings; physical GP geometry is configured separately."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SpatialFactorSettings:
    """Fixed-epoch, mean-field variational factor fitting.

    ``JointGlobalLocalGPSettings`` supplied to the fit configures geometry,
    kernels, local KL sampling, device, dtype, and location batch sizes. Its
    feature pretraining stages are not used by this joint factor model.
    """

    n_spatial_factors: int = 8
    n_nonspatial_factors: int = 0
    nonnegative: bool = True
    epochs: int = 100
    learning_rate: float = 0.01
    hyperparameter_learning_rate: float = 0.002
    feature_batch_size: int = 256
    n_samples: int = 4
    loading_l1: float = 0.0
    gradient_clip: float = 10.0
    learn_kernel: bool = True
    initialization: str = "nmf"
    initialization_spots: int = 10000
    initialization_iterations: int = 100
    global_initialization: str = "kernel_smooth"
    variational_scale_parameterization: str = "softplus"
    nonspatial_prior_sd: float = 1.0
    learn_nonspatial_prior: bool = True
    random_state: int = 0

    def __post_init__(self):
        if any(not isinstance(getattr(self, name), bool) for name in
               ("nonnegative", "learn_kernel", "learn_nonspatial_prior")):
            raise ValueError("nonnegative, learn_kernel and learn_nonspatial_prior must be booleans")
        for name in ("n_spatial_factors", "epochs", "feature_batch_size", "n_samples",
                     "initialization_spots", "initialization_iterations"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(self.n_nonspatial_factors, int)
                or isinstance(self.n_nonspatial_factors, bool) or self.n_nonspatial_factors < 0):
            raise ValueError("n_nonspatial_factors must be a nonnegative integer")
        for name in ("learning_rate", "hyperparameter_learning_rate", "gradient_clip",
                     "nonspatial_prior_sd"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.loading_l1) or self.loading_l1 < 0:
            raise ValueError("loading_l1 must be finite and nonnegative")
        if self.initialization not in {"nmf", "random"}:
            raise ValueError("initialization must be nmf or random")
        if self.global_initialization not in {"kernel_smooth", "nearest"}:
            raise ValueError("global_initialization must be kernel_smooth or nearest")
        if self.variational_scale_parameterization not in {"softplus", "raw"}:
            raise ValueError("variational_scale_parameterization must be softplus or raw")
        if not isinstance(self.random_state, int) or self.random_state < 0:
            raise ValueError("random_state must be a nonnegative integer")

    @classmethod
    def from_saved(cls, values):
        """Replay pre-stability-fix artifacts using their original parameterization."""
        values = dict(values)
        values.setdefault("global_initialization", "nearest")
        values.setdefault("variational_scale_parameterization", "raw")
        return cls(**values)
