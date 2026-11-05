"""Experimental all-observation Gaussian V-SKI global-pretraining objective.

Uses the fitting strategy's interpolation, not a screening-grid cache. The
quadratic statistics preserve the full Gaussian likelihood, not the minibatch
optimizer trajectory. No production/default training path imports this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np
from scipy import sparse


def _digest(array):
    a=np.ascontiguousarray(array)
    h=hashlib.sha256(str((a.shape,str(a.dtype))).encode())
    h.update(a.view(np.uint8))
    return h.hexdigest()


def _grid_identity(model):
    return _digest(model.variational_strategy.grid.detach().cpu().numpy())


def _validate_model(model):
    import gpytorch
    s=model.variational_strategy
    if not isinstance(s,gpytorch.variational.GridInterpolationVariationalStrategy):
        raise TypeError('Requires the deterministic grid-interpolation strategy')
    if not isinstance(s._variational_distribution,gpytorch.variational.MeanFieldVariationalDistribution):
        raise TypeError('Requires a mean-field inducing distribution')
    if list(model.named_priors()) or list(model.added_loss_terms()):
        raise ValueError('Additional priors/loss terms are outside this experimental objective')
    return s


def _torch_sparse(matrix,device):
    import torch
    m=matrix.tocsr()
    return torch.sparse_csr_tensor(torch.as_tensor(m.indptr.astype(np.int64),device=device),
        torch.as_tensor(m.indices.astype(np.int64),device=device),
        torch.as_tensor(m.data,device=device,dtype=torch.float64),size=m.shape)


@dataclass
class FittingGridGeometry:
    interpolation: sparse.csr_matrix
    gram: sparse.csr_matrix
    row_squared_norm_min: float
    grid_sha256: str
    coordinate_sha256: str
    coordinate_dtype: str
    build_seconds: float

    @classmethod
    def build(cls,model,coords,*,row_chunk_size=65536):
        import torch
        strategy=_validate_model(model)
        if row_chunk_size<1:raise ValueError('Positive row chunk size required')
        grid=strategy.grid
        x=np.asarray(coords,dtype=grid.detach().cpu().numpy().dtype)
        if x.ndim!=2 or x.shape[1]!=grid.shape[1] or not len(x) or not np.isfinite(x).all():
            raise ValueError('Finite nonempty coordinates of the fitting dimension required')
        started=time.perf_counter();parts=[]
        n_inducing=strategy.inducing_points.shape[-2]
        with torch.no_grad():
            for start in range(0,len(x),row_chunk_size):
                block=torch.as_tensor(x[start:start+row_chunk_size],device=grid.device,dtype=grid.dtype)
                indices,weights=strategy._compute_grid(block)
                # Strategy broadcasts identical geometric weights over features.
                while indices.ndim>2:indices,weights=indices[0],weights[0]
                indices=indices.cpu().numpy();weights=weights.cpu().numpy().astype(np.float64)
                width=indices.shape[1]
                w=sparse.csr_matrix((weights.ravel(),indices.ravel(),
                    np.arange(len(block)+1,dtype=np.int64)*width),shape=(len(block),n_inducing))
                w.sum_duplicates();w.eliminate_zeros();parts.append(w)
        w=sparse.vstack(parts,format='csr');del parts
        h=(w.T@w).tocsr();h.sum_duplicates();h.eliminate_zeros();h.sort_indices()
        minimum=float(np.asarray(w.multiply(w).sum(axis=1)).min())
        return cls(w,h,minimum,_grid_identity(model),_digest(x),str(x.dtype),time.perf_counter()-started)

    def assert_compatible(self,model):
        _validate_model(model)
        if self.grid_sha256!=_grid_identity(model):
            raise ValueError('Fitting grid changed: statistics cannot be reused')

    def summarize(self,targets,*,feature_chunk_size=32):
        """Targets must already use the exact fitted preprocessing/standardization."""
        if feature_chunk_size<1:raise ValueError('Positive feature chunk size required')
        if len(targets.shape)!=2 or targets.shape[0]!=self.interpolation.shape[0] or targets.shape[1]<1:
            raise ValueError('Targets must have every geometry row, in identical order')
        m=self.interpolation.shape[1];p=targets.shape[1]
        b=np.empty((m,p),dtype=np.float64);d=np.empty(p,dtype=np.float64)
        identity=hashlib.sha256();started=time.perf_counter()
        for start in range(0,p,feature_chunk_size):
            y=targets[:,start:start+feature_chunk_size]
            y=y.toarray() if sparse.issparse(y) else np.asarray(y)
            y=np.asarray(y,dtype=np.float64)
            if not np.isfinite(y).all():raise ValueError('Targets must be finite; missing observations unsupported')
            b[:,start:start+len(y.T)]=self.interpolation.T@y
            d[start:start+len(y.T)]=np.einsum('ij,ij->j',y,y)
            for column in y.T:identity.update(np.ascontiguousarray(column).view(np.uint8))
        return GaussianGridStatistics(self,b,d,identity.hexdigest(),time.perf_counter()-started)


@dataclass
class GaussianGridStatistics:
    geometry: FittingGridGeometry
    projection: np.ndarray
    squared_norm: np.ndarray
    target_sha256: str
    build_seconds: float

    def to(self,device):
        return GaussianGridObjective(self,device)


class GaussianGridObjective:
    def __init__(self,statistics,device):
        import torch
        self.statistics=statistics
        self.geometry=statistics.geometry
        self.h=_torch_sparse(self.geometry.gram,device)
        self.b=torch.as_tensor(statistics.projection,dtype=torch.float64,device=device)
        self.d=torch.as_tensor(statistics.squared_norm,dtype=torch.float64,device=device)
        self.diag_h=torch.as_tensor(self.geometry.gram.diagonal(),dtype=torch.float64,device=device)
        self.n=self.geometry.interpolation.shape[0]
        self.last_floor_correction_features=0

    def expected_log_likelihood(self,model,likelihood,*,floor_chunk_size=65536):
        import torch
        import gpytorch
        self.geometry.assert_compatible(model)
        if type(likelihood) is not gpytorch.likelihoods.GaussianLikelihood:
            raise TypeError('Only homoskedastic GaussianLikelihood is supported; no Poisson branch')
        if list(likelihood.named_priors()) or list(likelihood.added_loss_terms()):
            raise ValueError('Additional likelihood priors/loss terms unsupported')
        if not bool(model.variational_strategy.variational_params_initialized.item()):
            raise ValueError('Initialize the actual fitting model before evaluating statistics')
        if floor_chunk_size<1:raise ValueError('Positive floor chunk size required')
        q=model.variational_strategy._variational_distribution
        m=q.variational_mean
        # The installed mean-field forward uses RAW squared stddev, whereas
        # the unchanged structured KL uses its clamped variational_stddev property.
        v=q._variational_stddev.square()
        if m.ndim!=2 or m.T.shape!=self.b.shape or m.device!=self.b.device:
            raise ValueError('Model feature/inducing shape or device mismatch')
        m64=m.double();v64=v.double();noise=likelihood.noise.reshape(-1).double()
        residual=self.d-2*(m64.T*self.b).sum(0)+(m64.T*torch.sparse.mm(self.h,m64.T)).sum(0)
        variance_sum=(v64*self.diag_h).sum(1)
        floor=float(gpytorch.settings.min_variance.value(m.dtype))
        # A safe sufficient bound avoids all N-dependent likelihood work when
        # the variance floor is inactive. Keep a rounding margin near the bound.
        suspect=(v64.detach().amin(1)*self.geometry.row_squared_norm_min<=floor*(1+1e-5))
        chosen=torch.nonzero(suspect,as_tuple=False).ravel()
        self.last_floor_correction_features=int(chosen.numel())
        if chosen.numel():
            correction=torch.zeros(len(chosen),device=m.device,dtype=torch.float64)
            for start in range(0,self.n,floor_chunk_size):
                w=self.geometry.interpolation[start:start+floor_chunk_size]
                w2=_torch_sparse(w.multiply(w),m.device)
                raw=torch.sparse.mm(w2,v64.index_select(0,chosen).T)
                correction=correction+(raw.clamp_min(floor)-raw).sum(0)
            variance_sum=variance_sum.index_add(0,chosen,correction)
        return -.5*(self.n*(math.log(2*math.pi)+noise.log())+(residual+variance_sum)/noise)

    def negative_elbo(self,model,likelihood):
        """Per-feature, per-observation negative ELBO; unchanged model KL."""
        ll=self.expected_log_likelihood(model,likelihood)
        kl=model.variational_strategy.kl_divergence()
        return -ll/self.n+kl.double()/self.n


def fit_global_statistics(model,likelihood,objective,*,steps,learning_rate=.01,
                          hyperparameter_learning_rate=.05,callback=None):
    """Experimental full-data updates, explicitly counted as steps, not epochs.

    Caller initializes the model normally, builds matched statistics, and owns
    validation/artifact promotion. No local/joint updates or output maps occur.
    """
    import torch
    from needlepoint.gpytorch_joint_global_local import _build_optimizer, _model_parameter_partition, _backward_step
    if steps<1:raise ValueError('Positive full-data update count required')
    variational,hyper=_model_parameter_partition(model)
    optimizer,parameters=_build_optimizer(torch,[('variational',variational,learning_rate),
        ('hyperparameters',hyper+list(likelihood.parameters()),hyperparameter_learning_rate)])
    model.train();likelihood.train();history=[]
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss=objective.negative_elbo(model,likelihood).sum()
        norm=_backward_step(torch,loss,optimizer,parameters,gradient_clip=None,stage='experimental full-data Gaussian grid')
        record=dict(step=step+1,negative_elbo_sum=float(loss.detach()),gradient_norm=norm,
            floor_correction_features=objective.last_floor_correction_features)
        history.append(record)
        if callback is not None:callback(record)
    return history
