"""Anchors and low-rank distinctiveness for explicit microscopy components."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF
from threadpoolctl import threadpool_limits


def initialize_explicit_components(values, n_components=4, seed=384):
    """Training-only PC1/PC2 working anchors; residual NMF tissue factors.

    PC1 is oriented to have a positive dominant frame loading and shifted into
    a positive map. PC2 remains signed. These are provisional associations,
    not external anatomical or acquisition ground truth.
    """
    values = np.asarray(values, dtype=np.float64)
    center = values.mean(0)
    _, _, vt = np.linalg.svd(values-center, full_matrices=False)
    profiles = vt[:2].T.copy()
    signs = np.sign(profiles[np.argmax(abs(profiles), axis=0), np.arange(2)])
    profiles *= signs
    profiles[:, 0] = np.maximum(profiles[:, 0], 1e-6)
    profiles[:, 0] /= np.linalg.norm(profiles[:, 0])
    scores = (values-center) @ profiles
    white = np.maximum(scores[:, 0]-np.quantile(scores[:, 0], .01), .02)
    boundary = scores[:, 1]
    anchors = np.column_stack([white, boundary])
    residual = values-anchors@profiles.T
    intercept = np.quantile(residual, .01, axis=0)
    positive = np.maximum(residual-intercept, 0)
    nmf = NMF(n_components=n_components-2, init="nndsvdar", random_state=seed,
              max_iter=300, tol=1e-4)
    with threadpool_limits(limits=1):
        tissue = nmf.fit_transform(positive)
    tissue_profiles = nmf.components_.T
    norm = np.linalg.norm(tissue_profiles, axis=0).clip(1e-8)
    tissue_profiles /= norm
    tissue = np.maximum(tissue*norm, .02)
    factors = np.column_stack([anchors, tissue])
    loadings = np.column_stack([profiles, tissue_profiles])
    latent = factors.copy()
    indices = [0, *range(2, n_components)]
    f = np.maximum(factors[:, indices], .02)
    latent[:, indices] = f+np.log(-np.expm1(-f))
    return factors.astype(np.float32), loadings.astype(np.float32), latent.astype(np.float32), intercept.astype(np.float32)


def outside_nuisance_fraction(maps, profiles, *, center_maps=False):
    """Energy outside span(vec(W), vec(B)), without building pixel x frame maps.

    Gram(vec(a_j s_j^T), vec(a_k s_k^T)) is the product of the spatial
    and profile Gram entries. First two columns are W and B. Centering removes
    the intercept ambiguity, so a constant tissue map cannot count as structure.
    """
    import torch
    if maps.ndim != 2 or profiles.ndim != 2 or maps.shape[1] != profiles.shape[1] or maps.shape[1] < 3:
        raise ValueError("Need matching map/profile matrices with W, B and at least one tissue factor")
    if center_maps:
        maps = maps-maps.mean(0, keepdim=True)
    mn = maps.norm(dim=0)
    sn = profiles.norm(dim=0)
    a = maps/mn.clamp_min(1e-12)
    s = profiles/sn.clamp_min(1e-12)
    gram = (a.T@a)*(s.T@s)
    nuisance = gram[:2, :2]
    cross = gram[:2, 2:]
    projection = (cross*(torch.linalg.pinv(nuisance, hermitian=True, rtol=1e-6)@cross)).sum(0)
    fraction = (1-projection).clamp(0, 1)
    # Zero maps have no supported novel structure, irrespective of normalization.
    return torch.where((mn[2:]*sn[2:]) > 1e-8, fraction, torch.zeros_like(fraction))


def distinctiveness_penalty(maps, profiles, minimum_fraction):
    raw = outside_nuisance_fraction(maps, profiles)
    structural = outside_nuisance_fraction(maps, profiles, center_maps=True)
    penalty = (minimum_fraction-raw).clamp_min(0).square().sum()
    penalty = penalty+(minimum_fraction-structural).clamp_min(0).square().sum()
    return penalty, raw, structural
