"""Full-support diagnostics for conditional shared/private LBM experiments."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score
import torch

from .shared_private_lbm import gaussian_evidence


def section_diagnostics(section, probabilities, parameters, section_index, adjacency, reference):
    b=probabilities.detach().cpu().numpy()
    labels=b.argmax(1)
    entropy=-(b*np.log(np.clip(b,1e-30,None))).sum(1)
    edges=sparse.triu(adjacency,k=1).tocoo()
    equal=labels[edges.row]==labels[edges.col]
    same=sparse.coo_matrix((np.ones(2*equal.sum()),
        (np.r_[edges.row[equal],edges.col[equal]],np.r_[edges.col[equal],edges.row[equal]])),shape=adjacency.shape).tocsr()
    n_components, components=connected_components(same,directed=False)
    sizes=np.bincount(components)
    common=gaussian_evidence(section.shared,parameters['shared_mean'][section_index],parameters['shared_variance'])
    p=parameters['private'][section_index]
    private=gaussian_evidence(section.private,p['mean'],p['variance']) if p is not None else None
    report={'section':section.name,'n_spots':len(b),'hard_domain_count':int(np.unique(labels).size),
        'mean_max_probability':float(b.max(1).mean()),'mean_normalized_entropy':float(entropy.mean()/np.log(b.shape[1])),
        'fraction_probability_ge_0_8':float((b.max(1)>=.8).mean()),
        'native_boundary_fraction':float(1-equal.mean()) if len(equal) else 0.,
        'same_domain_components':int(n_components),
        'fraction_spots_in_components_lt_10':float(sizes[sizes<10].sum()/len(b)),
        'ari_vs_preferred_combined_R12':float(adjusted_rand_score(reference,labels)),
        'shared_expected_log_likelihood':float((probabilities*common).sum(1).mean()),
        'private_expected_log_likelihood_unscaled':None if private is None else float((probabilities*private).sum(1).mean()),
        'minimum_soft_domain_fraction':float(b.mean(0).min()),
        'domain_soft_fractions':b.mean(0).tolist(),'domain_hard_counts':np.bincount(labels,minlength=b.shape[1]).tolist()}
    return report


def shared_domain_separation(sections, parameters):
    """Label-identifiability diagnostics, not a biological accuracy score."""
    means=parameters['shared_mean']
    variance=parameters['shared_variance']
    mass=sections[0].shared.mass.to(means.dtype)
    mass=mass/mass.sum()
    pair_rows=[]
    for r in range(means.shape[2]):
        for s in range(r+1,means.shape[2]):
            scale=(variance[:,r]+variance[:,s])/2
            dist=((means[:,:,r]-means[:,:,s]).square()/scale*mass).sum(1).sqrt()
            pair_rows.append({'domain_a':r+1,'domain_b':s+1,
                **{sections[d].name:float(dist[d]) for d in range(len(sections))}})
    agreement=[]
    for r in range(means.shape[2]):
        distance=((means[0,:,r]-means[1,:,r]).square()/variance[:,r]*mass).sum().sqrt()
        agreement.append({'domain':r+1,'between_section_shared_signature_distance':float(distance)})
    return pair_rows,agreement
