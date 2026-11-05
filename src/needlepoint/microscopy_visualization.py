"""Map-focused displays of a full-data microscopy factorization."""
from __future__ import annotations

import numpy as np


def plot_component_maps(root,support,pred,loadings,intensity_scale,roles,*,signed_indices=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k=loadings.shape[1]
    amplitudes=pred["factor_mean"]*intensity_scale
    def label(j):
        return f"{j+1}. "+roles[j].replace("_anchor","").replace("_"," ")
    def draw(ax,j):
        field=np.full(support.shape,np.nan);field[support]=amplitudes[:,j]
        lim=max(float(np.percentile(abs(amplitudes[:,j]),99)),1e-8)
        signed=j in signed_indices
        im=ax.imshow(field,cmap="RdBu_r" if signed else "magma",vmin=-lim if signed else 0,vmax=lim)
        ax.set_title(label(j));ax.set_axis_off()
        return im
    def overview(indices,filename,title,columns):
        rows=(len(indices)+columns-1)//columns
        fig,axes=plt.subplots(rows,columns,figsize=(4*columns,4.4*rows),squeeze=False,layout="constrained")
        for ax in axes.flat: ax.set_axis_off()
        for ax,j in zip(axes.flat,indices):
            im=draw(ax,j);fig.colorbar(im,ax=ax,shrink=.7)
        fig.suptitle(title+"\nMap amplitude in stored intensity units; individual 99th-percentile color limits",fontsize=13)
        fig.savefig(root/filename,dpi=180);plt.close(fig)
    overview(list(range(k)),"components.png",f"Full-data spatial unmixing · {k} components",min(k,4))
    tissue=[j for j,role in enumerate(roles) if role.startswith("tissue_")]
    if not tissue: tissue=list(range(k))
    overview(tissue,"tissue_maps.png",f"Full-data fit · {len(tissue)} tissue components",min(len(tissue),3))
    cols=min(k,4);rows=(k+cols-1)//cols
    fig,axes=plt.subplots(rows,cols,figsize=(4*cols,3*rows),squeeze=False,layout="constrained")
    frames=np.arange(1,len(loadings)+1)
    for ax in axes.flat: ax.set_axis_off()
    for j,ax in enumerate(axes.flat):
        if j>=k: break
        ax.set_axis_on();ax.plot(frames,loadings[:,j],"o-",ms=3)
        ax.axhline(0,color="gray",lw=.6)
        ax.set(title=label(j),xlabel="Acquisition frame",ylabel="Unit-norm mixing weight")
    fig.savefig(root/"frame_profiles.png",dpi=180);plt.close(fig)
    for j in range(k):
        fig,axes=plt.subplots(1,2,figsize=(9,4.8),layout="constrained")
        im=draw(axes[0],j);fig.colorbar(im,ax=axes[0],shrink=.75,label="Map amplitude (counts)")
        axes[1].plot(frames,loadings[:,j],"o-",ms=4)
        axes[1].axhline(0,color="gray",lw=.6)
        axes[1].set(title="Frame profile",xlabel="Acquisition frame",ylabel="Unit-norm mixing weight")
        fig.savefig(root/f"component_{j+1:02d}.png",dpi=180);plt.close(fig)
