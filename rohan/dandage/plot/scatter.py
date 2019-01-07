import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os.path import exists, basename,dirname
from scipy import stats

from rohan.dandage.io_strs import make_pathable_string

def plot_reg(d,xcol,ycol,textxy=[0.65,1],
             scafmt='hexbin',
            rp=True,rs=True,vmax=10,cbar_label=None,
             axscale_log=False,
            ax=None,fig=None, 
            plotp=None,plotsave=False):
    d=d.dropna(subset=[xcol,ycol],how='any')
    d=d.dropna(subset=[xcol,ycol],how='all')
    if fig is None:
        fig=plt.figure(figsize=[3.5,3])
    if ax is None:
        ax=plt.subplot(111)
    if scafmt=='hexbin':
        ax=d.plot.hexbin(x=xcol,y=ycol,ax=ax,vmax=vmax,gridsize=25,cmap='Blues')
    elif scafmt=='sca':
        ax=d.plot.scatter(x=xcol,y=ycol,ax=ax,color='b',alpha=0.1)
    rpear=stats.pearsonr(d[xcol], d[ycol])[0]
    rspea=stats.spearmanr(d[xcol], d[ycol])[0]
    if rp and rs:
        textstr=f'$r$={rpear:.2f}\n$\\rho$={rspea:.2f}'
    elif rp and not rs:
        textstr=f'$r$={rpear:.2f}'
    elif not rp and rs:
        textstr=f'$\\rho$={rspea:.2f}'
    props = dict(facecolor='w', alpha=0.3)
    fig.text(textxy[0],textxy[1],textstr,
            ha='left',va='top',bbox=props)
    if scafmt=='hexbin':
        if cbar_label is None and d.index.name is None:
            cbar_label='# of points'
        else:
            cbar_label=f"# of {d.index.name}s"
        fig.text(1,0.5,cbar_label,rotation=90,
                ha='center',va='center')
    if axscale_log:
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlim(0,1)
#     ax.set_ylim(0,1)
    if not d.columns.name is None:
        ax.set_title(d.columns.name)
    plt.tight_layout()
    if plotsave:
        if plotp is None:
            plotp=f"plot/{scafmt}_{make_pathable_string(xcol)}_vs_{make_pathable_string(ycol)}.svg"
        print(plotp)
        plt.savefig(plotp)
    else:
        return fig, ax