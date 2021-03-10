import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os.path import exists, basename,dirname
# plt.style.use('ggplot')
from scipy import stats

from rohan.dandage.io_strs import make_pathable_string

def get_linspace(x,n=100):
    return np.linspace(np.min(x),np.max(x),n)
def get_grid(x,y,n=100):
#     return np.linspace(np.min(x),np.max(x),n)
    return np.mgrid[np.min(x):np.max(x), np.min(y):n:np.max(y)]
def annot_corners(labels,X,Y,ax,space=-0.2,fontsize=18):
    xlims,ylims=get_axlims(X,Y,space=space)
    print('corners xlims,ylims=',xlims,ylims)
    
    labeli=0
    for x in xlims:
        for y in ylims:
            ax.text(x,y,labels[labeli],
                color='k',
                fontsize=fontsize,
                ha='center',
                va='center',
                bbox=dict(facecolor='w',edgecolor='none',alpha=0.4),
                )
            labeli+=1
    return ax
def get_axlims(X,Y,space=0.2,equal=False):
    try:
        xmin=np.min(X)
        xmax=np.max(X)
    except:
        print(X)
    xlen=xmax-xmin
    ymin=np.min(Y)
    ymax=np.max(Y)
    ylen=ymax-ymin
    xlim=(xmin-space*xlen,xmax+space*xlen)
    ylim=(ymin-space*ylen,ymax+space*ylen)
    if not equal:
        return xlim,ylim
    else:
        lim=[np.min([xlim[0],ylim[0]]),np.max([xlim[1],ylim[1]])]
        return lim,lim

def get_grid3d(x,y,z,interp):
#     from matplotlib.mlab import griddata
    from scipy.interpolate import griddata
    xi=get_linspace(x)
    yi=get_linspace(y)
    zi = griddata(x, y, z, xi, yi, interp=interp)
    return xi,yi,zi
    
from rohan.dandage.plot.colors import get_cmap_subset,saturate_color
from rohan.dandage.plot.ax_ import set_colorbar,grid
def plot_contourf(x,y,z,test=False,ax=None,fig=None,
                 grid_n=50,
                  labelx='x',labely='y',labelz='z',
                 params_contourf={},
                  cbar=True,
                  params_cbar={'bbox_to_anchor':(1.01, 0.2, 0.5, 0.8)},
                  figsize=[4,4],
                 ):
    from scipy.interpolate import griddata
    xi=np.linspace(np.min(x),np.max(x),grid_n)
    yi=np.linspace(np.min(y),np.max(y),grid_n)

    X,Y= np.meshgrid(xi,yi)
    Z = griddata((x, y), z, (X, Y),method='linear')
    fig=plt.figure(figsize=figsize) if fig is None else fig
    ax=plt.subplot() if ax is None else ax
    ax_pc=ax.contourf(X,Y,Z,int(grid_n/5),**params_contourf)
    if test:
        ax.scatter(X,Y)        
    ax.set_xlabel(labelx),ax.set_ylabel(labely)    
    if cbar:
        fig=set_colorbar(fig,ax,ax_pc,label=labelz,**params_cbar)
    ax=grid(ax,axis='both')
    return fig,ax
def annot_contourf(colx,coly,colz,dplot,annot,ax=None,fig=None,vmin=0.2,vmax=1):
    """
    annot can be none, dict,list like anything..
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes    
    ax=plt.subplot() if ax is None else ax
    fig=plt.figure() if fig is None else fig
    if isinstance(annot,dict):
        #kdeplot
        # annot is a colannot
        for ann in annot:
            if not ann in ['line']:
                for subseti,subset in enumerate(list(annot[ann])[::-1]):
                    df_=dplot.loc[(dplot[ann]==subset),:]
                    sns.kdeplot(df_[colx],df_[coly],ax=ax,
                              shade_lowest=False,
                                n_levels=5,
                              cmap=get_cmap_subset(annot[ann][subset], vmin, vmax),
#                               cbar_ax=fig.add_axes([0.94+subseti*0.25, 0.15, 0.02, 0.32]), #[left, bottom, width, height]
                              cbar_ax=inset_axes(ax,
                                               width="5%",  # width = 5% of parent_bbox width
                                               height="50%",  # height : 50%
                                               loc=2,
                                               bbox_to_anchor=(1.36-subseti*0.35, -0.4, 0.5, 0.8),
                                               bbox_transform=ax.transAxes,
                                               borderpad=0,
                                               ),                                
                              cbar_kws={'label':subset},
                              linewidths=3,
                             cbar=True)
            if ann=='line':
                dannot=annot[ann]
                for subset in dannot:
                    ax.plot(dannot[subset]['x'],dannot[subset]['y'],marker='o', linestyle='-',
                            color=saturate_color(dannot[subset]['color'][0],1),
                           )
                    va_top=True
                    for x,y,s,c in zip(dannot[subset]['x'],dannot[subset]['y'],dannot[subset]['text'],dannot[subset]['color']):
                        ax.text(x,y,f" {s}",color=saturate_color(c,1.1),weight = 'bold',va='top' if va_top else 'bottom')
                        va_top=False if va_top else True #flip
    return fig,ax

def plot_contourf_fitted(fun,
                         ax=None,
                         test=False,
                         levels=5,zorder=1,
                         cmap='Reds',
                 **kws_fitting,):
    xg,yg,zg,zg_hat=fun(**kws_fitting)
    if test:
        plt.figure()
        plt.contourf(xg,yg,zg,levels=levels)
        plt.colorbar()
        
    if ax is None:
        _,ax=plt.subplots(figsize=[3,3])        
    ax=ax.contourf(xg,yg,zg_hat,
                levels=levels,cmap=cmap,
                  zorder=zorder)
    return ax