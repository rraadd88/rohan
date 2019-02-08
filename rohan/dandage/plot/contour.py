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
    
def plot_contourf(x,y,z,contourlevels=15,xlabel=None,ylabel=None,
                scatter=False,contour=False,
                annot_fit_land=True,
                streamlines=False,
                cmap="coolwarm",cbar=True,cbar_label="",
                cbar_ax_pos=[0.55, 0.3, 0.035, 0.5],
                a=0.5,vmin=None,vmax=None,interp='linear',#'nn',
                xlog=False,test=False,
                fig=None,ax=None,plot_fh=None):
    from matplotlib.mlab import griddata
#     from scipy.interpolate import griddata
    xi=get_linspace(x)
    yi=get_linspace(y)
    zi = griddata(x, y, z, xi, yi, interp=interp)
    if ax==None: 
        ax=plt.subplot(121)
    if vmax==None: 
        vmax=abs(zi).max()
    if vmin==None: 
        vmin=abs(zi).min()
    if contour:
        CS = ax.contour(xi, yi, zi, contourlevels, linewidths=0.5, colors='k',alpha=a)
    CS = ax.contourf(xi, yi, zi, contourlevels, 
                      cmap=cmap,
                      vmax=vmax, vmin=vmin)

    if streamlines:
        dy, dx = np.gradient(zi) # Flow goes down gradient (thus -zi)
        if test:
            print([np.shape(xi),np.shape(yi)])
            print([np.shape(dx),np.shape(dy)])
        ax.streamplot(xi, yi, dx, dy, color='k', density=0.5,
                         linewidth=1,minlength=0.05,arrowsize=1.5)
    if cbar:
        colorbar_ax = fig.add_axes(cbar_ax_pos) #[left, bottom, width, height]
        colorbar_ax2=fig.colorbar(CS, cax=colorbar_ax,extend='both',
#                                  boundaries=[-1,0,1]
                                 )
        colorbar_ax2.set_label(cbar_label)
        clim=[-1,1]
        colorbar_ax2.set_clim(clim[0],clim[1])
    # plot data points.
    if scatter:
        ax.scatter(x, y, marker='o', c='b', s=5, zorder=10)
    if annot_fit_land:
        labels=["$F:cB$","$F:B$","$cF:cB$","$cF:B$"]
        if xlog:
            x=x/1.61
        ax=annot_corners(labels,x,y,ax,fontsize=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot_fh!=None:
        fig.savefig(plot_fh,format="pdf")
    return ax  