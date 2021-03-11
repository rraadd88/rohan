from rohan.global_imports import *
from rohan.dandage.plot.annot import annot_contourf,annot_corners

from rohan.dandage.plot.colors import get_cmap_subset,saturate_color
from rohan.dandage.plot.ax_ import set_colorbar,grid

def plot_contourf(x,y,z,
                  method='linear',
                  params_grid={},
                  fun=None,
                  params_fun={},
                grids=50,
                cbar=True,
                params_cbar={'bbox_to_anchor':(1.01, 0.2, 0.5, 0.8),},
                zlabel='z',
                params_ax={'xlabel':'x','ylabel':'y'},
                ax=None,fig=None,
                figsize=[3,3],
                test=False,
                **kws_contourf,
                ):
    from rohan.dandage.stat.fit import get_grid
    xg,yg,zg=get_grid(x,y,z,
                grids=grids,
                method=method,**params_grid)
    if not fun is None:
        ## zg=zg_hat
        xg,yg,_,zg=fun(xg,yg,zg,**params_fun)
    
    fig=plt.figure(figsize=figsize) if fig is None else fig
    ax=plt.subplot() if ax is None else ax
    ax_pc=ax.contourf(xg,yg,zg,int(grids/5),**kws_contourf)
    if test:
        ax.scatter(xg,yg)        
    ax.set(**params_ax)
    if cbar:
        fig=set_colorbar(fig,ax,ax_pc,label=zlabel,**params_cbar)
    ax=grid(ax,axis='both')
    return fig,ax