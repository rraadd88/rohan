from rohan.global_imports import *
from rohan.lib.plot.annot import annot_contourf,annot_corners

from rohan.lib.plot.colors import get_cmap_subset,saturate_color
from rohan.lib.plot.ax_ import set_colorbar,grid

def plot_contourf(x,y,z,
                  # coutourf
                  levels=None,
                  contourf_remove_lowest=False,
                  # grid
                  method='linear',
                  grids=50,
                  params_grid={},
                  # fit
                  fun=None,
                  params_fun={},
                  # scatter
                  scatter=True,
                  params_scatter=dict(
                                    s=1,
                                    ),
                  # cbar
                cbar=True,
                params_cbar={'bbox_to_anchor':(1.01, 0.2, 0.5, 0.8),},
                zlabel='z',
                  # ax
                params_ax={'xlabel':'x','ylabel':'y'},
                ax=None,fig=None,
                figsize=[3,3],
                test=False,
                **kws_contourf,
                ):
    from rohan.lib.stat.fit import get_grid
    xg,yg,zg=get_grid(x,y,z,
                grids=grids,
                method=method,**params_grid)
    if not fun is None:
        ## zg=zg_hat
        xg,yg,_,zg=fun(xg,yg,zg,**params_fun)
    
    fig=plt.figure(figsize=figsize) if fig is None else fig
    ax=plt.subplot() if ax is None else ax
#     ax1=ax.contourf(xg,yg,zg,
#                       levels=int(grids/5) if levels is None else levels,
#                       **kws_contourf)
    ax1=ax.contour(xg,yg,zg,
                      levels=int(grids/5) if levels is None else levels,
                      **kws_contourf)
    if contourf_remove_lowest:
        for coll in ax1.collections:
            coll.remove()
            break    
    if scatter:
        ax.scatter(x=x,y=y,
                   color='k',
                   **params_scatter,
                   zorder=2)
    ax.set(**params_ax)
    if cbar:
        fig=set_colorbar(fig,ax,ax1,label=zlabel,**params_cbar)
    return fig,ax,ax1