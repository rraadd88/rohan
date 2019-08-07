import numpy as np
import matplotlib.pyplot as plt 
import logging

def set_equallim(ax,diagonal=False):
    min_,max_=np.min([ax.get_xlim()[0],ax.get_ylim()[0]]),np.max([ax.get_xlim()[1],ax.get_ylim()[1]])
    ax.plot([min_,max_],[min_,max_],'--',color='gray')
    ax.set_xticks(ax.get_yticks())
    ax.set_xlim(min_,max_)
    ax.set_ylim(min_,max_)
#     import matplotlib.ticker as plticker
#     loc = plticker.MultipleLocator(base=(max_-min_)/5) # this locator puts ticks at regular intervals
#     ax.xaxis.set_major_locator(loc)
#     ax.yaxis.set_major_locator(loc)    
    return ax

def grid(ax,axis=None):
    w,h=ax.figure.get_size_inches()
    if w/h>=1.1 or axis=='y' or axis=='both':
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
    if w/h<=0.9 or axis=='x' or axis=='both':
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
    return ax

## egend related stuff: also includes colormaps
def sort_legends(ax,params={}):
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels,**params)
    return ax

def set_colorbar(fig,ax,ax_pc,label,bbox_to_anchor=(0.05, 0.5, 1, 0.45),
                orientation="vertical",):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if orientation=="vertical":
        width,height="5%","50%"
    else:
        width,height="50%","5%"        
    axins = inset_axes(ax,
                       width=width,  # width = 5% of parent_bbox width
                       height=height,  # height : 50%
                       loc=2,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )
    fig.colorbar(ax_pc, cax=axins,
                 label=label,orientation=orientation,)
    return fig

def set_sizelegend(fig,ax,ax_pc,sizes,label,scatter_size_scale,xoff=0.05,bbox_to_anchor=(0, 0.2, 1, 0.4)):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(ax,
                       width="20%",  # width = 5% of parent_bbox width
                       height="60%",  # height : 50%
                       loc=2,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )
    axins.scatter(np.repeat(1,3),np.arange(0,3,1),s=sizes*scatter_size_scale,c='k')
    axins.set_ylim(axins.get_ylim()[0]-xoff*2,axins.get_ylim()[1]+0.5)
    for x,y,s in zip(np.repeat(1,3)+xoff*0.5,np.arange(0,3,1),sizes):
        axins.text(x,y,s,va='center')
    axins.text(axins.get_xlim()[1]+xoff,np.mean(np.arange(0,3,1)),label,rotation=90,ha='left',va='center')
    axins.set_axis_off() 
    return fig
