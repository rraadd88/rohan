import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os.path import exists, basename,dirname

def plot_summarystats(df,cols=['mean','min','max','50%'],plotp=None,ax=None,value_name=None):
    if ax is None:ax=plt.subplot(111)
    if not any([True if c in df else False for c in cols]):
        df=df.describe().T
    ax=df.loc[:,cols].plot(ax=ax)
    ax.fill_between(df.index, df['mean']-df['std'], df['mean']+df['std'], color='b', alpha=0.2,label='std')
    ax.legend(bbox_to_anchor=[1,1])
    if value_name is None:
        ax.set_ylabel('value')
    else:
        ax.set_ylabel(value_name)
    ax.set_xticklabels(df.index)    
    return ax
    
def plot_mean_std(df,cols=['mean','min','max','50%'],plotp=None):
    return plot_summarystats(df,cols=cols,plotp=plotp)
    
def plot_connections(dplot,params,colval='$r_{s}$',line_scale=40,legend_title='similarity',ax=None):
    ax=plt.subplot() is ax is None else ax
    from rohan.dandage.plot.ax_ import set_logo
    patches=[]
    params['rectangle']={'width':0.2,'height':0.32}
    label2xys_rectangle_centers={}
    for label in params['label2xy']:
        xy=params['label2xy'][label]
        rect = mpatches.Rectangle(xy, **params['rectangle'], fill=True,fc='w',lw=2,ec=params['element2color'][label])
        line_xys=[np.transpose(np.array(rect.get_bbox()))[0],np.transpose(np.array(rect.get_bbox()))[1][::-1]]
        ax.text(np.mean(line_xys[0]),np.mean(line_xys[1]),
                params['label2rename'][label],
                ha='center',va='center')
        label2xys_rectangle_centers[label]=[np.mean(line_xys[0]),np.mean(line_xys[1])]        
        patches.append(rect)
        img_logop=f"logos/{label.replace(' ','_')}.svg.png"
        if exists(img_logop):
            set_logo(imp=img_logop,size=0.4,
                     bbox_to_anchor=rect.get_bbox(),ax=ax)
    dplot.apply(lambda x: ax.plot(*[[label2xys_rectangle_centers[x[k]][0] for k in ['index','column']],
                                  [label2xys_rectangle_centers[x[k]][1] for k in ['index','column']]],
                                lw=(x[colval]-0.49)*line_scale,color='k',zorder=-1,alpha=0.65,
                                ),axis=1)            
    from matplotlib.lines import Line2D
    legend_elements=[Line2D([0], [0], color='k', linestyle='solid', lw=(i-0.49)*line_scale, alpha=0.65,
                            label=f' {colval}={i:1.1f}') for i in [1.0,0.8,0.6]]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.95, 1.1),ncol=3,frameon=False,title=legend_title)
    # label(grid[1], "Rectangle")
    collection = PatchCollection(patches,match_original=True)
    # collection.set_array(np.array(colors))
    ax.add_collection(collection)
    ax.set(**{'xlim':[0,1],'ylim':[0,1]})
    ax.set_axis_off()      