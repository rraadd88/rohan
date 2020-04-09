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
    
def plot_connections(dplot,label2xy,colval='$r_{s}$',line_scale=40,legend_title='similarity',
                        label2rename=None,
                        element2color=None,
                         xoff=0,yoff=0,
                     rectangle={'width':0.2,'height':0.32},
                     params_text={'ha':'center','va':'center'},
                     params_legend={'bbox_to_anchor':(1.1, 0.5),
                                  'ncol':1,
                                  'frameon':False},
                     params_line={'alpha':1},
                     ax=None,
                    test=False):
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    label2xy={k:[label2xy[k][0]+xoff,label2xy[k][1]+yoff] for k in label2xy}
    dplot['index xy']=dplot['index'].map(label2xy)
    dplot['column xy']=dplot['column'].map(label2xy)
    
    ax=plt.subplot() if ax is None else ax
    from rohan.dandage.plot.ax_ import set_logos
    patches=[]
    label2xys_rectangle_centers={}
    for label in label2xy:
        xy=label2xy[label]
        rect = mpatches.Rectangle(xy, **rectangle, fill=False,fc="none",lw=2,
                                  ec=element2color[label] if not element2color is None else 'k',
                                 zorder=0)

        patches.append(rect)
        line_xys=[np.transpose(np.array(rect.get_bbox()))[0],np.transpose(np.array(rect.get_bbox()))[1][::-1]]
        label2xys_rectangle_centers[label]=[np.mean(line_xys[0]),np.mean(line_xys[1])]
        inset_width=0.3
        axin=ax.inset_axes([*[i-(inset_width*0.5) for i in label2xys_rectangle_centers[label]],inset_width,inset_width])
        axin=set_logos(label=label,element2color=element2color,ax=axin)
        axin.text(np.mean(axin.get_xlim()),np.mean(axin.get_ylim()),
                 label2rename[label] if not label2rename is None else label,
                  **params_text,
#                  ha='center',va='center',zorder=5
                 )
#         ax.text(np.mean(line_xys[0]),np.mean(line_xys[1]),
#             label2rename[label] if not label2rename is None else label,
# #                 label2rename[label] if not label2rename is None else label,
#                 ha='center',va='center',zorder=5)
    dplot.apply(lambda x: ax.plot(*[[label2xys_rectangle_centers[x[k]][0] for k in ['index','column']],
                                  [label2xys_rectangle_centers[x[k]][1] for k in ['index','column']]],
                                lw=(x[colval]-0.49)*line_scale,color='k',zorder=-1,
                                  alpha=params_line['alpha'],
                                ),axis=1)            
    from matplotlib.lines import Line2D
    legend_elements=[Line2D([0], [0], color='k', linestyle='solid', lw=(i-0.49)*line_scale, 
                                alpha=params_line['alpha'],
                                label=f' {colval}={i:1.1f}') for i in [1.0,0.8,0.6]]
    ax.legend(handles=legend_elements,
              title=legend_title,**params_legend)
    ax.set(**{'xlim':[0,1],'ylim':[0,1]})
    if not test:
        ax.set_axis_off()      
    return ax