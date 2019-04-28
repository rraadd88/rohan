import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_rect(x,
            title,
            xlabel,
            ylabel,
            xlen=400,
            ylen=20000,
              color='blue',
              approx='y',
              fig=None,
            ):
    import matplotlib.lines as mlines
    import matplotlib.patches as patches
    if fig is None:
        fig=plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    rect=patches.Rectangle(
            (x, 0),
            xlen,ylen,
        color=color,
    #         fill=False      # remove background
         ) 
    x=np.arange(0,1,0.025)
    y=np.ravel([[0.75,0.7] for i in np.arange(len(x)/2)])
    line = mlines.Line2D(x,y, lw=2.,
                        color='w',
                         zorder=3,
                         dash_joinstyle='miter',
                        )

    # title
    ax.text(np.mean([rect.get_x(),rect.get_x()+rect.get_width()]),rect.get_height(),title,va='bottom',ha='center')
    # xlabel
    ax.text(np.mean([rect.get_x(),rect.get_x()+rect.get_width()]),rect.get_y()-0.015,xlabel,va='top',ha='center')
    # ylabel
    ax.text(-0.05,0,ylabel,va='bottom',ha='left',color=color,rotation=90)
    
    ax.add_patch(rect) 
    ax.add_line(line)
    plt.axis('off')
    return ax

def makekws_plot_rect(df,fig,idx,x_):
    kws_plot_rect=df.loc[idx,:].to_dict()
    kws_plot_rect['x']=x_
    kws_plot_rect['fig']=fig
    if idx!=0:kws_plot_rect['ylabel']=''
    return kws_plot_rect

def df2plotshape(dlen,xlabel_unit,ylabel_unit,
                 suptitle='',fix='h',xlabel_skip=[],
                 test=False):
    """
    _xlen: 
    _ylen:
    title:
    """
    dlen['xlabel']=dlen.apply(lambda x : f"{x['_xlen']}" if not x['title'] in xlabel_skip else '',axis=1)
    dlen['ylabel']=dlen.apply(lambda x : "",axis=1)
    ylen=dlen['_ylen'].unique()[0]
    
    if test: print(dlen.columns)
    if fix=='h':    
        dlen['xlen']=dlen['_xlen']/dlen['_xlen'].max()/len(dlen)*0.8
        dlen['ylen']=0.8
        subsets=[]
        for subset in [c for c in dlen if c.startswith('_ylen ')]:
            subsetcol=subset.replace('_ylen','ylen')
            dlen[subsetcol]=0.25
            subsets.append(subsetcol)
        subsets2cols=dict(zip([subsetcol.replace('ylen ','') for s in subsets],subsets))
        if test: print(dlen.columns)
        if test: print(subsets2cols)
    elif fix=='w':    
        dlen['xlen']=0.8
        dlen['ylen']=dlen['_ylen']/dlen['_ylen'].max()/len(dlen)*0.85
   
    dlen=dlen.drop([c for c in dlen if c.startswith('_')],axis=1)    
    if test: print(dlen.columns)
    if fig is None:fig = plt.figure(figsize=[4,4])
    for idx in dlen.index:
        if idx==0:x_=0
        kws_plot_rect=makekws_plot_rect(dlen,fig,idx,x_)
        if test: print(kws_plot_rect)
        kws_plot_rect_big={k:kws_plot_rect[k] for k in kws_plot_rect if not 'ylen ' in k}
        kws_plot_rect_big['color']='gray'
        ax=plot_rect(**kws_plot_rect_big)
        for subset in subsets2cols:
            kws_plot_rect=makekws_plot_rect(dlen.drop('ylen',axis=1).rename(columns={subsets2cols[subset]:'ylen'}),fig,idx,x_)
            kws_plot_rect['title']=''
            kws_plot_rect['xlabel']=''
            kws_plot_rect['ylabel']=subset
            if idx!=0:kws_plot_rect['ylabel']=''
            if test: print(kws_plot_rect)
            ax=plot_rect(**kws_plot_rect)            
        x_=kws_plot_rect['x']+dlen.loc[idx,'xlen']+0.1  
    
    ax.text(x_/2.3,-0.1,xlabel_unit,ha='center')
    ax.text(x_/2.3,0.9,suptitle,ha='center')
    ax.text(-0.1,0.4,f"total ~{ylen}{ylabel_unit}",va='center',rotation=90)
    if fig is not None:
        return fig,ax
    else:
        return ax
# slankey
from collections import OrderedDict
ordereddict=OrderedDict

def get_label2fractions(label2count,test=False):
    # list(label2count.values())[0]
    label2fraction_input=ordereddict({})
    for k in label2count:
        label2fraction_input[k]=round(label2count[k]/list(label2count.values())[0],2)
    if test:
        print(label2fraction_input)
    label2fraction_in=ordereddict({})   
    label2fraction_out=ordereddict({})
    label2fraction_drop=ordereddict({})
    for ki,k in enumerate(label2fraction_input):
        if ki < len(label2fraction_input.values())-1:
            label2fraction_in[k]=label2fraction_input[k]
            label2fraction_out[f"out {k}"]=-1*(list(label2fraction_input.values())[ki+1])
            label2fraction_drop[f"drop {k}"]=-1*(label2fraction_out[f"out {k}"]+label2fraction_in[k])
    if test:
        print(label2fraction_in)
        print(label2fraction_out)
        print(label2fraction_drop)
    label2fractions=[]
    for k1,k2,k3 in zip(label2fraction_in.keys(),label2fraction_drop.keys(),label2fraction_out.keys()):
        label2fraction=ordereddict({})
        label2fraction[k1]=label2fraction_in[k1]
        label2fraction[k3 if test else '']=label2fraction_out[k3]
        label2fraction[k2 if test else ' ']=label2fraction_drop[k2]
        label2fractions.append(label2fraction)
    return label2fractions
def plot_slankey(dslankeys):
    import matplotlib.pyplot as plt
    from matplotlib.sankey import Sankey
    bg_color='w'
#     fig=plt.figure()
    fig = plt.figure(facecolor=bg_color, edgecolor=bg_color)
    ax = fig.add_subplot(111)
    ax.patch.set_facecolor(bg_color)
    sankey = Sankey(ax=ax,scale=1,tolerance=0.000000000000001,offset=0.5)
    for prior,dslankey in enumerate(dslankeys):
        orientations=np.ravel([(0,0,-1) for i in range(int(len(dslankey.keys())/2))])
        pathlengths=np.ravel([(1,1,1) for i in range(int(len(dslankey.keys())/2))])
        sankey.add(flows=list(dslankey.values()),
               labels=list(dslankey.keys()),
               orientations=orientations,
                   pathlengths=pathlengths,
                   prior=prior-1 if prior!=0 else None, 
                   connect=(1, 0),
                  trunklength=1)
    sankey.finish()
    # Hide grid lines
    ax.grid(False)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
#     ax.set_zticks([])
    return ax