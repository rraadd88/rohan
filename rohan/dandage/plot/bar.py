import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def plot_value_counts(df,col,logx=False,
                      kws_hist={'bins':10},
                      kws_bar={},
                     grid=False,
                     axes=None,fig=None,
                     hist=True):
    dplot=pd.DataFrame(df[col].value_counts()).sort_values(by=col,ascending=True)
    figsize=(4, len(dplot)*0.4)
    if axes is None:
        if hist:
            fig, axes = plt.subplots(2,1, sharex=True, 
                                       gridspec_kw={'height_ratios': [1, 6]},
                                       figsize=figsize,
                                    )
        else:
            fig, axes = plt.subplots(1,1, 
                                    figsize=figsize,
                                    )
    if hist:
        _=dplot.plot.hist(ax=axes[0],legend=False,**kws_hist
        #                   orientation="horizontal"
                         )
        axbar=axes[1]
    else:
        axbar=axes
    _=dplot.plot.barh(ax=axbar,legend=False,**kws_bar)
    axbar.set_xlabel('count')
    from rohan.dandage.io_strs import linebreaker
    axbar.set_ylabel(col.replace(' ','\n'))
    if logx:
        if hist:
            for ax in axes.flat:
                ax.set_xscale("log")
                if grid:
                    ax.set_axisbelow(False)
        else:
            axes.set_xscale("log")
            if grid:
                axes.set_axisbelow(False)
            
def plot_barh_stacked_percentage(df1,cols_y,ax=None):
    dplot=pd.DataFrame({k:df1.drop_duplicates(k)[f'{k} '].value_counts() for k in cols_y})
    dplot_=dplot.apply(lambda x:x/x.sum()*100,axis=0)
    d=dplot.sum().to_dict()
    dplot_=dplot_.rename(columns={k:f"{k}\n(n={d[k]})" for k in d})
    dplot_.index.name='subset'
    if ax is None: ax=plt.subplot()
    dplot_.T.plot.barh(stacked=True,ax=ax)
    ax.legend(bbox_to_anchor=[1,1])
    ax.set(xlim=[0,100],xlabel='%')
    _=[ax.text(1,y,f"{s:.0f}%",va='center') for y,s in enumerate(dplot_.iloc[0,:])]
    return ax               
def plot_bar_intersections(dplot,cols,col_index):
    """
    upset
    """
    dplot=dplot.dropna(subset=[col_index]).drop_duplicates(subset=[col_index])
    ds=dplot.groupby(cols).agg({col_index:lambda x: len(np.unique(x))})[col_index]
    ds=ds/len(dplot[col_index].unique())*100
    # from rohan.dandage.io_strs import linebreaker
    # ds.index.names=[linebreaker(s,break_pt=25) for s in ds.index.names]
    import upsetplot as up
    fig=plt.figure(figsize=[10,10])
    d=up.plot(ds[ds>1],
            sort_by='cardinality',
           sort_categories_by=None,
             element_size=40,
              facecolor='gray',
             ) 
    d['totals'].set_visible(False)
    d['intersections'].set(ylabel=f'{col_index}s %')
    d['intersections'].get_children()[0].set_color("#f55f5f")
    d['intersections'].text(-0.25,ds.max(),f"{ds.max():.1f}%",
                            ha='left',va='bottom',color="#f55f5f")    
    return d
