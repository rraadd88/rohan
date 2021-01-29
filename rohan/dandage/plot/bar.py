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
            
# def plot_barh_stacked_percentage(df1,cols_y,ax=None):
#     dplot=pd.DataFrame({k:df1.drop_duplicates(k)[f'{k} '].value_counts() for k in cols_y})
#     dplot_=dplot.apply(lambda x:x/x.sum()*100,axis=0)
#     d=dplot.sum().to_dict()
#     dplot_=dplot_.rename(columns={k:f"{k}\n(n={d[k]})" for k in d})
#     dplot_.index.name='subset'
#     if ax is None: ax=plt.subplot()
#     dplot_.T.plot.barh(stacked=True,ax=ax)
#     ax.legend(bbox_to_anchor=[1,1])
#     ax.set(xlim=[0,100],xlabel='%')
#     _=[ax.text(1,y,f"{s:.0f}%",va='center') for y,s in enumerate(dplot_.iloc[0,:])]
#     return ax

def plot_barh_stacked_percentage(dplot,col_annot,
                     color=None,
                     yoff=0.3,
                     ax=None,
                     ):
    from rohan.dandage.plot.ax_ import get_ticklabel2position
    from rohan.dandage.plot.colors import get_colors_default
    if color is None:
        color=get_colors_default()[0]
    ax=plt.subplot() if ax is None else ax
    ax=dplot.plot.barh(stacked=True,ax=ax)
    ticklabel2position=get_ticklabel2position(ax,'y')
    _=dplot.reset_index().apply(lambda x: ax.text(0,
                                                  ticklabel2position[x['index']]-yoff,
                                                  f"{x[col_annot]:.1f}%",ha='left',va='top',
                                                 color=color),
                                axis=1)
    ax.legend(bbox_to_anchor=[1,1],title=dplot.columns.name)
    ax.set(xlim=[0,100],xlabel='%')
    return ax

def plot_bar_intersections(dplot,cols=None,col_index=None,
                            min_intersections=5,
                            figsize=[5,5],text_width=22,
                            sort_by='cardinality',
                            sort_categories_by=None,#'cardinality',
                            element_size=40,
                            facecolor='gray',
                            totals=False,
                            test=False,
                            **kws,
                          ):
    """
    upset
    """
    if isinstance(dplot,pd.DataFrame):
        assert(isinstance(col_index,str))
        assert(all(dplot.loc[:,cols].dtypes=='bool'))
#         dplot=dplot.log.dropna(subset=[col_index]).log.drop_duplicates(subset=[col_index])
#         ds=dplot.groupby(cols).agg({col_index:'nunique'})[col_index]
#         ds=ds/len(dplot[col_index].unique())*100
        df1=dplot.melt(id_vars=col_index,
                  value_vars=cols)
        if test:print(df1.shape,end='')
        df1=df1.loc[df1['value'],:]
        if test:print(df1.shape)
        ds=df1.rd.check_intersections('variable','paralog pair')
    elif isinstance(dplot,pd.Series):
        ds=dplot.copy()
    ds=(ds/ds.sum())*100
    # from rohan.dandage.io_strs import linebreaker
    # ds.index.names=[linebreaker(s,break_pt=25) for s in ds.index.names]
    import upsetplot as up
#     fig=plt.figure()
    d=up.plot(ds,
              figsize=figsize,
              text_width=text_width,
            sort_by=sort_by,
            sort_categories_by=sort_categories_by,
              facecolor=facecolor,element_size=element_size,
              **kws,
             ) 
    d['totals'].set_visible(totals)
    if totals:
        d['totals'].set(ylim=d['totals'].get_ylim()[::-1],
                       xlabel='%')
    d['intersections'].set(ylabel=f'{col_index}s %',
                          xlim=[-0.5,min_intersections-0.5],
                          )
    d['intersections'].get_children()[0].set_color("#f55f5f")
    d['intersections'].text(-0.25,ds.max(),f"{ds.max():.1f}%",
                            ha='left',va='bottom',color="#f55f5f")    
    return d
