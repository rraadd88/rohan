import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from rohan.lib.plot.ax_ import *

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
    from rohan.lib.io_strs import linebreaker
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

def plot_barh_stacked_percentage(df1,coly,colannot,
                     color=None,
                     yoff=0,
                     ax=None,
                     ):
    """
    :param dplot: values sum to 100% in rows
    :param coly: yticklabels, e.g. retained and dropped 
    :param colannot: col to annot
    """
    from rohan.lib.plot.ax_ import get_ticklabel2position
    from rohan.lib.plot.colors import get_colors_default
    if color is None:
        color=get_colors_default()[0]
    ax=plt.subplot() if ax is None else ax
    df2=df1.set_index(coly).apply(lambda x: (x/sum(x))*100, axis=1)
    ax=df2.plot.barh(stacked=True,ax=ax)
    ticklabel2position=get_ticklabel2position(ax,'y')
    from rohan.lib.plot.colors import saturate_color
    _=df2.reset_index().apply(lambda x: ax.text(1,
                                                  ticklabel2position[x[coly]]-yoff,
                                                  f"{x[colannot]:.1f}%",ha='left',va='center',
                                                 color=saturate_color(color,2),
                                               ),
                                axis=1)
    ax.legend(bbox_to_anchor=[1,1],title=df1.columns.name)
    d1=df1.set_index(coly).T.sum().to_dict()
    ax.set(xlim=[0,100],xlabel='%',
          yticklabels=[f"{t.get_text()}\n(n={d1[t.get_text()]})" for t in ax.get_yticklabels()])
    return ax

def plot_barh_stacked_percentage_intersections(df0,
                                               colxbool='paralog',
                                               colybool='essential',
                                               colvalue='value',
                                               colid='gene id',
                                               colalt='singleton',
                                               coffgroup=0.95,
                                               colgroupby='tissue',
                                              ):
    ##1 threshold for value by group
    def apply_(df):
        coff=np.quantile(df.loc[df[colybool],colvalue],coffgroup)
        df[colybool]=df[colvalue]<coff
        return df
    df1=df0.groupby(colgroupby).progress_apply(apply_)
    ##2 % 
    df2=df1.groupby([colid,colxbool]).agg({colybool: perc}).reset_index().rename(columns={colybool:f'% {colgroupby}s with {colybool}'},
                                                                                        errors='raise')
    coly=f"% of {colgroupby}s"
    ##3 bin y
    df2[coly]=pd.cut(df2[f'% {colgroupby}s with {colybool}'],bins=pd.interval_range(0,100,4),)
    ##3 % sum
    df3=df2.groupby(coly)[colxbool].agg([sum]).rename(columns={'sum':colxbool})
    dplot=df3.join(df2.groupby(coly).size().to_frame('total'))
    dplot[colalt]=dplot['total']-dplot[colxbool]
    dplot.index=[str(i) for i in dplot.index]
    dplot.index.name=coly
    dplot.columns.name=f"{colid} type"
    dplot=dplot.sort_values(coly,ascending=False)
    dplot=dplot.reset_index()
#     from rohan.lib.plot.bar import plot_barh_stacked_percentage
    fig,ax=plt.subplots(figsize=[3,3])
    plot_barh_stacked_percentage(df1=dplot.loc[:,[coly,colxbool,colalt]],
                                coly=coly,
                                colannot=colxbool,
                                ax=ax)
    set_ylabel(ax)
    ax.set(xlabel=f'% of {colid}s',
          ylabel=None)
    return ax

def plot_bar_intersections(dplot,cols=None,colvalue=None,
                            min_intersections=5,
                            figsize=[5,5],text_width=22,
                            sort_by='cardinality',
                            sort_categories_by=None,#'cardinality',
                            element_size=40,
                            facecolor='gray',
                            bari_annot=0,
                            totals=False,
                            test=False,
                            **kws,
                          ):
    """
    upset
    sort_by:{‘cardinality’, ‘degree’}
    If ‘cardinality’, subset are listed from largest to smallest. If ‘degree’, they are listed in order of the number of categories intersected.

    sort_categories_by:{‘cardinality’, None}
    Whether to sort the categories by total cardinality, or leave them in the provided order.

    Ref: https://upsetplot.readthedocs.io/en/stable/api.html
    """
    if isinstance(dplot,pd.DataFrame):
        assert(isinstance(colvalue,str))
        assert(all(dplot.loc[:,cols].dtypes=='bool'))
#         dplot=dplot.log.dropna(subset=[colvalue]).log.drop_duplicates(subset=[colvalue])
#         ds=dplot.groupby(cols).agg({colvalue:'nunique'})[colvalue]
#         ds=ds/len(dplot[colvalue].unique())*100
        df1=dplot.melt(id_vars=colvalue,
                  value_vars=cols)
        if test:print(df1.shape,end='')
        df1=df1.loc[df1['value'],:]
        if test:print(df1.shape)
        ds=df1.rd.check_intersections('variable','paralog pair')
    elif isinstance(dplot,pd.Series):
        ds=dplot.copy()
    ds2=(ds/ds.sum())*100
    # from rohan.lib.io_strs import linebreaker
    # ds.index.names=[linebreaker(s,break_pt=25) for s in ds.index.names]
    import upsetplot as up
#     fig=plt.figure()
    d=up.plot(ds2,
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
    d['intersections'].set(ylabel=f"{colvalue}s %\n(total={ds.sum()})",
                          xlim=[-0.5,min_intersections-0.5],
                          )
    d['intersections'].get_children()[bari_annot].set_color("#f55f5f")
    if sort_by=='cardinality':
        y=ds2.max()
    elif sort_by=='degree':
        y=ds2.loc[tuple([True for i in ds2.index.names])]
    print(sort_by,y)
    d['intersections'].text(-0.25,y,f"{y:.1f}%",
                            ha='left',va='bottom',color="#f55f5f",zorder=10)    
#     d['intersections'].text(d['intersections'].get_xlim()[1],
#                             d['intersections'].get_ylim()[1],
#                             f"total {colvalue}s\n={ds.sum()}",
#                             ha='right',va='bottom',color="lightgray")
    return d
