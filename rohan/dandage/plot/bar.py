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
            
               