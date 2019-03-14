import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_value_counts(df,col,logx=False,
                      kws_hist={'bins':10},
                      kws_bar={},
                     grid=False,
                     axes=None,fig=None):
    dplot=pd.DataFrame(df[col].value_counts()).sort_values(by=col,ascending=True)
    if axes is None:
        fig, axes = plt.subplots(2,1, sharex=True, 
        #                          sharey=True,
                                       gridspec_kw={'height_ratios': [1, 6]},
                                       figsize=(4, 7),
                                )
    _=dplot.plot.hist(ax=axes[0],legend=False,**kws_hist
    #                   orientation="horizontal"
                     )
    _=dplot.plot.barh(ax=axes[1],legend=False,**kws_bar)
    if logx:
        for ax in axes.flat:
            ax.set_xscale("log")
            if grid:
                ax.set_axisbelow(False)