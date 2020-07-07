import pandas as pd
import numpy as np

def get_subset2metrics(dplot,colvalue,colsubset,order):
    subset_control=order[-1]
    from scipy.stats import mannwhitneyu
    df0=filter_rows_bydict(dplot,{colsubset: subset_control})
    cols_subsets=[colsubset]
    subset2metrics=dplot.loc[(dplot[colsubset]!=subset_control),:].groupby(cols_subsets).apply(lambda df : mannwhitneyu(df0[colvalue],df[colvalue],alternative='two-sided')).apply(pd.Series)[1].to_dict()
    from rohan.dandage.plot.annot import pval2annot
    subset2metrics={k: pval2annot(subset2metrics[k],fmt='<',alternative='two-sided',linebreak=False) for k in subset2metrics}
    return subset2metrics