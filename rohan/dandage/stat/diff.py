import pandas as pd
import numpy as np

def get_subset2metrics(dplot,colvalue,colsubset,order,outstr=False):
    subset_control=order[-1]
#     from rohan.dandage.io_dfs import filter_rows_bydict
    from scipy.stats import mannwhitneyu    
    df0=dplot.loc[(dplot[colsubset]==subset_control),:]
#     df0=filter_rows_bydict(dplot,{colsubset: subset_control})
    cols_subsets=[colsubset]
    subset2metrics=dplot.loc[(dplot[colsubset]!=subset_control),:].groupby(cols_subsets).apply(lambda df : mannwhitneyu(df0[colvalue],df[colvalue],alternative='two-sided')).apply(pd.Series)[1].to_dict()
    if outstr:
        from rohan.dandage.plot.annot import pval2annot
        subset2metrics={k: pval2annot(subset2metrics[k],fmt='<',alternative='two-sided',linebreak=False) for k in subset2metrics}
    return subset2metrics

def diff(a,b,absolute=True): 
    diff=a-b
    if absolute:
        return abs(diff)
    else:
        return diff
def balance(a,b,absolute=True):
    sum_=a+b
    if sum_!=0:
        return 1-(diff(a,b,absolute=absolute)/(sum_))
    else:
        return np.nan
    
def get_col2metrics(df,colxs,coly,method='mannwhitneyu',alternative='two-sided'):
    """
    coly: two values
    """
    from scipy import stats
    class1,class2=df[coly].unique()
    d={}
    for colx in colxs:
        _,d[colx]=getattr(stats,method)(df.loc[(df[coly]==class1),colx],
                                       df.loc[(df[coly]==class2),colx],
                                       alternative=alternative)
    return d    