import pandas as pd
import numpy as np
import logging

def get_subset2metrics(df,colvalue,colsubset,colindex,outstr=False,subset_control=None):
    if subset_control is None:
        subset_control=df[colsubset].unique().tolist()[-1]
    from scipy.stats import mannwhitneyu    
    df1=df.merge(df.loc[(df[colsubset]==subset_control),:],on=colindex, 
                 how='left',
                 suffixes=['',' reference'],
                )
    subset2metrics=df1.groupby(colsubset).apply(lambda df : mannwhitneyu(df[colvalue],df[f"{colvalue} reference"],
                                                     alternative='two-sided')).apply(pd.Series)[1].to_dict()
    if subset2metrics[subset_control]<0.9:
        logging.warning(f"pval for reference condition vs reference condition = {subset2metrics[subset_control]}. shouldn't be. check colindex")
    subset2metrics={k:subset2metrics[k] for k in subset2metrics if k!=subset_control}
    if outstr:
        from rohan.dandage.plot.annot import pval2annot
        subset2metrics={k: pval2annot(subset2metrics[k],
                                      fmt='<',
                                      alternative='two-sided',
                                      linebreak=False) for k in subset2metrics}
    return subset2metrics

def get_ratio_sorted(a,b):
    l=sorted([a,b])
    if l[1]!=0:
        return l[1]/l[0]

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

def get_paired_values_stats(a,b,sort=False,logscaled=False):
    if sort:
        a,b=sorted([a,b])
    d={}
    d['sum']=a+b
    d['min']=a+b
    d['max']=a+b
    if logscaled:
        d['difference']=b-a
    else:
        d['ratio']=b/a
    return d