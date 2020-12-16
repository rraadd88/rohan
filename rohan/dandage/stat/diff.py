import pandas as pd
import numpy as np
import scipy as sc
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

## for linear dfs
def get_pval(df,
             colvalue='value',
             colsubset='subset',
             colvalue_bool=False,
             subsets=None):
    """
    either colsubset or subsets are needed 
    """
    if not ((colvalue in df) and (colsubset in df)):
        logging.error(f"colvalue or colsubset not found in df: {colvalue} or {colsubset}")
    if subsets is None:
        subsets=df[colsubset].unique()
    if len(subsets)!=2:
        logging.error('need only 2 subsets')
        return
    else:
        df=df.loc[df[colsubset].isin(subsets),:]
    if colvalue_bool and df[colvalue].dtype==bool:
        df[colsubset]=df[colsubset]==subsets[0]
    if colvalue_bool and not df[colvalue].dtype==bool:        
        logging.warning(f"colvalue_bool {colvalue} is not bool")
        return
    if not colvalue_bool:
        try:
            return sc.stats.mannwhitneyu(
            df.loc[(df[colsubset]==subsets[0]),colvalue],
            df.loc[(df[colsubset]==subsets[1]),colvalue],
                alternative='two-sided')
        except:
            return np.nan,np.nan
    else:
        ct=pd.crosstab(df[colvalue],df[colsubset])
        if ct.shape==(2,2):
            return sc.stats.fisher_exact(ct)
        
def get_stats(df1=None,
              colsubset='subset',
              colvalue='value',
              subsets=None,
              cols_subsets=['subset1', 'subset2'],
              df2=None,
              stats=[np.mean,np.median,np.var]+[len],
              debug=False,
             ):
    """
    :params df2: pairs of comparisons between subsets
    either colsubset or subsets are needed 
    """
    if debug:
        subsets=list('abcd')
        np.random.seed(88)
        df1=pd.concat({s:pd.Series([np.random.uniform(0,si+1) for _ in range((si+1)*10)]) for si,s in enumerate(subsets)},
                 axis=0).reset_index(0).rename(columns={'level_0':'subset',0:'value'})
        
    if not ((colvalue in df1) and (colsubset in df1)):
        logging.error(f"colvalue or colsubset not found in df: {colvalue} or {colsubset}")
        return
    if subsets is None:
        subsets=df1[colsubset].unique()
    colvalue_bool=df1[colvalue].dtype==bool
    if df2 is None:
        import itertools
        df2=pd.DataFrame([t for t in list(itertools.permutations(subsets,2))])
        df2.columns=cols_subsets
    df2=df2.groupby(cols_subsets).apply(lambda df: get_pval(df1,colvalue=colvalue,
                                                            colsubset=colsubset,
                                                            subsets=df.name,
                                                            colvalue_bool=colvalue_bool,
                                                           )).apply(pd.Series)
    df2=df2.rename(columns={0:f"stat ({'MWU' if not colvalue_bool else 'FE'} test)",
                            1:f"P ({'MWU' if not colvalue_bool else 'FE'} test)",
                            }).reset_index()
    from rohan.dandage.io_dfs import merge_paired
    df3=merge_paired(df2,
        df=df1.groupby([colsubset])[colvalue].agg(stats if not colvalue_bool else [sum,len]).reset_index(),
        left_ons=cols_subsets,
        right_on=colsubset,
        right_ons_common=[],
        suffixes=[f" {s}" for s in cols_subsets],
        how='left',
        dryrun=False,
        test=False,
    #     **kws_merge,
    )
    return df3
