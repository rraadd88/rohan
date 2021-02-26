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
def get_demo_data():
    subsets=list('abcd')
    np.random.seed(88)
    df1=pd.concat({s:pd.Series([np.random.uniform(0,si+1) for _ in range((si+1)*100)]) for si,s in enumerate(subsets)},
             axis=0).reset_index(0).rename(columns={'level_0':'subset',0:'value'})
    df1['bool']=df1['value']>df1['value'].quantile(0.5)
    return df1

def get_pval(df,
             colvalue='value',
             colsubset='subset',
             colvalue_bool=False,
             subsets=None,
            test=False):
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
#         df.to_csv('test/get_pval_bool.tsv',sep='\t')
        ct=pd.crosstab(df[colvalue],df[colsubset])
        if ct.shape==(2,2):
            ct=ct.sort_index(axis=0,ascending=False).sort_index(axis=1,ascending=False)
            if test:
                print(ct)
            return sc.stats.fisher_exact(ct)
        else:
            return np.nan,np.nan            
def get_stat(df1,
              colsubset,
              colvalue,
              subsets=None,
              cols_subsets=['subset1', 'subset2'],
              df2=None,
              stats=[np.mean,np.median,np.var]+[len],
#               debug=False,
             ):
    """
    Either MWU or FE
    :params df2: pairs of comparisons between subsets
    either colsubset or subsets are needed 
    """        
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
#     print(df2.shape)
#     print(df2.groupby(cols_subsets).size())
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
def get_stats(df1,
              colsubset,
              cols_value,
              subsets=None,
              cols_subsets=['subset1', 'subset2'],
              df2=None,
              stats=[np.mean,np.median,np.var,len],
              **kws_get_stats):
#     from rohan.dandage.io_dfs import to_table
#     to_table(df1,'test/get_stats.tsv')
#     stats=[s for s in stats if not s in [sum,len]]
    from tqdm import tqdm
    dn2df={}
    for colvalue in tqdm(cols_value):
        df1_=df1.dropna(subset=[colsubset,colvalue])
        if len(df1_[colsubset].unique())>1:
            dn2df[colvalue]=get_stat(df1_,
                          colsubset=colsubset,
                          colvalue=colvalue,
                          subsets=subsets,
                          cols_subsets=cols_subsets,
                          df2=df2,
                          stats=stats,
                          **kws_get_stats,
                         ).set_index(cols_subsets)
        else:
            logging.warning(f"not processed: {colvalue}; probably because of dropna")
    df3=pd.concat(dn2df,
          ignore_index=False,
#           join='outer',
                  axis=0,
                 names=['variable'])
#     df3=df3.droplevel(0,axis=1)
    df3=df3.reset_index()
    return df3

def get_significant_changes(df1,alpha=0.025,
                            changeby="",
                            fdr=True,
                           ):
    """
    groupby to get the comparable groups 
    # if both mean and median are high
    :param changeby: "" if check for change by both mean and median
    """    
    if any([f'{s} subset1' in df1 for s in ['mean','median']]) :
        for s in ['mean','median']:
            df1[f'difference between {s} (subset1-subset2)']=df1[f'{s} subset1']-df1[f'{s} subset2']
        df1.loc[(df1.loc[:,df1.filter(like=f'difference between {changeby}').columns.tolist()]>0).apply(all,axis=1),'change']='increase'
        df1.loc[(df1.loc[:,df1.filter(like=f'difference between {changeby}').columns.tolist()]<0).apply(all,axis=1),'change']='decrease'
        df1['change']=df1['change'].fillna('ns')
    from statsmodels.stats.multitest import multipletests
    for test in ['MWU','FE']:
        if not f'P ({test} test)' in df1:
            continue
        if fdr:
            df1[f'change is significant ({test} test, FDR corrected)'],df1[f'P ({test} test, FDR corrected)'],df1[f'{test} alphacSidak'],df1[f'{test} alphacBonf']=multipletests(df1[f'P ({test} test)'],
                                                                   alpha=alpha, method='fdr_bh',
                                                                   is_sorted=False,returnsorted=False)
        else:
            df1[f'change is significant ({test} test)']=df1[f'P ({test} test)']<alpha
        #     info(f"corrected alpha alphacSidak={alphacSidak},alphacBonf={alphacBonf}")
        if test!='FE':
            df1.loc[df1[f"change is significant ({test} test{(', FDR corrected' if fdr else '')})"],f"significant change ({test} test)"]=df1.loc[df1[f"change is significant ({test} test{(', FDR corrected' if fdr else '')})"],'change']
            df1[f"significant change ({test} test)"]=df1[f"significant change ({test} test)"].fillna('ns')
    return df1

# def annotate_difference(df1,cols_subset_comparison,cols_index,
#                        ):
#     cols_subset=['subset1','subset2']
#     df1.rd.check_duplicated(params['cols_index']+cols_subset)
#     for s in ['mean','median']:
#         df1[f'{s} difference (subset1-subset2)']=df1[f'{s} subset1']-df1[f'{s} subset2']
#     df1.loc[(df1.loc[:,df1.filter(like='difference').columns.tolist()]>0).apply(all,axis=1),'change']='increase'
#     df1.loc[(df1.loc[:,df1.filter(like='difference').columns.tolist()]<0).apply(all,axis=1),'change']='decrease'
#     df1['change']=df1['change'].fillna('ns')
#     df1=df1.groupby(cols_subset+params['cols_subset_comparison']).progress_apply(get_change)
# #     info(df1['change is significant (MWU test, FDR corrected)'].value_counts())
#     # df1['change'].value_counts()
#     # df1['significant change'].value_counts()
#     return df1

from rohan.dandage.plot.diff import plot_stats_diff