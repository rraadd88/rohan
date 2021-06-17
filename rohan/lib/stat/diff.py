import pandas as pd
import numpy as np
import scipy as sc
import logging
from icecream import ic as info
from rohan.lib import to_class,stat
from rohan.lib.io_sets import *

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
        from rohan.lib.plot.annot import pval2annot
        subset2metrics={k: pval2annot(subset2metrics[k],
                                      fmt='<',
                                      alternative='two-sided',
                                      linebreak=False) for k in subset2metrics}
    return subset2metrics

def get_ratio_sorted(a,b):
    l=sorted([a,b])
    if l[0]!=0 and l[1]!=0:
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
#         try:
        x,y=df.loc[(df[colsubset]==subsets[0]),colvalue].tolist(),df.loc[(df[colsubset]==subsets[1]),colvalue].tolist()
        if len(x)!=0 and len(y)!=0 and (nunique(x+y)!=1):
            return sc.stats.mannwhitneyu(x,y,alternative='two-sided')
        else:
            #if empty list: RuntimeWarning: divide by zero encountered in double_scalars  z = (bigu - meanrank) / sd
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
             verb=False,
             ):
    """
    Either MWU or FE
    :param df2: pairs of comparisons between subsets
    either colsubset or subsets are needed 
    """        
    if not ((colvalue in df1) and (colsubset in df1)):
        logging.error(f"colvalue or colsubset not found in df: {colvalue} or {colsubset}")
        return
    if subsets is None:
        subsets=df1[colsubset].unique()
    if len(subsets)<2:
        logging.error("len(subsets)<2")
        return
    colvalue_bool=df1[colvalue].dtype==bool
    if df2 is None:
        import itertools
        df2=pd.DataFrame([t for t in list(itertools.permutations(subsets,2))] if len(subsets)>2 else [subsets])
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
    from rohan.lib.io_dfs import merge_paired
    df3=merge_paired(df2,
        df=df1.groupby([colsubset])[colvalue].agg(stats if not colvalue_bool else [sum,len]).reset_index(),
        left_ons=cols_subsets,
        right_on=colsubset,
        right_ons_common=[],
        suffixes=[f" {s}" for s in cols_subsets],
        how='left',
        dryrun=False,
        test=False,
        verb=verb,
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
              axis=1, # concat 
              **kws_get_stats):
    """
    :param axis: 1 if different tests else use 0.
    """
#     from rohan.lib.io_dfs import to_table
#     to_table(df1,'test/get_stats.tsv')
#     stats=[s for s in stats if not s in [sum,len]]
#     from tqdm import tqdm
    dn2df={}
    for colvalue in cols_value:
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
    if len(dn2df.keys())==0:
        return 
    df3=pd.concat(dn2df,
                  ignore_index=False,
                  axis=axis,
                  verify_integrity=True,
                  names=None if axis==1 else ['variable'],
                 )
    if axis==1:
        df3=df3.reset_index().rd.flatten_columns()
    return df3

@to_class(stat)
def get_significant_changes(df1,alpha=0.025,
                            changeby="",
                            fdr=True,
                            value_aggs=['mean','median'],
                           ):
    """
    groupby to get the comparable groups 
    # if both mean and median are high
    :param changeby: "" if check for change by both mean and median
    """    
    if df1.filter(regex='|'.join([f"{s} subset(1|2)" for s in value_aggs])).shape[1]:
        for s in value_aggs:
            df1[f'difference between {s} (subset1-subset2)']=df1[f'{s} subset1']-df1[f'{s} subset2']
        ## call change if both mean and median are changed
        df1.loc[((df1.filter(like=f'difference between {changeby}')>0).T.sum()==2),'change']='increase'
        df1.loc[((df1.filter(like=f'difference between {changeby}')<0).T.sum()==2),'change']='decrease'
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
            df1.loc[df1[f"change is significant ({test} test{(', FDR corrected' if fdr else '')})"],f"significant change ({test} test{(', FDR corrected' if fdr else '')})"]=df1.loc[df1[f"change is significant ({test} test{(', FDR corrected' if fdr else '')})"],'change']
            df1[f"significant change ({test} test{(', FDR corrected' if fdr else '')})"]=df1[f"significant change ({test} test{(', FDR corrected' if fdr else '')})"].fillna('ns')
    return df1

@to_class(stat)
def apply_get_significant_changes(df1,cols_value,
                                    cols_groupby, # e.g. genes id
                                    cols_grouped, # e.g. tissue
                                    fast=False,
                                    **kws,
                                    ):
    d1={}
    from tqdm import tqdm
    for c in tqdm(cols_value):
        df1_=df1.set_index(cols_groupby).filter(regex=f"^{c} .*").filter(regex="^(?!("+' |'.join([s for s in cols_value if c!=s])+")).*")
        df1_=df1_.rd.renameby_replace({f'{c} ':''}).reset_index()
        d1[c]=getattr(df1_.groupby(cols_groupby),'apply' if not fast else 'parallel_apply')(lambda df: get_significant_changes(df,**kws))
    d1={k:d1[k].set_index(cols_groupby) for k in d1}
    ## to attach the info
    d1['grouped']=df1.set_index(cols_groupby).loc[:,cols_grouped]
    df2=pd.concat(d1,
                  ignore_index=False,
                  axis=1,
                  verify_integrity=True,
                 )    
    df2=df2.rd.flatten_columns().reset_index()
    assert(not df2.columns.duplicated().any())
    return df2

@to_class(stat)
def binby_pvalue_coffs(df1,coffs=[0.01,0.05,0.25],
                      color=False,
                       testn='MWU test, FDR corrected',
                       colindex='genes id',
                       colgroup='tissue',
                      preffix='',
                      colns=None, # plot as ns, not counted
                      palette=None,#['#f55f5f','#ababab','#046C9A',],
                      ):
    if palette is None:
        from rohan.lib.plot.colors import get_colors_default
        palette=get_colors_default()[:3]
    assert(len( palette)==3)
    coffs=np.array(sorted(coffs))
    # df1[f'{preffix}P (MWU test, FDR corrected) bin']=pd.cut(x=df1[f'{preffix}P (MWU test, FDR corrected)'],
    #       bins=[0]+coffs+[1],
    #        labels=coffs+[1],
    #        right=False,
    #       ).fillna(1)
    from rohan.lib.plot.colors import saturate_color
    d1={}
    for i,coff in enumerate(coffs[::-1]):
        col=f"{preffix}significant change, P ({testn}) < {coff}"
        df1[col]=df1.apply(lambda x: 'increase' if ((x[f'{preffix}P ({testn})']<coff) \
                                                    and (x[f'{preffix}difference between mean (subset1-subset2)']>0))\
                                else 'decrease' if ((x[f'{preffix}P ({testn})']<coff) \
                                                    and (x[f'{preffix}difference between mean (subset1-subset2)']<0))\
                                else 'ns', axis=1)
        if color:
            if i==0:
                df1.loc[(df1[col]=='ns'),'c']=palette[1]
            saturate=1-((len(coffs)-(i+1))/len(coffs))
            d2={}
            d2['increase']=saturate_color(palette[0],saturate)
            d2['decrease']=saturate_color(palette[2],saturate)
            d1[coff]=d2
            df1['c']=df1.apply(lambda x: d2[x[col]] if x[col] in d2 else x['c'],axis=1)
            assert(df1['c'].isnull().sum()==0)
    if color:
        import itertools
        from rohan.lib.stat.transform import rescale
        d3={}
        for i,(k,coff) in enumerate(list(itertools.product(['increase','decrease'],coffs))):
            col=f"{preffix}significant change, P ({testn}) < {coff}"
            d4={}
            d4['y alpha']=rescale(1-(list(coffs).index(coff))/len(coffs),[0,1],[0.5,1])
            d4['y']=-1*(np.log10(coff))
            d4['y text']=f" P < {coff}"
            d4['x']=df1.loc[(df1[col]==k),f'{preffix}difference between mean (subset1-subset2)'].min() if k=='increase' \
                                        else df1.loc[(df1[col]==k),f'{preffix}difference between mean (subset1-subset2)'].max()
            d4['change']=k
            d4['text']=f"{df1.loc[(df1[col]==k),colindex].nunique()}/{df1.loc[(df1[col]==k),colgroup].nunique()}"
            d4['color']=d1[coff][k]
            d3[i]=d4
        df2=pd.DataFrame(d3).T
    if not colns is None:
        df1.loc[df1[colns],'c']=palette[1]
#     info(df1.shape,df1.shape)
    return df1,df2

# from rohan.lib.plot.diff import plot_stats_diff