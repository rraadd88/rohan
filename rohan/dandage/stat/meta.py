from rohan.global_imports import *

from rohan.dandage.plot.annot import get_dmetrics

def compare_bools_scores(df0,colbool,colscore,colpivotindex,colpivotcolumns):
    from rohan.dandage.stat.binary import compare_bools_jaccard_df
    df1=pd.concat({'correlation':dmap2lin(df0.pivot_table(columns=colpivotcolumns,index=colpivotindex,values=colscore).corr(method='pearson')),
           'jaccard index':dmap2lin(compare_bools_jaccard_df(df0.pivot_table(columns=colpivotcolumns,index=colpivotindex,values='interaction bool')))},axis=0,names=['metric type']).reset_index()
    # .rename(columns={'index':'row',:'col'})
    df1=df1.loc[(df1['index']!=df1['column']),:]
    metric2index={'correlation':list(itertools.combinations(df1['index'].unique(),2)),}
    metric2index['jaccard index']=[t[::-1] for t in metric2index['correlation']]
    df2=pd.concat([df1.loc[(df1['metric type']==metric),:].set_index(['index','column']).loc[metric2index[metric],:] for metric in metric2index],
             axis=0).reset_index()
    df2['value']=df2['value'].apply(float)
#     print(df2.head())
    dmap=df2.pivot_table(index='index',columns='column',values='value').fillna(1)
    dmap.columns.name='$r_p$'
    dmap.index.name='Jaccard index'
    return dmap

def compare_by_zscore(df,col1,col2,coldn='comparison',log=True):
    if log:
        from rohan.dandage.stat.transform import plog
        df[f'{coldn} ratio {col1}/{col2} (log2 scale)']=(df[col1].apply(lambda x: plog(x,p=0.5,base=2))-df[col2].apply(lambda x: plog(x,p=0.5,base=2)))
    else:
        df[f'{coldn} ratio {col1}/{col2} (log2 scale)']=df[col1]-df[col2]
    from scipy.stats import zscore
    df[f'{coldn} ratio {col1}/{col2} (log2 scale) zscore']=zscore(df[f'{coldn} ratio {col1}/{col2} (log2 scale)'])
    df[f'{coldn} ratio {col1}/{col2} (log2 scale) zscore significantly high']=df[f'{coldn} ratio {col1}/{col2} (log2 scale) zscore']>=2
    df[f'{coldn} ratio {col1}/{col2} (log2 scale) zscore significantly low']=df[f'{coldn} ratio {col1}/{col2} (log2 scale) zscore']<=-2
    return df

def get_stats_by_bins(df,colx,coly,fun,bins=4):
    from rohan.dandage.stat.variance import confidence_interval_95
    from rohan.dandage.stat.transform import get_qbins
    print(df.shape,end=' ')
    df=df.dropna(subset=[colx,coly])
    print(df.shape)
    dn2df={}
    dn2df['all']=pd.Series({'all':fun(df[colx],df[coly])})
    for col in [colx,coly]:
        df[f"{col} bins"]=get_qbins(ds=df[col],bins=bins,
                                    value='mid')
#         qcut(df[col],bins,duplicates='drop').apply(lambda x: x.mid)
        dn2df[f"{col} bins"]=df.groupby([f"{col} bins"]).apply(lambda df : fun(df[colx],df[coly]))
        
    return pd.DataFrame(pd.concat(dn2df))

import numpy as np
from rohan.dandage.stat.diff import diff,balance 
def aggbypair(a,b,absolute=True): return np.mean([a,b]),diff(a,b,absolute=absolute),balance(a,b)

def agg_paired_values(a,b,sort=True,logscaled=False):
    if sort:
        a,b=sorted([a,b])
    d={}
    d['mean']=(a+b)/2
    d['min']=min([a,b])
    d['max']=max([a,b])
    if logscaled:
        d['difference']=b-a
    else:
        d['ratio']=b/a
    return d