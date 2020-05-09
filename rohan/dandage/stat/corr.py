import pandas as pd
import numpy as np
import scipy as sc
import logging

def df2corrmean(df):
    return df.corr(method='spearman').replace(1,np.nan).mean().mean()

from scipy.stats import spearmanr,pearsonr
def corrdfs(df1,df2,method):
    """
    df1 in columns
    df2 in rows    
    """
    if len(df1)!=len(df2):
        from rohan.dandage.io_sets import list2intersection
        index_common=list2intersection([df1.index.tolist(),df2.index.tolist()])
        df1=df1.loc[index_common,:]
        df2=df2.loc[index_common,:]
        logging.warning('intersection of index is used for correlations')
    dcorr=pd.DataFrame(columns=df1.columns,index=df2.columns)
    dpval=pd.DataFrame(columns=df1.columns,index=df2.columns)
    for c1 in df1:
        for c2 in df2:
            if method=='spearman':
                dcorr.loc[c2,c1],dpval.loc[c2,c1]=spearmanr(df1[c1],df2[c2],
                                                        nan_policy='omit'
                                                       )
            elif method=='pearson':
                dcorr.loc[c2,c1],dpval.loc[c2,c1]=pearsonr(df1[c1],df2[c2],
#                                                         nan_policy='omit'
                                                       )
                
    if not df1.columns.name is None:
        dcorr.columns.name=df1.columns.name
        dpval.columns.name=df1.columns.name
    if not df2.columns.name is None:
        dcorr.index.name=df2.columns.name
        dpval.index.name=df2.columns.name
    return dcorr,dpval

def get_spearmanr(x,y):
    t=sc.stats.spearmanr(x,y,nan_policy='omit')
#     print(t)
#     print(t.pvalue)    
    return t.correlation,float(t.pvalue)

from rohan.dandage.plot.annot import pval2annot
def get_spearmanr_str(x,y):    
    r,p=get_spearmanr(x,y)
    return f"$\\rho$={r:.1e} ({pval2annot(p,fmt='<')})".replace('\n','')

def get_corr_str(x,y,method='spearman'):
    """
    TODO bootstraping
    """
    r,p=getattr(sc.stats,f"{method}r")(x, y,nan_policy='omit')
    return f"$r_{method[0]}$={r:.2f}\n{pval2annot(p,fmt='<',linebreak=False)}"

def get_partial_corrs(df,xs,ys):
    """
    xs=['protein expression balance','coexpression']
    ys=[
    'coexpression',
    'combined_score',['combined_score','coexpression'],]

    """
    import pingouin as pg
    chunks=np.array_split(df.sample(frac=1,random_state=88),5)
    df1=pd.concat({chunki:pd.concat({"$r_{s}$"+f" {x} versus "+(y if isinstance(y,str) else f"{y[0]} (corrected for {' '.join(y[1:])})"):pg.partial_corr(data=chunk, x=x, y=(y if isinstance(y,str) else y[0]),y_covar=(None if isinstance(y,str) else y[1:]), tail='two-sided', method='spearman') for x,y in list(itertools.product(xs,ys)) if x!=(y if isinstance(y,str) else y[1])},axis=0) for chunki,chunk in enumerate(chunks)},axis=0)
    df1.index.names=['chunk #','correlation name','correlation method']
    return df1.reset_index()
