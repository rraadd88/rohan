import pandas as pd
import numpy as np
import scipy as sc
import logging

from rohan.lib import to_class,rd

from scipy.stats import spearmanr,pearsonr
def get_spearmanr(x,y):
    t=sc.stats.spearmanr(x,y,nan_policy='omit')
    return t.correlation,float(t.pvalue)
def get_pearsonr(x,y):
    return sc.stats.pearsonr(x,y)

def get_corr_bootstrapped(x,y,method='spearman',ci_type='max'):
    from rohan.lib.stat.ml import get_cvsplits
    from rohan.lib.stat.variance import get_ci
    cv2xy=get_cvsplits(x,y,cv=5,outtest=False)
    rs=[globals()[f"get_{method}r"](**cv2xy[k])[0] for k in cv2xy]
    return np.mean(rs), get_ci(rs,ci_type=ci_type)

def get_corr(x,y,method='spearman',bootstrapped=False,ci_type='max',
             n=False,
            outstr=False):
    """
    between vectors
    """
    from rohan.lib.plot.annot import pval2annot
    from rohan.lib.io_strs import num2str
    if bootstrapped:
        r,ci=get_corr_bootstrapped(x,y,method=method,ci_type=ci_type)
        _,p=globals()[f"get_{method}r"](x, y)
        if not outstr:
            return r,ci
        else:
            return f"$r_{method[0]}$={r:.2f}$\pm${ci:.2f}{ci_type if ci_type!='max' else ''}\n{pval2annot(p,fmt='<',linebreak=False)}"
    else:
        r,p=globals()[f"get_{method}r"](x, y)
        if not outstr:
            return r,p
        else:
            return f"$r_{method[0]}$={r:.2f}\n{pval2annot(p,fmt='<',linebreak=False)}"+('' if not n else f"\nn="+num2str(num=len(x),magnitude=False))        
# def get_corr_str(x,y,method='spearman',bootstrapped=False,ci_type='max',
#             outstr=True):
#     return get_corr(x,y,method='spearman',bootstrapped=False,ci_type='max',
#             outstr=False):    

@to_class(rd)
def corr_within(df,
         method='spearman',
         ):
    """
    :returns: linear 
    """
    
    df1=df.corr(method=method).rd.fill_diagonal(filler=np.nan)
    df2=df1.melt(ignore_index=False)
    col=df2.index.name
    df2=df2.rename(columns={col:col+'2'}).reset_index().rename(columns={col:col+'1','value':f'$r_{method[0]}$'})
    return df2

@to_class(rd)
def corrdf(df1,
           colindex,
           colsample,
           colvalue,
           colgroupby=None,
           min_samples=1,
           **kws):
    """
    linear df
    """
    if len(df1[colsample].unique())<min_samples:
        return        
    if colgroupby is None:
        df2=corr_within(
            df1.pivot(index=colindex,columns=colsample,values=colvalue),
            **kws,
            )
    else:
        df2=df1.groupby(colgroupby).apply(lambda df: corr_within(
            df.pivot(index=colindex,columns=colsample,values=colvalue),
            **kws,
            )).rd.clean().reset_index(0)
    return df2

@to_class(rd)
def corr_between(df1,df2,method):
    """
    df1 in columns
    df2 in rows    
    """
    from rohan.lib.io_sets import list2intersection
    index_common=list2intersection([df1.index.tolist(),df2.index.tolist()])
    # get the common indices with loc. it sorts the dfs too.
    df1=df1.loc[index_common,:]
    df2=df2.loc[index_common,:]
    
    dcorr=pd.DataFrame(columns=df1.columns,index=df2.columns)
    dpval=pd.DataFrame(columns=df1.columns,index=df2.columns)
    from tqdm import tqdm
    for c1 in tqdm(df1.columns):
        for c2 in df2:
            if method=='spearman':
                dcorr.loc[c2,c1],dpval.loc[c2,c1]=get_spearmanr(df1[c1],df2[c2],)
            elif method=='pearson':
                dcorr.loc[c2,c1],dpval.loc[c2,c1]=get_pearsonr(df1[c1],df2[c2],)                
    dn2df={f'$r_{method[0]}$':dcorr,
          f'P ($r_{method[0]}$)':dpval,}
    dn2df={k:dn2df[k].melt(ignore_index=False) for k in dn2df}
    df3=pd.concat(dn2df,
             axis=0,
             )
    df3=df3.rename(columns={df1.columns.name:df1.columns.name+'2'}
              ).reset_index(1).rename(columns={df1.columns.name:df1.columns.name+'1'})
    df3.index.name='variable correlation'
    return df3.reset_index()

@to_class(rd)
def corrdfs(df1,df2,
           colindex,
           colsample,
           colvalue,
           colgroupby=None,
           min_samples=1,
           **kws):
    """
    linear df
    """
    if len(df1[colsample].unique())<min_samples or len(df2[colsample].unique())<min_samples:
        return
    if colgroupby is None:
        df3=corr_between(
            df1.pivot(index=colindex,columns=colsample,values=colvalue),
            df2.pivot(index=colindex,columns=colsample,values=colvalue),
            **kws,
            )
    else:
        df3=df1.groupby(colgroupby).apply(lambda df: corr_between(
            df.pivot(index=colindex,columns=colsample,values=colvalue),
            df2.loc[(df2[colgroupby]==df.name),:].pivot(index=colindex,columns=colsample,values=colvalue),
            **kws,
            )).reset_index(0)
    return df3

# def get_corr_str(r,p):
#     from rohan.lib.plot.annot import pval2annot
#     return f"$\\rho$={r:.1e} ({pval2annot(p,fmt='<')})".replace('\n','')

# def get_spearmanr_str(x,y):    
#     r,p=get_spearmanr(x,y)
#     return get_correlation_str(r,p)

## apply on paths
def get_corr_within(p,
           colmut,
           colindex,
           colsample,
           colvalue,
           colgroupby,
           force=False,
           **kws_replacemany):
    from rohan.lib.io_strs import replacemany
    from rohan.lib.io_files import dirname,basename,basenamenoext,exists
    from rohan.lib.io_dfs import read_table,to_table
    outp=replacemany(p,**kws_replacemany)
    if exists(outp) and not force: 
        return
#     info(outp)
    df01=read_table(p)
    if not len(df01[colsample].unique())>=3:
        return
    if colmut in df01:
        df01=df01.loc[((df01[colmut]=='no') & ~(df01['rearrangement fusion'])),:]
#     from rohan.lib.stat.corr import corrdf
    df1=corrdf(df01,
               colindex=colindex,
               colsample=colsample,
               colvalue=colvalue,
           colgroupby=colgroupby,
                    method='spearman'
          )
    to_table(df1,outp)

def get_corr_between(p,
           colmut,
           colindex,
           colsample,
           colvalue,
           colgroupby=None,
           force=False,
           **kws_replacemany):
    from rohan.lib.io_strs import replacemany
    from rohan.lib.io_files import dirname,basename,basenamenoext,exists
    from rohan.lib.io_dfs import read_table,to_table
    outp=p
    p=replacemany(p,**kws_replacemany)
    ps=[f"{dirname(p)}/{s}.pqt" for s in basenamenoext(outp).split('--')]    
    if exists(outp) and not force: 
        return
    dfs=[read_table(p) for p in ps]
    if colmut in dfs[0]:
        dfs=[df.loc[((df[colmut]=='no') & ~(df['rearrangement fusion'])),:] for df in dfs]
    if not all([len(df[colsample].unique())>=3 for df in dfs]):
        return
#     from rohan.lib.stat.corr import corrdfs
    df1=corrdfs(*dfs,
               colindex=colindex,
               colsample=colsample,
               colvalue=colvalue,
               colgroupby=colgroupby,                 
                 method='spearman')
    to_table(df1,outp)

## partial 

@to_class(rd)
def get_partial_corrs(df,xs,ys,method='spearman',splits=5):
    """
    xs=['protein expression balance','coexpression']
    ys=[
    'coexpression',
    'combined_score',['combined_score','coexpression'],]

    """
    import pingouin as pg
    import itertools
    chunks=np.array_split(df.sample(frac=1,random_state=88),splits)
    dn2df={}
    for chunki,chunk in enumerate(chunks):
        dn2df_={}
        for x,y in list(itertools.product(xs,ys)):
            if (x if isinstance(x,str) else x[0])!=(y if isinstance(y,str) else y[0]): 
                params=dict(
                        x=(x if isinstance(x,str) else x[0]),x_covar=(None if isinstance(x,str) else x[1:]), 
                        y=(y if isinstance(y,str) else y[0]),y_covar=(None if isinstance(y,str) else y[1:]), )
                label=str(params)#+(x if isinstance(x,str) else f"{x[0]} (corrected for {' '.join(x[1:])})")+" versus "+(y if isinstance(y,str) else f"{y[0]} (corrected for {' '.join(y[1:])})")
                dn2df_[label]=pg.partial_corr(data=chunk, 
                                        tail='two-sided', method=method,
                                         **params)
#                 print(dn2df_[label])
#                 print(params)
                for k in params:
                    dn2df_[label].loc[method,k]=params[k] if isinstance(params[k],str) else str(params[k])
        dn2df[chunki]=pd.concat(dn2df_,axis=0)
    df1=pd.concat(dn2df,axis=0)
    df1.index.names=['chunk #','correlation name','correlation method']
    for c in ['x','y']:
        if f"{c}_covar" in df1:
            df1[f"{c}_covar"]=df1[f"{c}_covar"].apply(eval)
    return df1.reset_index()


def check_collinearity(df3,threshold=0.7):
    df4=df3.corr(method='spearman')
    # df4=df4.applymap(abs)
    from rohan.lib.io_dfs import get_offdiagonal_values
    df5=get_offdiagonal_values(df4.copy())
    df6=df5.melt(ignore_index=False).dropna().reset_index()
    df6['value']=df6['value'].apply(abs)
    df6['is collinear']=df6['value'].apply(lambda x: x>=threshold)
    perc=(df6['is collinear'].sum()/len(df6))*100
    if perc==0:
        logging.info(f"% collinear vars: {perc}")
        return
    logging.info(f"% collinear vars: {perc}")
    df6=df6.loc[(df6['is collinear']),:]
    from rohan.lib.stat.network import get_subgraphs
    df7=get_subgraphs(df6.loc[df6['is collinear'],:],'index','variable')
    df7=df7.groupby('subnetwork name').agg({'node name':list}).reset_index()
    return df7.groupby('subnetwork name').apply(lambda df: df6.apply(lambda x: x['value'] if len(set([x['index'],x['variable']]) - set(df['node name'].tolist()[0]))==0 else np.nan,axis=1).min()).sort_values(ascending=False)
