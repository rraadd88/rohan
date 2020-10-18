import pandas as pd
import numpy as np
import scipy as sc
import logging

def dflogcol(df,col,base=10,pcount=0):
    if base==10:
        log =np.log10 
    elif base==2:
        log =np.log2
    elif base=='e':
        log =np.log        
    else:
        ValueError(f"unseen base {base}")
    df[f"{col} (log{base} scale)"]=df[col].apply(lambda x : log(x+pcount)).replace([np.inf, -np.inf], np.nan)
    return df

def df2zscore(df,cols=None): 
    if cols is None:
        cols=df.columns
    return (df[cols] - df[cols].mean())/df[cols].std()

def get_zscore_robust(x,median,mad):
    return (x-median)/(mad*1.4826)
def apply_zscore_robust(a):
    """
    Example:
    t = sc.stats.norm.rvs(size=100, scale=1, random_state=123456)
    plt.hist(t,bins=40)
    plt.hist(apply_zscore_robust(t),bins=40)
    print(np.median(t),np.median(apply_zscore_robust(t)))
    """
    median=np.median(a)
    mad=sc.stats.median_abs_deviation(a)
    if mad==0:
        logging.error('mad==0')
    return [get_zscore_robust(x,median,mad) for x in a]

def plog(x,p = 0.5,base=None):
    """
    psudo-log

    :param x: number
    :param p: number added befor logarithm 
    """
    if base is None:
        logging.warning(f"base is {base}")
        return np.log(x+p)
    else:
        return np.log2(x+p)/np.log(base)

def glog(x,l = 2):
    """
    Generalised logarithm

    :param x: number
    :param p: number added befor logarithm 

    """
    return np.log((x+np.sqrt(x**2+l**2))/2)/np.log(l)

def rescale(a, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (a - range1[0]) / delta1) + range2[0]

def get_bins(ds,bins,value='mid'):
    logging.warning('assuming 1:1 mapping between index and values')
    i=len(ds.unique())/len(ds)
    if i!=1:
        logging.warning(f'fraction of unique values={i}')
    if i<0.1:
        logging.error(f'many duplicated values.')
        return 
    return pd.cut(ds,bins,duplicates='drop').apply(lambda x: getattr(x,value)).astype(float).to_dict()

def get_qbins(ds,bins,value='mid'):
    logging.warning('assuming 1:1 mapping between index and values')
    i=len(ds.unique())/len(ds)
    if i!=1:
        logging.warning(f'fraction of unique values={i}')
    if i<0.1:
        logging.error(f'use get_bins instead. many duplicated values.')
        return 
    return pd.qcut(ds,bins,duplicates='drop').apply(lambda x: getattr(x,value)).astype(float).to_dict()

def aggcol_by_qbins(df,colx,coly,colgroupby=None,bins=10):
    df[f"{colx} qbin"]=pd.qcut(df[colx],bins,duplicates='drop')    
    if colgroupby is None:
        colgroupby='del'
        df[colgroupby]='del'
    from rohan.dandage.stat.variance import confidence_interval_95
    dplot=df.groupby([f"{colx} qbin",colgroupby]).agg({coly:[np.mean,confidence_interval_95],})
    from rohan.dandage.io_dfs import coltuples2str
    dplot.columns=coltuples2str(dplot.columns)
    dplot=dplot.reset_index()
    dplot[f"{colx} qbin midpoint"]=dplot[f"{colx} qbin"].apply(lambda x:x.mid).astype(float)
    dplot[f"{colx} qbin midpoint"]=dplot[f"{colx} qbin midpoint"].apply(float)
    if 'del' in dplot:
        dplot=dplot.drop(['del'],axis=1)
    return dplot