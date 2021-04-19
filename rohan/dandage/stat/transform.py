import pandas as pd
import numpy as np
import scipy as sc
import logging

def log(df,col,base=10,pcount=0):
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

## TODO deprecated
dflogcol=log
from rohan.dandage.stat.norm import zscore_cols as df2zscore

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

def rescale(a, range1=None, range2=[0,1]):
    if range1 is None:
        range1=[np.min(a),np.max(a)]
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (a - range1[0]) / delta1) + range2[0]

def get_bins(ds,bins,value='mid',ignore=False):
    if not ignore: logging.warning('assuming 1:1 mapping between index and values')
    i=len(ds.unique())/len(ds)
    if i!=1 and not ignore:
        logging.warning(f'fraction of unique values={i}')
    if i<0.1 and not ignore:
        logging.error(f'many duplicated values.')
        return 
    return pd.cut(ds,bins,duplicates='drop').apply(lambda x: getattr(x,value)).astype(float).to_dict()

def get_qbins(ds,bins,value='mid',error='raise'):
    logging.warning('assuming 1:1 mapping between index and values')
    i=len(ds.unique())/len(ds)
    if i!=1:
        logging.warning(f'fraction of unique values={i}')
    if i<0.1:
        logging.error(f'use get_bins instead. many duplicated values.')
        if error=='raise':
            return 
    return pd.qcut(ds,bins,duplicates='drop').apply(lambda x: getattr(x,value)).astype(float).to_dict()
