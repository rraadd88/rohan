## Also includes list, np vectors and tuple things

from functools import reduce
import numpy as np
import pandas as pd

def list2intersection(l):
    return reduce(np.intersect1d, (l))
def list2union(l):
    return np.unique(np.ravel(l))

# lists mostly for agg
def dropna(x):
    x_=[]
    for i in x:
        if not pd.isnull(i):
            x_.append(i)
    return x_

def unique(l,drop=None):
    l=[str(s) for s in l]
    if drop is not None:
        l=[s for s in l if s!=drop]
    return tuple(np.unique(l))

def tuple2str(tup,sep='; '): 
    if not isinstance(tup,list):
        tup=tuple(tup)
    tup=[str(s) for s in tup]
    tup=sep.join(list(tup))
    return tup

def unique_dropna(l): return dropna(unique(l,drop='nan'))
def unique_dropna_str(l,sep='; '): return tuple2str(dropna(unique(l,drop='nan')),sep=sep)

def merge_unique_dropna(l): return dropna(unique(list(itertools.chain(*l)),drop='nan'))

def rankwithlist(l,lwith,test=False):
    """
    rank l wrt lwith
    """
    if not (isinstance(l,list) and isinstance(lwith,list)):
        l,lwith=list(l),list(lwith)
    from scipy.stats import rankdata
    if test:
        print(l,lwith)
        print(rankdata(l),rankdata(lwith))
        print(rankdata(l+lwith))
    return rankdata(l+lwith)[:len(l)]

# getting sections from boolian vector
def bools2intervals(v):
    return np.flatnonzero(np.diff(np.r_[0,v,0])!=0).reshape(-1,2) - [0,1]
def dfbool2intervals(df,colbool):
    """
    ds contains bool values
    """
    df.index=range(len(df))
    intervals=bools2intervals(df[colbool])
    for intervali,interval in enumerate(intervals):
        df.loc[interval[0]:interval[1],f'{colbool} interval id']=intervali
        df.loc[interval[0]:interval[1],f'{colbool} interval start']=interval[0]
        df.loc[interval[0]:interval[1],f'{colbool} interval stop']=interval[1]
        df.loc[interval[0]:interval[1],f'{colbool} interval length']=interval[1]-interval[0]+1
        df.loc[interval[0]:interval[1],f'{colbool} interval index']=range(interval[1]-interval[0]+1)    
    return df