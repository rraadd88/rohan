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
def unique_dropna_str(l): return tuple2str(dropna(unique(l,drop='nan')))

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