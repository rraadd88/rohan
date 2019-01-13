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

def unique_dropna(l): return unique(l,drop='nan')
def merge_unique_dropna(l): return unique(list(itertools.chain(*l)),drop='nan')

def tuple2str(tup,sep=' '): 
    if isinstance(tup,tuple):
        tup=[str(s) for s in tup]
        tup=sep.join(list(tup))
    return tup



def rankwithlist(l,lwith,test=False):
    if not (isinstance(l,list) and isinstance(lwith,list)):
        l,lwith=list(l),list(lwith)
    from scipy.stats import rankdata
    if test:
        print(l,lwith)
        print(rankdata(l),rankdata(lwith))
        print(rankdata(l+lwith))
    return rankdata(l+lwith)[:len(l)]