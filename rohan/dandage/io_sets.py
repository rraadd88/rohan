## Also includes list, np vectors and tuple things

from functools import reduce
import numpy as np

def list2intersection(l):
    return reduce(np.intersect1d, (l))
def list2union(l):
    return np.unique(np.ravel(l))

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

def dropna(x): return x[~pd.isnull(x)]
