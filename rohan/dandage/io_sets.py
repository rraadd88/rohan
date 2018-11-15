## Also includes list and tuple things

from functools import reduce
import numpy as np

def list2intersection(l):
    return reduce(np.intersect1d, (l))
def list2union(l):
    return np.unique(np.ravel(l))

def unique(l):
    l=[str(s) for s in l]
    return tuple(np.unique(l))
def tuple2str(tup,sep=' '): 
    if isinstance(tup,tuple):
        tup=[str(s) for s in tup]
        tup=sep.join(list(tup))
    return tup
