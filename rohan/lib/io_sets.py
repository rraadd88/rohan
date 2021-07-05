## Also includes list, np vectors and tuple things
import itertools
import numpy as np
import pandas as pd
import logging

from functools import reduce
def union(l):return reduce(np.union1d, (l))
def intersection(l):return reduce(np.intersect1d, (l))

def list2union(l): return union(l)
def list2intersection(l): return intersection(l)    

def nunion(l): return len(union(l))
def nintersection(l): return len(intersection(l))

# lists mostly for agg
def dropna(x):
    x_=[]
    for i in x:
        if not pd.isnull(i):
            x_.append(i)
    return x_

def unique(l): return list(np.unique(l))
def unique_str(l1): 
    l2=unique(l1)
    assert(len(l2)==1)
    return l2[0]
def nunique(l,**kws): return len(unique(l,**kws))
def unique_dropna(l): return dropna(unique(l,drop='nan'))
def unique_dropna_str(l,sep='; '): return tuple2str(dropna(unique(l,drop='nan')),sep=sep)
def merge_unique_dropna(l): return dropna(unique(list(itertools.chain(*l)),drop='nan'))


def list_value_counts(l):return dict(zip(*np.unique(l, return_counts=True)))
def tuple2str(tup,sep='; '): 
    if not isinstance(tup,list):
        tup=tuple(tup)
    tup=[str(s) for s in tup]
    tup=sep.join(list(tup))
    return tup
def list2str(x):
    x=list(x)
    if len(x)>1:
        logging.warning('more than 1 str value encountered, returning list')
        return x
    else:
        return x[0]

def flatten(l):
    return list(np.hstack(np.array(l)))

def get_alt(l1,s,): 
    assert(s in l1)
    return [i for i in l1 if i!=s][0]

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
        df.loc[interval[0]:interval[1],f'{colbool} interval within index']=range(interval[1]-interval[0]+1)    
    df[f'{colbool} interval index']=df.index    
    return df

def intersections(dn2list,jaccard=False,count=True,fast=False,test=False):
    """
    TODO: feed as an estimator to df.corr()
    TODO: way to fill up the symetric half of the adjacency matrix
    """
    df=pd.DataFrame(index=dn2list.keys(),
                columns=dn2list.keys())
    if jaccard:
        dn2list={k:set(dn2list[k]) for k in dn2list}
    from tqdm import tqdm
    for k1i,k1 in tqdm(enumerate(dn2list.keys())):
#         if test:
#             print(f"{(k1i/len(dn2list.keys()))*100:.02f}")
        for k2i,k2 in enumerate(dn2list.keys()):
            if fast and k1i>=k2i:
                continue
            if jaccard:
                if len(dn2list[k1].union(dn2list[k2]))!=0:
                    l=len(set(dn2list[k1]).intersection(dn2list[k2]))/len(dn2list[k1].union(dn2list[k2]))
                else:
                    l=np.nan
            else:
                l=list(set(dn2list[k1]).intersection(dn2list[k2]))
            if count:
                df.loc[k1,k2]=len(l)
            else:
                df.loc[k1,k2]=l
    return df

def jaccard_index_dict(dn2list,jaccard=True,count=False,fast=False,test=False):
    return intersections(dn2list,
                         jaccard=jaccard,count=count,
                         fast=fast,test=test)

compare_lists_jaccard=intersections

## stats
def jaccard_index_df(df):
    from rohan.lib.io_sets import jaccard_index_dict
    return jaccard_index_dict(df.apply(lambda x: dropna(x)).to_dict())

def jaccard_index(l1,l2):
    l1,l2=dropna(l1),dropna(l2)
    i=len(set(l1).intersection(l2))
    u=len(set(l1).union(l2))
    return i/u,i,u

def difference(l1,l2): return list(set(l1).difference(l2))

def group_list_bylen(l,length): return list(zip(*(iter(l),) * length))
def sort_list_by_list(l,byl): return [x for x,_ in sorted(zip(l,byl))]

## ranges
def range_overlap(l1,l2):
    return list(set.intersection(set(range(l1[0],l1[1]+1,1)),
                            set(range(l2[0],l2[1]+1,1))))