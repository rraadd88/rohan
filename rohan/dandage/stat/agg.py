import numpy as np
def diff(a,b,absolute=True): 
    diff=a-b
    if absolute:
        return abs(diff)
    else:
        return diff
def balance(a,b,absolute=True):
    sum_=a+b
    if sum_!=0:
        return 1-(diff(a,b,absolute=absolute)/(sum_))
    else:
        return np.nan
def aggs_by_pair(a,b,absolute=True): return np.mean([a,b]),diff(a,b,absolute=absolute),balance(a,b)