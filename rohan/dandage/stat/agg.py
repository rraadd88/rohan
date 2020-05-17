import numpy as np
def diff(a,b): return abs(a-b)
def balance(a,b):
    sum_=a+b
    if sum_!=0:
        return 1-(abs(a-b)/(sum_))
    else:
        return np.nan
def aggs_by_pair(a,b): return np.mean([a,b]),diff(a,b),balance(a,b)