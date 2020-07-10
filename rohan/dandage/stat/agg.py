import numpy as np
from rohan.dandage.stat.diff import diff,balance 
def aggs_by_pair(a,b,absolute=True): return np.mean([a,b]),diff(a,b,absolute=absolute),balance(a,b)