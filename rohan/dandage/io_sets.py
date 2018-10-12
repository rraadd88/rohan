from functools import reduce
import numpy as np

def list2intersection(l):
    return reduce(np.intersect1d, (l))