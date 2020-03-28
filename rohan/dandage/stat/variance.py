import numpy as np
def confidence_interval_95(x):return 1.96*np.std(x)/np.sqrt(len(x))

def balance(a,b):
    sum_=a+b
    if sum_!=0:
        return 1-(abs(a-b)/(sum_))
    else:
        return np.nan