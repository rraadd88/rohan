import numpy as np
def confidence_interval_95(x):return 1.96*np.std(x)/np.sqrt(len(x))

def get_ci(rs,ci_type):
    if ci_type.lower()=='max':
        return max([abs(r-np.mean(rs)) for r in rs])
    elif ci_type.lower()=='sd':
        return np.std(rs)
    elif ci_type.lower()=='ci':
        return confidence_interval_95(rs)
    else:
        raise ValueError("ci_type invalid")