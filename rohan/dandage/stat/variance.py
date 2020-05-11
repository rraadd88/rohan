import numpy as np
def confidence_interval_95(x):return 1.96*np.std(x)/np.sqrt(len(x))