import pandas as pd
import numpy as np

def df2corrmean(df):
    return df.corr().replace(1,np.nan).mean().mean()