import pandas as pd
import numpy as np
def dflogcol(df,col):
    df[f"{col} (log scale)"]=df[col].apply(np.log10).replace([np.inf, -np.inf], np.nan)
    return df