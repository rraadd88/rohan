import pandas as pd
import numpy as np
def dflogcol(df,col,base=10,pcount=0):
    if base==10:
        log =np.log10 
    elif base==2:
        log =np.log2
    elif base=='e':
        log =np.log        
    else:
        ValueError(f"unseen base {base}")
    df[f"{col} (log{base} scale)"]=df[col].apply(lambda x : log(x+pcount)).replace([np.inf, -np.inf], np.nan)
    return df

def df2zscore(df,cols=None): 
    if cols is None:
        cols=df.columns
    return (df[cols] - df[cols].mean())/df[cols].std()