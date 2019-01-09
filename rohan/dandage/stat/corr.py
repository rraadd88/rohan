import pandas as pd
import numpy as np

def df2corrmean(df):
    return df.corr(method='spearman').replace(1,np.nan).mean().mean()

from scipy.stats import spearmanr
def corrdfs(df1,df2):
    """
    df1 in columns
    df2 in rows    
    """
    dcorr=pd.DataFrame(columns=df1.columns,index=df2.columns)
    dpval=pd.DataFrame(columns=df1.columns,index=df2.columns)
    for c1 in df1:
        for c2 in df2:    
            dcorr.loc[c2,c1],dpval.loc[c2,c1]=spearmanr(df1[c1],df2[c2],
                                                        nan_policy='omit'
                                                       )
    if not df1.columns.name is None:
        dcorr.columns.name=df1.columns.name
        dpval.columns.name=df1.columns.name
    if not df2.columns.name is None:
        dcorr.index.name=df2.columns.name
        dpval.index.name=df2.columns.name
    return dcorr,dpval