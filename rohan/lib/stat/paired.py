# import pandas as pd
from rohan.global_imports import *

def get_stats_paired(df1,cols,input_logscale,drop_cols=False,
                     take_max=True,
                    fast=False):
    assert(len(cols)==2)
    s1=get_fix(*cols,common=True,clean=True)
    info(s1)
    from rohan.lib.stat.diff import get_ratio_sorted,get_diff_sorted
    df1[f"{s1} {'ratio' if not input_logscale else 'diff'}"]=getattr(df1,'parallel_apply' if fast else "apply")(lambda x: (get_ratio_sorted if not input_logscale else get_diff_sorted)(x[cols[0]],
                                                            x[cols[1]]),
                                                            axis=1)
    assert(not any(df1[f"{s1} {'ratio' if not input_logscale else 'diff'}"]<0))
    if take_max:
        df1[f'{s1} max']=df1.loc[:,cols].max(axis=1)
    if drop_cols:
        df1=df1.drop(cols,axis=1)
    return df1