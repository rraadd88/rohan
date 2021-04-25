"""
io_df -> io_dfs -> io_files
"""
import pandas as pd
import numpy as np

import logging
from icecream import ic

## log
def log_apply(df, fun, *args, **kwargs):
    """
    """
    d1={}
    d1['from']=df.shape
    df = getattr(df, fun)(*args, **kwargs)
    d1['to  ']=df.shape
    if d1['from']!=d1['to  ']:
        for k in d1:
            logging.info(f'{fun}: shape changed {k} {d1[k]}')
    return df
