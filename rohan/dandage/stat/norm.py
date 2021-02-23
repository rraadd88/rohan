import pandas as pd
import numpy as np
from scipy import stats

def norm_by_quantile(X):
    """Normalize the columns of X to each have the same distribution.

    Given an expression matrix (microarray data, read counts, etc) of M genes
    by N samples, quantile normalization ensures all samples have the same
    spread of data (by construction).

    The data across each row are averaged to obtain an average column. Each
    column quantile is replaced with the corresponding quantile of the average
    column.

    Parameters
    ----------
    X : 2D array of float, shape (M, N)
        The input data, with M rows (genes/features) and N columns (samples).

    Returns
    -------
    Xn : 2D array of float, shape (M, N)
        The normalized data.
    """
    # compute the quantiles
    quantiles = np.mean(np.sort(X, axis=0), axis=1)

    # compute the column-wise ranks. Each observation is replaced with its
    # rank in that column: the smallest observation is replaced by 1, the
    # second-smallest by 2, ..., and the largest by M, the number of rows.
    ranks = np.apply_along_axis(stats.rankdata, 0, X)

    # convert ranks to integer indices from 0 to M-1
    rank_indices = ranks.astype(int) - 1

    # index the quantiles for each rank with the ranks matrix
    Xn = quantiles[rank_indices]
    
    if isinstance(X,pd.DataFrame):
        return pd.DataFrame(Xn,columns=X.columns,index=X.index)
    else:
        return(Xn)

def quantile_norm(X):
    """
    alias
    """
    return norm_by_quantile(X)

def norm_by_gaussian_kde(values):
    """
    source: https://github.com/saezlab/protein_attenuation/blob/6c1e81af37d72ef09835ee287f63b000c7c6663c/src/protein_attenuation/utils.py
    """    
    kernel = stats.gaussian_kde(values)
    return Series({k: np.log(kernel.integrate_box_1d(-1e4, v) / kernel.integrate_box_1d(v, 1e4)) for k, v in values.to_dict().items()})

from rohan.dandage.stat.transform import rescale

## z-scores
def zscore(ds): 
    return (ds - ds.mean())/ds.std()
def zscore_cols(df,cols=None): 
    if cols is None:
        cols=df.columns
    return (df[cols] - df[cols].mean())/df[cols].std()
def get_zscore_robust(x,median,mad):
    return (x-median)/(mad*1.4826)
def zscore_robust(a):
    """
    Example:
    t = sc.stats.norm.rvs(size=100, scale=1, random_state=123456)
    plt.hist(t,bins=40)
    plt.hist(apply_zscore_robust(t),bins=40)
    print(np.median(t),np.median(apply_zscore_robust(t)))
    """
    median=np.median(a)
    mad=sc.stats.median_abs_deviation(a)
    if mad==0:
        logging.error('mad==0')
    return [get_zscore_robust(x,median,mad) for x in a]




