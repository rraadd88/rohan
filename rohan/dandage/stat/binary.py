from rohan.global_imports import *

def log_likelihood(y_true, y_pred):
    """
    source: https://github.com/saezlab/protein_attenuation/blob/6c1e81af37d72ef09835ee287f63b000c7c6663c/src/protein_attenuation/utils.py
    """
    n = len(y_true)
    ssr = np.power(y_true - y_pred, 2).sum()
    var = ssr / n

    l = np.longfloat(1 / (np.sqrt(2 * np.pi * var))) ** n * np.exp(-(np.power(y_true - y_pred, 2) / (2 * var)).sum())
    ln_l = np.log(l)

    return ln_l


def f_statistic(y_true, y_pred, n, p):
    """
    source: https://github.com/saezlab/protein_attenuation/blob/6c1e81af37d72ef09835ee287f63b000c7c6663c/src/protein_attenuation/utils.py
    """
    msm = np.power(y_pred - y_true.mean(), 2).sum() / p
    mse = np.power(y_true - y_pred, 2).sum() / (n - p - 1)

    f = msm / mse

    f_pval = stats.f.sf(f, p, n - p - 1)

    return f, f_pval

def compare_bools_auc(true,test,outmore=False):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(true,test)
    a=auc(fpr, tpr)
    if not outmore:
        return a
    else:
        return fpr, tpr, thresholds,a
    
def compare_bools_jaccard(x,y):
    x = np.asarray(x, np.bool) # Not necessary, if you keep your data
    y = np.asarray(y, np.bool) # in a boolean array already!
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())        