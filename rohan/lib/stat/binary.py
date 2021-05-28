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

def get_roc_auc(true,test,outmore=False):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(true,test)
    a=auc(fpr, tpr)
    if not outmore:
        return a
    else:
        return fpr, tpr, thresholds,a
    
def compare_bools_jaccard(x,y):
    #https://stackoverflow.com/a/40589850/3521099
    x = np.asarray(x, np.bool) # Not necessary, if you keep your data
    y = np.asarray(y, np.bool) # in a boolean array already!
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

def compare_bools_jaccard_df(df):
    from rohan.lib.stat.binary import compare_bools_jaccard
    dmetrics=pd.DataFrame(index=df.columns.tolist(),columns=df.columns.tolist())
    for c1i,c1 in enumerate(df.columns):
        for c2i,c2 in enumerate(df.columns):
            if c1i>c2i:
                dmetrics.loc[c1,c2]=compare_bools_jaccard(df.dropna(subset=[c1,c2])[c1],df.dropna(subset=[c1,c2])[c2])
            elif c1i==c2i:
                dmetrics.loc[c1,c2]=1
    for c1i,c1 in enumerate(df.columns):
        for c2i,c2 in enumerate(df.columns):
            if c1i<c2i:
                dmetrics.loc[c1,c2]=dmetrics.loc[c2,c1]
    return dmetrics

def classify_bools(l): return 'both' if all(l) else 'either' if any(l) else 'neither'

## agg
def perc(x): return (sum(x)/len(x))*100