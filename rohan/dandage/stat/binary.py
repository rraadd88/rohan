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
    #https://stackoverflow.com/a/40589850/3521099
    x = np.asarray(x, np.bool) # Not necessary, if you keep your data
    y = np.asarray(y, np.bool) # in a boolean array already!
    return np.double(np.bitwise_and(x, y).sum()) / np.double(np.bitwise_or(x, y).sum())

def compare_bools_jaccard_df(df):
    from rohan.dandage.stat.binary import compare_bools_jaccard
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

# set enrichment
# from rohan.dandage.stat.binary import compare_bools_jaccard
from scipy.stats import hypergeom,fisher_exact
def get_intersection_stats(df,coltest,colset):
    hypergeom_p=hypergeom.sf(sum(df[coltest] & df[colset])-1,len(df),df[colset].sum(),df[coltest].sum(),)
    _,fisher_exactp=fisher_exact(pd.crosstab(df[coltest], df[colset]),alternative='two-sided')
    jaccard=compare_bools_jaccard(df[coltest],df[colset])
    return hypergeom_p,fisher_exactp,jaccard
def get_set_enrichment_stats(test,background,sets):
    delement=pd.DataFrame(index=background)
    delement.loc[test,'test']=True
    for k in sets:
        delement.loc[sets[k],k]=True
    delement=delement.fillna(False)
    dmetric=pd.DataFrame({colset:get_intersection_stats(delement,'test',colset) for colset in delement if colset!='test'}).T.rename(columns=dict(zip([0,1,2],['hypergeom p-val','fisher_exact p-val','jaccard index'])))
    from statsmodels.stats.multitest import multipletests
    for c in dmetric:
        if c.endswith(' p-val'):
            dmetric[f"{c} corrected"]=multipletests(dmetric[c], alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)[1]
    return dmetric
