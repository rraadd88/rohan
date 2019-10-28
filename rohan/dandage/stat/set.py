from rohan.global_imports import *

# set enrichment
from rohan.dandage.stat.binary import compare_bools_jaccard
from scipy.stats import hypergeom,fisher_exact
def get_intersection_stats(df,coltest,colset,background_size=None):
    """
    :param background: size of the union (int)
    """
    hypergeom_p=hypergeom.sf(sum(df[coltest] & df[colset])-1,
                             len(df) if background_size is None else background_size,
                             df[colset].sum(),
                             df[coltest].sum(),)
    _,fisher_exactp=fisher_exact(pd.crosstab(df[coltest], df[colset]),alternative='two-sided')
    jaccard=compare_bools_jaccard(df[coltest],df[colset])
    return hypergeom_p,fisher_exactp,jaccard

def get_set_enrichment_stats(test,sets,background,fdr_correct=True):
    """
    test:
        get_set_enrichment_stats(background=range(120),
                        test=range(100),
                        sets={f"set {i}":list(np.unique(np.random.randint(low=100,size=i+1))) for i in range(100)})
        # background is int
        get_set_enrichment_stats(background=110,
                        test=unique(range(100)),
                        sets={f"set {i}":unique(np.random.randint(low=140,size=i+1)) for i in range(0,140,10)})                        
    """
    if isinstance(background,list):
        background_elements=np.unique(background)
        background_size=None
    elif isinstance(background,(int,float)):
        background_elements=list2union([test]+list(sets.values()))
        background_size=background
        if len(background_elements)>background_size:
            logging.error(f"invalid data type of background {type(background)}")
    else:
        logging.error(f"invalid data type of background {type(background)}")
    delement=pd.DataFrame(index=background_elements)
    delement.loc[np.unique(test),'test']=True
    for k in sets:
        delement.loc[np.unique(sets[k]),k]=True
    delement=delement.fillna(False)
    dmetric=pd.DataFrame({colset:get_intersection_stats(delement,'test',colset,background_size=background_size) for colset in delement if colset!='test'}).T.rename(columns=dict(zip([0,1,2],['hypergeom p-val','fisher_exact p-val','jaccard index'])))
    if fdr_correct:
        from statsmodels.stats.multitest import multipletests
        for c in dmetric:
            if c.endswith(' p-val'):
                dmetric[f"{c} corrected"]=multipletests(dmetric[c], alpha=0.05, method='fdr_bh', is_sorted=False,returnsorted=False)[1]
    return dmetric
