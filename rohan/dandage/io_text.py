from rohan.global_imports import *
from rohan.dandage.stat.cluster import get_clusters
def corpus2clusters(corpus, index,params_clustermap={'vmin':0,'vmax':1,'figsize':[6,6]}):
    """
    corpus: list of strings
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vect = TfidfVectorizer()
    tfidf = vect.fit_transform(corpus)

    df=pd.DataFrame((tfidf * tfidf.T).A,
                index=index,
                columns=index,
                )
    clustergrid=sns.clustermap(df,**params_clustermap
    #                                method='complete', metric='canberra',
                                  )
    dclusters=get_clusters(clustergrid,axis=0,criterion='maxclust',clusters_fraction=0.25)
    return dclusters