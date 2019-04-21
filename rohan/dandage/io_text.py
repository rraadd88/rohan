from rohan.dandage.stat.cluster import get_clusters
def corpus2clusters(corpus, index):
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
    clustergrid=sns.clustermap(df,
    #                                method='complete', metric='canberra',
                               vmin=-0.15,vmax=0.15,
                               cmap='coolwarm'
                                  )
    dclusters=get_clusters(clustergrid,axis=0,criterion='maxclust',clusters_fraction=0.25)
    return dclusters