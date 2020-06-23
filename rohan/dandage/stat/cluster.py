from rohan.dandage.plot.heatmap import get_clusters
from rohan.dandage.io_dfs import *
# dynamic time wrapping
def get_ddist_dtw(df,window):
    from dtaidistance import dtw
    return dmap2lin(make_symmetric_across_diagonal(pd.DataFrame(dtw.distance_matrix_fast(df.applymap(np.double).values,
                                                                                  window=window,),
                    index=df.index,
                    columns=df.index,
                    ).replace(np.inf,np.nan)),colvalue_name='distance').set_index(['index','column'])
    
# compare    
def get_ddist(df,window_size_max=10,corr=False):
    print(df.shape)
    print(f"window=",end='')
    method2ddists={}
    for window in range(1,window_size_max,1):
        print(window,end=' ')
        method2ddists[f'DTW (window={window:02d})']=get_ddist_dtw(df,window)
    if corr:
        method2ddists['1-spearman']=dmap2lin((1-df.T.corr(method='spearman')),colvalue_name='distance').set_index(['index','column'])
        method2ddists['1-pearson']=dmap2lin((1-df.T.corr(method='pearson')),colvalue_name='distance').set_index(['index','column'])

    ddist=pd.concat(method2ddists,axis=1,)
    ddist.columns=coltuples2str(ddist.columns)
    ddist=ddist.reset_index()
    ddist=ddist.loc[(ddist['index']!=ddist['column']),:]
    ddist['interaction id']=ddist.apply(lambda x : '--'.join(list(sorted([x['index'],x['column']]))),axis=1)
    print(ddist.shape,end='')
    ddist=ddist.drop_duplicates(subset=['interaction id'])
    print(ddist.shape)   
    return ddist

# scikit learn below        
def get_clusters(X,n_clusters,test=False):
    from sklearn import cluster,metrics
    kmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=88,
                            ).fit(X)
    clusters=kmeans.predict(X)
    ds=pd.Series(dict(zip(X.index,clusters)))
    ds.name='cluster #'
    df=pd.DataFrame(ds)
    df.index.name=X.index.name
    df['cluster #'].value_counts()
    # Compute the silhouette scores for each sample
    df['silhouette value'] = metrics.silhouette_samples(X, clusters)
    is_optimum=(df.groupby(['cluster #']).agg({'silhouette value':np.max})['silhouette value']>=df['silhouette value'].mean()).all()
    if test:
        print(f"{n_clusters} cluster : silhouette average score {df['silhouette value'].mean():1.2f}, ok?: {is_optimum}")
    if not is_optimum:
        logging.warning(f"{n_clusters} cluster : silhouette average score {df['silhouette value'].mean():1.2f}, ok?: {is_optimum}")
    return df
def get_n_clusters_optimum(df3):
    df=df3.groupby('clusters total').agg({'silhouette value':np.mean}).reset_index().sort_values(by=['silhouette value','clusters total',],ascending=[False,False])
    df.index=range(len(df))
    for i,n in enumerate(df['clusters total'].diff()):
        if n < 0:
            return df.iloc[i-1,:].to_dict()['clusters total']
def plot_n_clusters_optimization(df,n_clusters_optimum,ax=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax=plt.subplot() if ax is None else ax
    ax=sns.swarmplot(data=df.groupby(['clusters total','cluster #']).agg({'silhouette value':np.max}).reset_index(),
                     y='silhouette value',x='clusters total',
                     color='r',alpha=0.7,
                     ax=ax)
    ax=sns.pointplot(data=df.groupby('clusters total').agg({'silhouette value':np.mean}).reset_index(),
                     y='silhouette value',x='clusters total',
                     color='k',
                     ax=ax)
    ax.annotate('optimum', xy=(n_clusters_optimum-int(ax.get_xticklabels()[0].get_text()), 0.2),  xycoords='data',
                    xytext=(+50, +50), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",ec='k',
                                    connectionstyle="angle3,angleA=0,angleB=-90"),
                    )
    ax.set_xlabel('clusters')  
    return ax
def get_clusters_optimum(X,n_clusters_range_min=2,n_clusters_range_max=10,
                         test=False,
                         out_optimized_only=False,
                        ):
    """
    :params X: samples to cluster in indexed 
    """
    dn2df={}
    for n_clusters in list(range(n_clusters_range_min,n_clusters_range_max+1)):
        dn2df[n_clusters]=get_clusters(X,n_clusters,test=test)
    df1=pd.concat(dn2df,axis=0,names=['clusters total']).reset_index()
    n_clusters_optimum=get_n_clusters_optimum(df1) 
    if test or plot:
        plot_n_clusters_optimization(df=df1,n_clusters_optimum=n_clusters_optimum)
    if not out_optimized_only:
        return df1,n_clusters_optimum
    else:
        return dn2df[n_clusters_optimum]