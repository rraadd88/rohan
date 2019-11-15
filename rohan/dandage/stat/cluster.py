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

