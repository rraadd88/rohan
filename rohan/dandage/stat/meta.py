from rohan.global_import import *

def compare_bools_scores(df0,colbool,colscore,colpivotindex,colpivotcolumns):
    from rohan.dandage.stat.binary import compare_bools_jaccard_df
    df1=pd.concat({'correlation':dmap2lin(df0.pivot_table(columns=colpivotcolumns,index=colpivotindex,values=colscore).corr(method='pearson')),
           'jaccard index':dmap2lin(compare_bools_jaccard_df(df0.pivot_table(columns=colpivotcolumns,index=colpivotindex,values='interaction bool')))},axis=0,names=['metric type']).reset_index()
    # .rename(columns={'index':'row',:'col'})
    df1=df1.loc[(df1['index']!=df1['column']),:]
    metric2index={'correlation':list(itertools.combinations(df1['index'].unique(),2)),}
    metric2index['jaccard index']=[t[::-1] for t in metric2index['correlation']]
    df2=pd.concat([df1.loc[(df1['metric type']==metric),:].set_index(['index','column']).loc[metric2index[metric],:] for metric in metric2index],
             axis=0).reset_index()
    df2['value']=df2['value'].apply(float)
#     print(df2.head())
    dmap=df2.pivot_table(index='index',columns='column',values='value').fillna(1)
    dmap.columns.name='$r_p$'
    dmap.index.name='Jaccard index'
    return dmap