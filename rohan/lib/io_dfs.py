"""
io_df -> io_dfs -> io_files

dtypes
'b'       boolean
'i'       (signed) integer
'u'       unsigned integer
'f'       floating-point
'c'       complex-floating point
'O'       (Python) objects
'S', 'a'  (byte-)string
'U'       Unicode
'V'       raw data (void)
"""
from rohan.dandage.io_df import *
from rohan.dandage import add_method_to_class
from rohan.global_imports import rd

def filter_dfs(dfs,cols,how='inner'):
    """
    
    """
    def apply_(dfs,col,how):
        from rohan.dandage.io_sets import list2intersection,list2union
        if how=='inner':
            l=list(list2intersection([df[col].tolist() for df in dfs]))
        elif how=='outer':
            l=list(list2union([df[col].tolist() for df in dfs]))
        else:
            raise ValueError("")
        logging.info(f"len({col})={len(l)}")
        return [df.loc[(df[col].isin(l)),:] for df in dfs]
    if isinstance(cols,str):
        cols=[cols]
    # sort columns by nunique
    cols=dfs[0].loc[:,cols].nunique().sort_values().index.tolist()
    for c in cols:
        dfs=apply_(dfs=dfs,col=c,how=how)
    return dfs

@add_method_to_class(rd)
def merge_paired(dfpair,df,
                        left_ons=['gene1 id','gene2 id'],
                        right_on='gene id',right_ons_common=[],
                        suffixes=[' gene1',' gene2'],how='left',
                                    dryrun=False, 
                                    test=False,
                                    verb=True,
                                   **kws_merge):
    """
    :param right_ons_common: columns to merge the right ones. eg. cell line
    """
#     left_ons=[[s] isinstance(s,str) else s for s in left_ons]
#     right_on=[right_on] isinstance(right_on,str) else right_on
#     right_ons_common=[right_ons_common] isinstance(right_ons_common,str) else right_ons_common
    
#     # force suffixes
#     df1=df.copy()
#     df1.columns=df1.columns+suffixes[0]
#     df2=df.copy()
#     df2.columns=df2.columns+suffixes[1]

#     merge2side2cols={1:{'left':[]}}
#     merge2side2cols[1]['left']=left_ons[0]+right_ons_common
#     merge2side2cols[1]['right']=right_on+right_ons_common

    d1={}
    d1['from']=dfpair.shape
    # force suffixes                        
    df1=df.copy()
    df1.columns=df1.columns+suffixes[0]
    df2=df.copy()
    df2.columns=df2.columns+suffixes[1]
    
    merge1_left_on=left_ons[0]
    merge1_right_on=(f'{right_on}{suffixes[0]}' if isinstance(right_on,str) else [f'{c}{suffixes[0]}' for c in right_on])
    merge2_left_on=left_ons[1] if len(right_ons_common)==0 else [left_ons[1]]+[f'{c}{suffixes[0]}' for c in right_ons_common]
    merge2_right_on=(f'{right_on}{suffixes[1]}' if isinstance(right_on,str) else [f'{c}{suffixes[1]}' for c in right_on]) if len(right_ons_common)==0 else [f'{right_on}{suffixes[1]}']+[f'{c}{suffixes[1]}' for c in right_ons_common]
    
    if dryrun or test:
#         from pprint import pprint
        logging.info('> merge1_left_on');logging.info(merge1_left_on)
        logging.info('> merge1_right_on');logging.info(merge1_right_on)
        logging.info('> merge2_left_on');logging.info(merge2_left_on)
        logging.info('> merge2_right_on');logging.info(merge2_right_on)
        logging.info('> dfpair');logging.info(dfpair.columns.tolist())
        logging.info('> df1');logging.info(df1.columns.tolist())
        logging.info('> df2');logging.info(df2.columns.tolist())
        return    
    
    dfpair_merge1=dfpair.merge(df1,
                    left_on=merge1_left_on,
                    right_on=merge1_right_on,
                    how=how,**kws_merge)
    if test:
        logging.info('> dfpair_merge1 columns'); logging.info(dfpair_merge1.columns.tolist())
    dfpair_merge2=dfpair_merge1.merge(df2,
                left_on=merge2_left_on,
                right_on=merge2_right_on,
                how=how,
                 **kws_merge)
    if test:
        logging.info('> dfpair_merge2 columns');logging.info(dfpair_merge2.columns.tolist())

    cols_same_right_on=[f"{right_on}{s}" for s in suffixes]
    cols_same_right_ons_common=[f"{c}{s}" for c in right_ons_common for s in suffixes]
    cols_same=list(np.ravel(list(zip(left_ons,cols_same_right_on))))+cols_same_right_ons_common
    cols_del=cols_same_right_on+cols_same_right_ons_common[1::2]
    if all([all(dfpair_merge2[c1]==dfpair_merge2[c2]) for c1,c2 in np.reshape(cols_same,(int(len(cols_same)/2),2))]):
        if test:
            logging.info('merged correctly')
        dfpair_merge2=dfpair_merge2.drop(cols_del,axis=1)
        dfpair_merge2=dfpair_merge2.rename(columns=dict(zip(cols_same_right_ons_common[0::2],right_ons_common)),
                                          errors='raise')
        
    d1['to  ']=dfpair_merge2.shape        
    if verb and d1['from']!=d1['to  ']:
        for k in d1:
            logging.info(f'shape changed {k} {d1[k]}')
    return dfpair_merge2
### alias to be deprecated   
merge_dfpairwithdf=merge_paired
merge_dfs_paired_with_unpaireds=merge_paired

## symmetric dfs eg. submaps    
@add_method_to_class(rd)
def dfmap2symmcolidx(df,test=False):
    geneids=set(df.index).union(set(df.columns))
    df_symm=pd.DataFrame(columns=geneids,index=geneids)
    df_symm.loc[df.index,:]=df.loc[df.index,:]
    df_symm.loc[:,df.columns]=df.loc[:,df.columns]
    if test:
        logging.debug(df_symm.shape)
    return df_symm

@add_method_to_class(rd)
def fill_diagonal(df,filler=None):
    if df.shape[0]!=df.shape[1]:
        logging.warning('df not symmetric')      
#     ids=set(df.columns).intersection(df.index)
    if filler is None:
        filler=np.nan
    if str(df.dtypes.unique()[0])=='bool' and (pd.isnull(filler)) :
        logging.info('warning diagonal is replaced by True rather than nan')
    np.fill_diagonal(df.values, filler)        
    return df

@add_method_to_class(rd)
def get_diagonalvals(df):
    if df.shape[0]!=df.shape[1]:
        logging.warning('df not symmetric')
#     ids=set(df.columns).intersection(df.index)
    ds=pd.Series(np.diag(df), index=[df.index, df.columns])
#     id2val={}
#     for i,c in zip(ids,ids):
#         id2val[i]=df.loc[i,c]
    return pd.DataFrame(ds,columns=['diagonal value']).reset_index()

@add_method_to_class(rd)
def fill_symmetricdf_indices(dsubmap,vals=None):
    if vals is None:
        vals=np.unique(dsubmap.index.tolist()+dsubmap.columns.tolist())
    for v in vals: 
        if not v in dsubmap.columns:            
            dsubmap[v]=np.nan
        if not v in dsubmap.index:
            dsubmap.loc[v,:]=np.nan
    return dsubmap.loc[vals,vals]

@add_method_to_class(rd)
def fill_symmetricdf_across_diagonal(df,fill=None):
    df=fill_symmetricdf_indices(dsubmap=df,vals=None)
    for c1i,c1 in enumerate(df.columns):
        for c2i,c2 in enumerate(df.columns):
            if c1i>c2i:
                if fill is None:
                    bools=[pd.isnull(i) for i in [df.loc[c1,c2],df.loc[c2,c1]]]
                    if sum(bools)==1:
                        if bools[0]==True:
                            df.loc[c1,c2]=df.loc[c2,c1]
                        elif bools[1]==True:
                            df.loc[c2,c1]=df.loc[c1,c2]
                elif fill=='lower': 
                    df.loc[c1,c2]=df.loc[c2,c1]
                elif fill=='upper':
                    df.loc[c2,c1]=df.loc[c1,c2]                            
    return df

@add_method_to_class(rd)
def get_offdiagonal_values(dcorr,side='lower',take_diag=False,replace=np.nan):
    for ii,i in enumerate(dcorr.index):
        for ci,c in enumerate(dcorr.columns):            
            if side=='lower' and ci>ii:
                dcorr.loc[i,c]=replace
            elif side=='upper' and ci<ii:
                dcorr.loc[i,c]=replace
            if not take_diag:
                if ci==ii:
                    dcorr.loc[i,c]=replace
    return dcorr

## GROUPBY
# aggregate dataframes
def get_group(groups,i=None,verbose=True):
    if not i is None: 
        dn=list(groups.groups.keys())[i]
    else:
        dn=groups.size().sort_values(ascending=False).index.tolist()[0]
    logging.info(dn)
    df=groups.get_group(dn)
    df.name=dn
    return df
        
@add_method_to_class(rd)
def dropna_by_subset(df,colgroupby,colaggs,colval,colvar,test=False):
    df_agg=dfaggregate_unique(df,colgroupby,colaggs)
    df_agg['has values']=df_agg.apply(lambda x : len(x[f'{colval}: list'])!=0,axis=1)
    varswithvals=df_agg.loc[(df_agg['has values']),colvar].tolist()
    if test:
        df2info(df_agg)
    df=df.loc[df[colvar].isin(varswithvals),:] 
    return df

# multiindex
def coltuples2str(cols,sep=' '):
    from rohan.dandage.io_strs import tuple2str
    cols_str=[]
    for col in cols:
        cols_str.append(tuple2str(col,sep=sep))
    return cols_str

@add_method_to_class(rd)
def column_suffixes2multiindex(df,suffixes,test=False):
    cols=[c for c in df if c.endswith(f' {suffixes[0]}') or c.endswith(f' {suffixes[1]}')]
    if test:
        logging.info(cols)
    df=df.loc[:,cols]
    df=df.rename(columns={c: (s,c.replace(f' {s}','')) for s in suffixes for c in df if c.endswith(f' {s}')},
                errors='raise')
    df.columns=pd.MultiIndex.from_tuples(df.columns)
    return df

## dtype conversion
@add_method_to_class(rd)
def colobj2str(df,test=False):
    cols_obj=df.dtypes[df.dtypes=='object'].index.tolist()
    if test:
        logging.info(cols_obj)
    for c in cols_obj:
        df[c]=df[c].astype('|S80')
    return df

@add_method_to_class(rd)
def split_rows(df,collist,rowsep=None):
    """
    for merging dfs with names with df with synonymns
    param colsyn: col containing tuples of synonymns 
    """
    if not rowsep is None:
        df.loc[:,collist]=df.loc[:,collist].apply(lambda x : x.split(rowsep))
    return dellevelcol(df.set_index([c for c in df if c!=collist])[collist].apply(pd.Series).stack().reset_index().rename(columns={0:collist},
                                                                                                                         errors='raise'))        
### alias
meltlistvalues=split_rows

## apply
@add_method_to_class(rd)
def apply_as_map(df,index,columns,values,
                 fun,**kws):
    """
    :params fun: map to map
    """
    df1=df.pivot(index=index,columns=columns,values=values)
    df2=fun(df1,**kws)
    return df2.melt(ignore_index=False,value_name=values).reset_index()

@add_method_to_class(rd)
def apply_expand_ranges(df,col_list=None,col_start=None,col_end=None,fun=range,
                       col_out='position'):
    if col_list is None:
        col_list='_col_list'
        df[col_list]=df.apply(lambda x: range(x[col_start],x[col_end]+1),axis=1)
    df1=df[col_list].apply(pd.Series)
    if len(df1)==1:
        df1=df1.T
        df1.columns=[col_out]
        return df1
    else:
        return dmap2lin(df1).rename(columns={'value':col_out},
                                   errors='raise')[col_out].dropna()

    
## drop duplicates by aggregating the dups
@add_method_to_class(rd)
def drop_duplicates_by_agg(df,cols_groupby,cols_value,aggfunc='mean',fast=False):
    col2aggfunc={}
    for col in cols_value:
        if isinstance(aggfunc,dict):
            aggfunc_=aggfunc[col]
        else:
            aggfunc_=aggfunc
        if not isinstance(aggfunc_,list):
            aggfunc_=[aggfunc_]
        col2aggfunc[col]=[getattr(np,k) if isinstance(k,str) else k for k in aggfunc_]
    def agg(x,col2aggfunc):
        xout=pd.Series()
        for col in col2aggfunc:
            for fun in col2aggfunc[col]:
                xout[f'{col} {fun.__name__}']=fun(x[col])
            if x[col].dtype in [float,int]:
                xout[f'{col} var']=np.var(x[col])
        return xout
#         agg({k:col2aggfunc[k]+[np.std] for k in cols_value})
#     logging.info(cols_groupby)
#     logging.info(col2aggfunc)
#     logging.info(df.columns.tolist())
    df1=getattr(df.groupby(cols_groupby),f"{'progress' if not fast else 'parallel'}_apply")(lambda x: agg(x,col2aggfunc))
#     df1.columns=[c.replace(f' {aggfunc}','') for c in coltuples2str(df1.columns)]
    return df1.reset_index()

## sorting
@add_method_to_class(rd)
def sort_col_by_list(df, col,l):
    df[col]=pd.Categorical(df[col],categories=l, ordered=True)
    df=df.sort_values(col)
    df[col]=df[col].astype(str)
    return df

@add_method_to_class(rd)
def dfswapcols(df,cols):
    df[f"_{cols[0]}"]=df[cols[0]].copy()
    df[cols[0]]=df[cols[1]].copy()
    df[cols[1]]=df[f"_{cols[0]}"].copy()
    df=df.drop([f"_{cols[0]}"],axis=1)
    return df

# def dfsortbybins(df, col):
#     d=dict(zip(bins,[float(s.split(',')[0].split('(')[1]) for s in bins]))
#     df[f'{col} dfrankbybins']=df.apply(lambda x : d[x[col]] if not pd.isnull(x[col]) else x[col], axis=1)
#     df=df.sort_values(f'{col} dfrankbybins').drop(f'{col} dfrankbybins',axis=1)
#     return df

# def sort_binnnedcol(df,col):
#     df[f'_{col}']=df[col].apply(lambda s : float(s.split('(')[1].split(',')[0]))
#     df=df.sort_values(by=f'_{col}')
#     df=df.drop([f'_{col}'],axis=1)
#     return df

## apply_agg
def agg_by_order(x,order):
    """
    TODO: convert categories to numbers and take min
    """
    # damaging > other non-conserving > other conserving
    if len(x)==1:
#         print(x.values)
        return list(x.values)[0]
    for k in order:
        if k in x.values:
            return k
def agg_by_order_counts(x,order):
    """
    demo:
    df=pd.DataFrame({'a1':['a','b','c','a','b','c','d'],
    'b1':['a1','a1','a1','b1','b1','b1','b1'],})
    df.groupby('b1').apply(lambda df : agg_by_order_counts(x=df['a1'],
                                                   order=['b','c','a'],
                                                   ))
    """    
    ds=x.value_counts()
    ds=ds.add_prefix(f"{x.name}=")
    ds[x.name]=agg_by_order(x,order)
    return ds.to_frame('').T

@add_method_to_class(rd)
def groupby_agg_merge(df,col_groupby,col_aggby,
                      funs=['mean'],
                      ascending=True):
    if subset is None:
        df1=gs.agg({col_sortby:getattr(np,func)}).reset_index()
        df2=df.merge(df1,
                on=col_groupby,how='inner',suffixes=['',f' per {col_groupby}'])
        logging.warning(f'column added to df: {col_sortby} per {col_groupby}')
        return df2.sort_values(f'{col_sortby} per {col_groupby}',ascending=ascending)

@add_method_to_class(rd)
def groupby_sort_values(df,col_groupby,col_sortby,
                 subset=None,
                 col_subset=None,
                 func='mean',ascending=True):
    gs=df.groupby(col_groupby)
    if subset is None:
        df1=gs.agg({col_sortby:getattr(np,func)}).reset_index()
        df2=df.merge(df1,
                on=col_groupby,how='inner',suffixes=['',f' per {col_groupby}'])
        logging.warning(f'column added to df: {col_sortby} per {col_groupby}')
        return df2.sort_values(f'{col_sortby} per {col_groupby}',ascending=ascending)
    else:
        df1=df.groupby(col_subset).get_group(subset)
        df2=df1.groupby(col_groupby).agg({col_sortby:getattr(np,func)}).reset_index()
        return sort_col_by_list(df, 
                                col_groupby,
                                df2.sort_values(col_sortby,
                                                ascending=ascending)[col_groupby])
#         return df2.sort_values(f'{col_sortby} per {col_groupby}',ascending=ascending)
sort_values_groupby=groupby_sort_values

@add_method_to_class(rd)
def sort_columns_by_values(df,cols_sortby=['mutation gene1','mutation gene2'],
                            suffixes=['gene1','gene2'], # no spaces
                            ):
    """
    sorts in ascending order. 
    `sorted` means values are sorted because gene1>gene2. 
    """
    assert((df.rd.check_na_percentage(cols=cols_sortby)==0).all())
    suffixes=[s.replace(' ','') for s in suffixes]
    dn2df={}
    # keys: (equal, to be sorted)
    dn2df[(False,False)]=df.loc[(df[cols_sortby[0]]<df[cols_sortby[1]]),:]
    dn2df[(False,True)]=df.loc[(df[cols_sortby[0]]>df[cols_sortby[1]]),:]
    dn2df[(True,False)]=df.loc[(df[cols_sortby[0]]==df[cols_sortby[1]]),:]
    dn2df[(True,True)]=df.loc[(df[cols_sortby[0]]==df[cols_sortby[1]]),:]
    ## rename columns of of to be sorted
    rename={c:c.replace(suffixes[0],suffixes[1]) if (suffixes[0] in c) else c.replace(suffixes[1],suffixes[0]) if (suffixes[1] in c) else c for c in df}
    for k in [True, False]:
        dn2df[(k,True)]=dn2df[(k,True)].rename(columns=rename,
                                              errors='raise')
        
    df1=pd.concat(dn2df,names=['equal','sorted']).reset_index([0,1])
    logging.info(df1.groupby(['equal','sorted']).size())
    return df1

# @add_method_to_class(rd)
# def sort_by_column_pairs_many_categories(df,
#     preffixes=['gene','protein'],
#     suffixes=[1,2],
#     test=False,
#     ):
#     """
#     append reciprocal
#     """
#     from rohan.dandage.io_strs import replacemany
#     kws={i:[f"{s}{i}" for s in preffixes] for i in suffixes}
#     replaces={i:{s:s.replace(str(i),str(suffixes[0] if i==suffixes[1] else suffixes[1])) for s in kws[i]} for i in kws}
#     renames={c:replacemany(c,replaces[1]) if any([s in c for s in kws[1]]) else replacemany(c,replaces[2]) if any([s in c for s in kws[2]]) else c for c in df.columns}
#     if test:
#         logging.info('renames',renames)
#     df1=dellevelcol(pd.concat({False:df,
#                       True:df.rename(columns=renames)},names=['is reciprocal']).reset_index())
#     if 'index' in df1:
#         df1=df1.drop(['index'],axis=1)
#     return df1
# def apply_sorted_column_pair(x,colvalue,suffixes,categories=None,how='all',
#                                     test=False):
#     """
#     Apply
#     Checks if values in pair of columns is sorted.
    
#     Numbers are sorted in ascending order.
    
#     :returns : True if sorted else 
#     """
#     if categories is None:
#         if x[f'{colvalue} {suffixes[0]}'] < x[f'{colvalue} {suffixes[1]}']:
#             return True
#         else:
#             return False            
#     else:
#         if test:
#             logging.info([x[f'{colvalue} {suffixes[0]}'],x[f'{colvalue} {suffixes[1]}']],
#               categories,
#               getattr(np,how)([x[f'{colvalue} {suffixes[0]}']==categories[0],
#                             x[f'{colvalue} {suffixes[1]}']==categories[1]]),
#               getattr(np,how)([x[f'{colvalue} {suffixes[0]}']==categories[1],
#                             x[f'{colvalue} {suffixes[1]}']==categories[0]]))
#         if categories[0]!=categories[1]:
#             if getattr(np,how)([x[f'{colvalue} {suffixes[0]}']==categories[0],
#                                 x[f'{colvalue} {suffixes[1]}']==categories[1]]):
#                 return True
#             elif getattr(np,how)([x[f'{colvalue} {suffixes[0]}']==categories[1],
#                                   x[f'{colvalue} {suffixes[1]}']==categories[0]]):
#                 return False
#             else:
#                 return np.nan        
#         else:
#             return True            
        
# @add_method_to_class(rd)
# def sort_by_column_pairs(df,colvalue,suffixes,categories=None,how='all',test=False,fast=True): 
#     """
#     sort values in pair of columns and sort the index accordingly.
#     """
#     suffix2cols={s:sorted(df.filter(like=s).columns.tolist()) for s in suffixes}
#     if len(suffix2cols[suffixes[0]])!=len(suffix2cols[suffixes[1]]):
#         logging.error("df should contain paired columns")
#         logging.info(suffix2cols)
#         return 
#     from rohan.dandage.io_sets import list2intersection
#     if len(list2intersection(list(suffix2cols.values())))!=0:
#         logging.error("df should contain non-overlapping paired columns")
#         logging.info(suffix2cols)
#         return 
# #     df['fsorted {colvalue}']=getattr(df,'parallel_apply' if fast else 'apply')(lambda x: apply_sorted_column_pair(x,colvalue=colvalue,
# #                                                                                                       suffixes=suffixes,
# #                                                                                                       categories=categories,
# #                                                                                                       how=how,test=test),axis=1)
#     df[f'sorted {colvalue}']=((df[f'{colvalue}{suffixes[0]}']==categories[0]) & (df[f'{colvalue}{suffixes[1]}']==categories[1]))
#     if test:
#         logging.info(df.shape)
# #     df=df.dropna(subset=[f'sorted {colvalue}'])
# #     df[f'sorted {colvalue}']=df[f'sorted {colvalue}'].astype(bool)
#     if test:
#         logging.info(df.shape)
#     df1,df2=df.loc[df[f'sorted {colvalue}'],:],df.loc[~df[f'sorted {colvalue}'],:]
#     # rename cols of df2 (not sorted) 
#     rename=dict(zip(suffix2cols[suffixes[0]]+suffix2cols[suffixes[1]],
#              suffix2cols[suffixes[1]]+suffix2cols[suffixes[0]]))
#     if test:
#         logging.info(rename)
#     df2=df2.rename(columns=rename)
#     df3=df1.append(df2)
#     if test:
#         logging.info(df1.shape,df2.shape,df3.shape)
#     return df3#.drop([f'sorted {colvalue}'],axis=1)

# quantile bins
@add_method_to_class(rd)
def aggcol_by_qbins(df,colx,coly,colgroupby=None,bins=10):
    """
    get_stats_by_bins(df,colx,coly,fun,bins=4)
    """
    from rohan.dandage.stat.transform import get_qbins
    df[f"{colx} qbin midpoint"]=get_qbins(ds=df[colx],
                                          bins=bins,
                                          value='mid')
#     qcut(df[colx],bins,duplicates='drop')    
    if colgroupby is None:
        colgroupby='del'
        df[colgroupby]='del'
    from rohan.dandage.stat.variance import confidence_interval_95
    dplot=df.groupby([f"{colx} qbin",colgroupby]).agg({coly:[np.mean,confidence_interval_95],})
    from rohan.dandage.io_dfs import coltuples2str
    dplot.columns=coltuples2str(dplot.columns)
    dplot=dplot.reset_index()
    dplot[f"{colx} qbin midpoint"]=dplot[f"{colx} qbin"].apply(lambda x:x.mid).astype(float)
    dplot[f"{colx} qbin midpoint"]=dplot[f"{colx} qbin midpoint"].apply(float)
    if 'del' in dplot:
        dplot=dplot.drop(['del'],axis=1)
    return dplot

# subsets
from rohan.dandage.io_sets import dropna
@add_method_to_class(rd)
def get_intersectionsbysubsets(df,cols_fracby2vals,
                               cols_subset,
                               col_ids,
                               bins
#                                params_qcut={'bins':10},
                              ):
    """
    cols_fracby:
    cols_subset:
    """
    from rohan.dandage.stat.transform import get_qbins
    for coli,col in enumerate(cols_subset):
        if is_col_numeric(df[col]):
            try:
                df[f"{col} bin"]=get_qbins(ds=df[col],bins=bins,value='mid')
#                 qcut(df[col],**params_qcut)
            except:
                logging.info(col)
            cols_subset[coli]=f"{col} bin"
    for col_fracby in cols_fracby2vals:
        val=cols_fracby2vals[col_fracby]
        ids=df.loc[(df[col_fracby]==val),col_ids].dropna().unique()
        for col_subset in cols_subset:
            for subset in dropna(df[col_subset].unique()):
                ids_subset=df.loc[(df[col_subset]==subset),col_ids].dropna().unique()
                df.loc[(df[col_subset]==subset),f'P {col_fracby} {col_subset}']=len(set(ids_subset).intersection(ids))/len(ids_subset)
    return df


@add_method_to_class(rd)
def get_colsubset2stats(dannot,colssubset=None):
    if colssubset is None:
        colssubset=dannot_stats.columns
    dannot_stats=dannot.loc[:,colssubset].apply(pd.Series.value_counts)

    colsubset2classes=dannot_stats.apply(lambda x: x.index,axis=0)[dannot_stats.apply(lambda x: ~pd.isnull(x),axis=0)].apply(lambda x: dropna(x),axis=0).to_dict()
    colsubset2classns=dannot_stats[dannot_stats.apply(lambda x: ~pd.isnull(x),axis=0)].apply(lambda x: dropna(x),axis=0).to_dict()
    colsubset2classns={k:[int(i) for i in colsubset2classns[k]] for k in colsubset2classns}
    return dannot_stats,colsubset2classes,colsubset2classns
# 2d subsets
@add_method_to_class(rd)
def subset_cols_by_cutoffs(df,col2cutoffs,quantile=False,outdf=False,
                           fast=False,
                           test=False):    
    for col in col2cutoffs:
        if isinstance(col2cutoffs[col],float):
            if quantile:
                cutoffs=[col2cutoffs[col],col2cutoffs[col]]
                col2cutoffs[col]=[df[col].quantile(c) for c in cutoffs]
            else:
                col2cutoffs[col]=[col2cutoffs[col],col2cutoffs[col]]
        elif not isinstance(col2cutoffs[col],list):
            logging.error("cutoff should be float or list")
        df.loc[(df[col]<=col2cutoffs[col][0]),f"{col} (high or low)"]='low'
        df.loc[(df[col]>col2cutoffs[col][1]),f"{col} (high or low)"]='high'
        logging.info(df.loc[:,f"{col} (high or low)"].isnull().sum())
    colout=f"{'-'.join(list(col2cutoffs.keys()))} (low or high)"
    def get_subsetname(x):
        l=[x[f"{col} (high or low)"] for col in col2cutoffs]
        if not any([pd.isnull(i) for i in l]):
            return '-'.join(l)
        else:
            return np.nan
    df[colout]=getattr(df,'progress_apply' if not fast else 'parallel_apply')(lambda x:  get_subsetname(x),axis=1)
    if test:
        logging.info(col2cutoffs)
        if len(col2cutoffs.keys())==2:
            element2color={'high-high':'r',
                   'high-low':'g',
                   'low-high':'b',
                   'low-low':'k',
                  }
            import matplotlib.pyplot as plt
            ax=plt.subplot()
            df.groupby(colout).apply(lambda x: x.plot.scatter(x=list(col2cutoffs.keys())[0],
                                                              y=list(col2cutoffs.keys())[1],
                                                              alpha=1,ax=ax,label=x.name,color=element2color[x.name]))
    if not outdf:
        return df[colout]
    else:
        return df

## make ids
get_ids_sorted=lambda x: '--'.join(sorted(x))
@add_method_to_class(rd)
def make_ids_sorted(df,cols,ids_have_equal_length):
    if ids_have_equal_length:
        return np.apply_along_axis(get_ids_sorted, 1, df.loc[:,cols].values)
    else:
        return df.loc[:,cols].agg(lambda x: '--'.join(sorted(x)),axis=1)

@add_method_to_class(rd)    
def split_ids(df1,col):
    df=df1[col].str.split('--',expand=True)
    for i in range(len(df.columns)):
        df1[f"{col} {i+1}"]=df[i]
    return df1

@add_method_to_class(rd)
def sort_ids_paired(df1,cols=['g1','g2'],col=None):
    if col is None:
        col=f"{cols[0].split('1')[0]}s id"
    ## combine id
    df1[col]=make_ids_sorted(df1,cols,ids_have_equal_length=True)
    ## drop dups
    df1=df1.drop(cols,axis=1).log.drop_duplicates()
    ## split id
#     if any(df1[cols[0]]==df1[cols[1]]):
#         df1['homomeric interaction']=df1[cols[0]]==df1[cols[1]]
    return df1
    
## merge/map ids
@add_method_to_class(rd)
def map_ids(df,df2,colgroupby,col_mergeon,order_subsets=None,**kws_merge):
    """
    :params df: target
    :params df2: sources. labels in colgroupby 
    """
    order_subsets=df[colgroupby].unique() if colgroupby is None else order_subsets
    dn2df={}
    for k in order_subsets:
        dn2df[k]=df.merge(df2.groupby(colgroupby).get_group(k),
                       on=col_mergeon,
#                        how='inner',suffixes=[' Broad',''],
                     **kws_merge)
        dn2df[k]['merged on']=col_mergeon
        df=df.loc[~df[col_mergeon].isin(dn2df[k][col_mergeon]),:]
        logging.info(df[col_mergeon].nunique())
    df3=pd.concat(dn2df,axis=0,names=[colgroupby]).reset_index(drop=True)
    return df3,df

## tables io
def dict2df(d,colkey='key',colvalue='value'):
    d={k:d[k] if isinstance(d[k],list) else list(d[k]) for k in d}
    return pd.DataFrame(pd.concat({k:pd.Series(d[k]) for k in d})).droplevel(1).reset_index().rename(columns={'index':colkey,0:colvalue},
                                                                                                    errors='raise')

## append
def append_dfs(dfs,cols_index=None,cols_value=None):
    from rohan.dandage.io_sets import list2intersection,unique
    if cols_index is None: cols_index=list(list2intersection([list(df) for df in dfs]))
    if cols_value is None: cols_value=[[c for c in df if not c in cols_index] for df in dfs]
    coli2cols={i:list(cols) for i,cols in enumerate(list(zip(*cols_value)))}
    for i in coli2cols:
        dtypes=unique([dfs[dfi][col].dtype for dfi,col in enumerate(coli2cols[i])])
        if len(dtypes)!=1:
            logging.error('the dtypes of columns should match')   
        dtype=dtypes[0]
        if dtype.startswith('float') or dtype.startswith('int'):
            renameto='value'
        df1=pd.concat({col:dfs[dfi].rename(columns={col:renameto},
                                          errors='raise').loc[:,cols_index+[renameto]] for dfi,col in enumerate(coli2cols[i])},axis=0,names=['variable']).reset_index(level=0)
    return df1

## merge dfs
def merge_dfs(dfs,
             **params_merge):
    from functools import reduce
    return reduce(lambda df1,df2: pd.merge(df1,df2,**params_merge), dfs)

def merge_dfs_auto(dfs,how='left',suffixes=['','_'],
              test=False,fast=False,drop_duplicates=True,
              sort=True,
              **params_merge):
    """
    
    """
    from rohan.dandage.io_sets import list2intersection,flatten
    if isinstance(dfs,dict):
        dfs=list(dfs.values())
    if all([isinstance(df,str) for df in dfs]):
        dfs=[read_table(p) for p in dfs]
    if not 'on' in params_merge:
        params_merge['on']=list(list2intersection([df.columns for df in dfs]))
        if len(params_merge['on'])==0:
            logging.error('no common columns found for infer params_merge[on]')
            return
    else:
        if isinstance(params_merge['on'],str):
            params_merge['on']=[params_merge['on']]
    params_merge['how']=how
    params_merge['suffixes']=suffixes
    # sort largest first
    if test:
        logging.info(params_merge)
        d={dfi:[len(df)] for dfi,df in enumerate(dfs)}
        logging.info(f'size: {d}')
    dfi2cols_value={dfi:df.select_dtypes([int,float]).columns.tolist() for dfi,df in enumerate(dfs)}
    cols_common=list(np.unique(params_merge['on']+list(list2intersection(dfi2cols_value.values()))))
    dfi2cols_value={k:list(set(dfi2cols_value[k]).difference(cols_common)) for k in dfi2cols_value}
    dfis_duplicates=[dfi for dfi in dfi2cols_value if len(dfs[dfi])!=len(dfs[dfi].loc[:,cols_common].drop_duplicates())]
    if test:
        logging.info(f'cols_common: {cols_common}',)
        logging.info(f'dfi2cols_value: {dfi2cols_value}',)
        logging.info(f'duplicates in dfs: {dfis_duplicates}',)
    for dfi in dfi2cols_value:
        if (dfi in dfis_duplicates) and drop_duplicates:
            dfs[dfi]=drop_duplicates_by_agg(dfs[dfi],cols_common,dfi2cols_value[dfi],fast=fast)
    if sort:
        d={dfi:[len(df)] for dfi,df in enumerate(dfs)}
        logging.info(f"size agg: {d}")
        from rohan.dandage.io_dict import sort_dict
        sorted_indices_by_size=sort_dict({dfi:[len(df.drop_duplicates(subset=params_merge['on']))] for dfi,df in enumerate(dfs)},0)
        logging.info(f'size dedup: {sorted_indices_by_size}')
        sorted_indices_by_size=list(sorted_indices_by_size.keys())#[::-1]
        dfs=[dfs[i] for i in sorted_indices_by_size]
#     from functools import reduce
#     df1=reduce(lambda df1,df2: pd.merge(df1,df2,**params_merge), dfs)
    df1=merge_dfs(dfs,**params_merge)
    cols_std=[f"{c} var" for c in flatten(list(dfi2cols_value.values())) if f"{c} var" in df1]
    cols_del=[c for c in cols_std if df1[c].isnull().all()]
    df1=df1.drop(cols_del,axis=1)
    return df1

def merge_subset(df,colsubset,subset,cols_value,
                          on,how='left',suffixes=['','.1'],
                          **kws_merge):
    """
    merge a subset from a linear df, sideways
    """
    if isinstance(on,str): on=[on]
    return df.loc[(df[colsubset]!=subset),:].merge(
                                            df.loc[(df[colsubset]==subset),on+cols_value],
                                          on=on,
                                          how=how, 
                                        suffixes=suffixes,
                                          **kws_merge,
                                            )


