#!usr/bin/python

# Copyright 2018, Rohan Dandage <rraadd_8@hotmail.com,rohan@igib.in>
# This program is distributed under General Public License v. 3.  

"""
================================
``io_dfs``
================================

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
from rohan.dandage.io_files import *
from rohan.dandage.io_sys import is_interactive_notebook
import pandas as pd
import numpy as np
import sys
import logging
from rohan.dandage import add_method_to_class
from rohan.global_imports import rd

## log
def log_apply(df, fun, *args, **kwargs):
    logging.info(f'{fun}: shape changed from {df.shape}', )
    df1 = getattr(df, fun)(*args, **kwargs)
    logging.info(f'{fun}: shape changed to   {df1.shape}')
    return df1

## delete unneeded columns
@add_method_to_class(rd)
def drop_unnamedcol(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'Unnamed' in c]
    return df.drop(cols_del,axis=1)
### alias
delunnamedcol=drop_unnamedcol

@add_method_to_class(rd)
def drop_levelcol(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'level' in c]
    return df.drop(cols_del,axis=1)
### alias
dellevelcol=drop_levelcol

@add_method_to_class(rd)
def flatten_columns(df):
    df.columns=coltuples2str(df.columns)
    return df

@add_method_to_class(rd)
def rename_byreplace(df,replaces,**kws):
    from rohan.dandage.io_strs import replacemany
    df.columns=[replacemany(c,replaces,**kws) for c in df]
    return df

@add_method_to_class(rd)
def clean(df,cols=[]):
    """
    Deletes temporary columns
    :param df: pandas dataframe
    """
    cols_del=df.filter(regex="^(?:index|level|Unnamed|chunk|_).*$").columns.tolist()+df.filter(regex="^.*(?:\.1)$").columns.tolist()+cols
    if len(cols_del)!=0:
        logging.warning(f"dropped columns: {', '.join(cols_del)}")
        return df.drop(cols_del,axis=1)
    else:
        return df
@add_method_to_class(rd)
def compress(df1,coff_categories=20,test=False):
    if test: ini=df1.memory_usage().sum()
    ds=df1.select_dtypes('object').nunique()
    for c in ds[ds<=coff_categories].index:
        df1[c]=df1[c].astype('category')
    if test: logging.info(f"compression={((ini-df1.memory_usage().sum())/ini)*100:.1f}%")
    return df1

@add_method_to_class(rd)
def clean_compress(df,**kws_compress): return df.rd.clean().rd.compress(**kws_compress)
    
#filter df
@add_method_to_class(rd)
def filter_rows(df,d,sign='==',logic='and',test=False):
    qry = f' {logic} '.join([f"`{k}` {sign} '{v}'" for k,v in d.items()])
    df1=df.query(qry)
    if test:
        logging.info(df1.loc[:,list(d.keys())].drop_duplicates())
    if len(df1)==0:
        logging.warning('may be some column names are wrong..')
        logging.warning([k for k in d if not k in df])
    return df1

filter_rows_bydict=filter_rows

def filter_dfs(dfs,col,how='inner'):
    from rohan.dandage.io_sets import list2intersection,list2union
    if how=='inner':
        l=list(list2intersection([df[col].tolist() for df in dfs]))
    elif how=='outer':
        l=list(list2union([df[col].tolist() for df in dfs]))
    else:
        raise ValueError("")
    logging.info(f"len({col})={len(l)}")
    return [df.loc[(df[col].isin(l)),:] for df in dfs]

## conversion
def convert_to_bools(df,col):
    for s in df[col].unique():
        df[f"{col} {s}"]=df[col]==s
    return df

## conversion to type
@add_method_to_class(rd)
def to_dict(df,cols):
    if not df.rd.check_duplicated([cols[0]]):
        return df.set_index(cols[0])[cols[1]].to_dict()

## reshape df
@add_method_to_class(rd)
def dmap2lin(df,idxn='index',coln='column',colvalue_name='value'):
    """
    dmap: ids in index and columns 
    idxn,coln of dmap -> 1st and 2nd col
    TODO: deprecate. use melt(ignore_index=False) instead.
    """
#     df.columns.name=coln
    if isinstance(df.index,pd.MultiIndex):
        df.index.names=[f"index level {i}" if pd.isnull(idxn) else idxn for i,idxn in enumerate(df.index.names)]
        id_vars=df.index.names        
    else:
        df.index.name=idxn
        id_vars=[df.index.name]
    return df.reset_index().melt(id_vars=id_vars,
                             var_name=coln,value_name=colvalue_name)
### alias
@add_method_to_class(rd)
def map2lin(df, var_name_index='index',
         var_name_col='column',
         value_name='value',):
    """
    TODO: deprecate. use melt(ignore_index=False) instead.
    """
    return dmap2lin(df,idxn=var_name,coln=var_name_col,colvalue_name=value_name)
    

## paired dfs
@add_method_to_class(rd)
def melt_paired(df,
                suffixes=['gene1','gene2'],
                ):
    cols_common=[c for c in df if not any([s in c for s in suffixes])]
    dn2df={}
    for s in suffixes:
        cols=[c for c in df if s in c]
        dn2df[s]=df.loc[:,cols_common+cols].rename(columns={c:c.replace(s,'') for c in cols})
    df1=pd.concat(dn2df,axis=0,names=['suffix']).reset_index(0)
    return df1.rename(columns={c: c[:-1] if c.endswith(' ') else c[1:] if c.startswith(' ') else c for c in df1})
### alias
# @add_method_to_class(rd)
unpair_df=melt_paired

@add_method_to_class(rd)
def merge_paired(dfpair,df,
                        left_ons=['gene1 id','gene2 id'],
                        right_on='gene id',right_ons_common=[],
                        suffixes=[' gene1',' gene2'],how='left',
                                    dryrun=False, 
                                    test=False,
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
        dfpair_merge2=dfpair_merge2.rename(columns=dict(zip(cols_same_right_ons_common[0::2],right_ons_common)))
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
    df=df.rename(columns={c: (s,c.replace(f' {s}','')) for s in suffixes for c in df if c.endswith(f' {s}')})
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
        df.loc[:,colsyn]=df.loc[:,colsyn].apply(lambda x : x.split(rowsep))
    return dellevelcol(df.set_index([c for c in df if c!=collist])[collist].apply(pd.Series).stack().reset_index().rename(columns={0:collist}))        
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
        return dmap2lin(df1).rename(columns={'value':col_out})[col_out].dropna()

## nans:
@add_method_to_class(rd)
def check_na_percentage(df,cols=None):
    if cols is None:
        cols=df.columns.tolist()
    return (df.loc[:,cols].isnull().sum()/df.loc[:,cols].agg(len))*100

## duplicates:
@add_method_to_class(rd)
def check_duplicated(df,cols):
    if df.duplicated(subset=cols).any():
        logging.error('duplicates in the table!')  
        return True
    else:
        return False
        
@add_method_to_class(rd)        
def check_mappings(df,cols=None):
    """
    identify duplicates within columns
    """
    if cols is None:
        cols=df.columns.tolist()
    import itertools
    d={}
    for t in list(itertools.permutations(cols,2)):
        d[t]=df.groupby(t[0])[t[1]].nunique().value_counts()
    return pd.concat(d,axis=0,ignore_index=False,names=['from','to','map to']).to_frame('map from').sort_index().reset_index(-1).loc[:,['map from','map to']]

@add_method_to_class(rd)        
def check_intersections(df,colgroupby=None,colvalue=None,plot=False,**kws_plot):
    if isinstance(df,dict):
        df=dict2df(df)
        colgroupby='key'
        colvalue='value'
    if df.rd.check_duplicated([colgroupby,colvalue]):
        df=df.log.drop_duplicates(subset=[colgroupby,colvalue])
    df['_value']=True
    df1=df.pivot(index=colvalue,columns=colgroupby,values='_value').fillna(False)
    ds=df1.groupby(df1.columns.to_list()).size()
    if plot:
        from rohan.dandage.plot.bar import plot_bar_intersections
        return plot_bar_intersections(dplot=ds,colvalue=colvalue,**kws_plot)
    else:
        return ds
    
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
        dn2df[(k,True)]=dn2df[(k,True)].rename(columns=rename)
        
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
def get_chunks(df1,colindex,colvalue,bins=10,value='right')
    """
    based on other df
    bins=int(np.ceil(df2.memory_usage().sum()/0.1e9))
    """
    from rohan.dandage.stat.transform import get_qbins
    d1=get_qbins(df1.set_index(colindex)[colvalue],
                 bins=bins,
                 value=value)
    df1['chunk']=df1[colindex].map(d1)
    df1['chunk']=df1['chunk'].astype(int)
    return df1['chunk']

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
    return pd.DataFrame(pd.concat({k:pd.Series(d[k]) for k in d})).droplevel(1).reset_index().rename(columns={'index':colkey,0:colvalue})

from rohan.dandage.io_text import get_header

def read_table(p,
               params_read_csv={},
               ext=None,
               **kws_manytables,):
    """
    'decimal':'.'
    
    examples:
    s='.vcf|vcf.gz'
    read_table(p,
               params_read_csv=dict(
               #compression='gzip',
               sep='\t',comment='#',header=None,
               names=replacemany(get_header(path,comment='#',lineno=-1),['#','\n'],'').split('\t'))
               )
    """
    if isinstance(p,list) or '*' in p:
        return read_manytables(p,params_read_csv=params_read_csv,
                               **kws_manytables)
    if len(params_read_csv.keys())!=0:
        return drop_unnamedcol(pd.read_csv(p,**params_read_csv))        
    else:
        if any([(p.endswith(s) or s==ext) for s in ['.tsv','.tsv.gz','.tab','.txt']]):
            return drop_unnamedcol(pd.read_csv(p,sep='\t',
                                              compression='gzip' if p.endswith('.gz') else None,
                                              ))
        elif any([(p.endswith(s) or s==ext) for s in ['.csv','.csv.gz']]):
            return drop_unnamedcol(pd.read_csv(p,sep=',',
                                              compression='gzip' if p.endswith('.gz') else None,
                                              ))
        elif any([(p.endswith(s) or s==ext) for s in ['.pqt','.parquet']]):#p.endswith('.pqt') or p.endswith('.parquet'):
            return read_table_pqt(p)
        elif p.endswith('.vcf') or p.endswith('.vcf.gz'):
            from rohan.dandage.io_strs import replacemany
            return pd.read_table(p,
#                        params_read_csv=dict(
                       compression='gzip' if p.endswith('.vcf.gz') else None,
                       sep='\t',comment='#',header=None,
                       names=replacemany(get_header(path=p,comment='#',lineno=-1),['#','\n'],'').split('\t'),
#                                 )
                       )            
        else: 
            logging.error(f'unknown extension {p}')
                        
def read_table_pqt(p):
    return pd.read_parquet(p,engine='fastparquet').rd.clean()
def read_ps(ps):
    if isinstance(ps,str) and '*' in ps:
        ps=glob(ps)
    return ps
def apply_on_paths(ps,func,
                    fast=False, 
                   read_path=False,
                   drop_index=True,
                   fun_rename_path=None,
                   progress_bar=True,
                   params_read_csv={},
                   **kws,
                  ):
    def read_table_(df,read_path=False):
        p=df.iloc[0,:]['path']
        if read_path:
            return p
        else:
            return read_table(p,params_read_csv=params_read_csv,)
    if not fun_rename_path is None:
        drop_index=False
    ps=read_ps(ps)
    if len(ps)==0:
        logging.error('no paths found')
        return
    df1=pd.DataFrame({'path':ps})
    if fast and not progress_bar:
        progress_bar=True
    df2=getattr(df1.groupby('path',as_index=True),
                            f"{'parallel' if fast else 'progress'}_apply" if progress_bar else "apply"
               )(lambda df: func(read_table_(df,read_path=read_path),
                                 **kws))
    df2=df2.rd.clean().reset_index(drop=drop_index).rd.clean()
    if not fun_rename_path is None:
        if fun_rename_path=='basenamenoext':
            fun_rename_path=basenamenoext
        rename={p:fun_rename_path(p) for p in df2['path'].unique()}
        df2['path']=df2['path'].map(rename)
    return df2

def read_manytables(ps,
                    fast=False,
                    drop_index=True,
                    to_dict=False,
                    params_read_csv={},
                    **kws,
                   ):
    if not to_dict:
        df2=apply_on_paths(ps,func=lambda df: df,fast=fast,drop_index=drop_index,
                           params_read_csv=params_read_csv,
                           **kws)
        return df2
    else:
        return {p:read_table(p,
                             params_read_csv=params_read_csv) for p in read_ps(ps)}
    
## save table
def to_table(df,p,
             colgroupby=None,
             test=False,**kws):
    if is_interactive_notebook(): test=True
#     from rohan.dandage.io_strs import make_pathable_string
#     p=make_pathable_string(p)
    if not 'My Drive' in p:
        p=p.replace(' ','_')
    else:
        logging.warning('probably working on google drive; space/s left in the path.')
    if not df.index.name is None:
        df=df.reset_index()
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    if not colgroupby is None:
        to_manytables(df,p,colgroupby)
        return
    if p.endswith('.tsv') or p.endswith('.tab'):
        df.to_csv(p,sep='\t')
        if test:
            logging.info(p)
    elif p.endswith('.pqt') or p.endswith('.parquet'):
        to_table_pqt(df,p,**kws)
        if test:
            logging.info(p)
    else: 
        logging.error(f'unknown extension {p}')
        
def to_manytables(df,p,colgroupby):
    outd,ext=splitext(p)
    df.groupby(colgroupby).progress_apply(lambda x: to_table(x,f"{outd}/{x.name if not isinstance(x.name, tuple) else '/'.join(x.name)}{ext}"))
    
def to_table_pqt(df,p,**kws_pqt):
    if len(df.index.names)>1:
        df=df.reset_index()    
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    df.to_parquet(p,engine='fastparquet',compression='gzip',**kws_pqt)

def tsv2pqt(p):
    to_table_pqt(pd.read_csv(p,sep='\t',low_memory=False),f"{p}.pqt")
def pqt2tsv(p):
    to_table(read_table(p),f"{p}.tsv")
    
def read_excel(p,sheet_name=None,params_read_excel={},to_dict=False):
#     if not 'xlrd' in sys.modules:
#         logging.error('need xlrd to work with excel; pip install xlrd')
    if not to_dict:
        if sheet_name is None:
            xl = pd.ExcelFile(p)
#             xl.sheet_names  # see all sheet names
            if sheet_name is None:
                sheet_name=input(', '.join(xl.sheet_names))
            return xl.parse(sheet_name) 
        else:
            return pd.read_excel(p, sheet_name, **params_read_excel)
    else:
        xl = pd.ExcelFile(p)
        # see all sheet names
        sheetname2df={}
        for sheet_name in xl.sheet_names:
            sheetname2df[sheet_name]=xl.parse(sheet_name) 
        return sheetname2df
        
def to_excel(sheetname2df,datap,append=False):
#     if not 'xlrd' in sys.modules:
#         logging.error('need xlrd to work with excel; pip install xlrd')
    writer = pd.ExcelWriter(datap)
    startrow=0
    for sn in sheetname2df:
        if not append:
            sheetname2df[sn].to_excel(writer,sn)
        else:
            sheetname2df[sn].to_excel(writer,startrow=startrow)  
            startrow+=len(sheetname2df[sn])+2
    writer.save()

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
        df1=pd.concat({col:dfs[dfi].rename(columns={col:renameto}).loc[:,cols_index+[renameto]] for dfi,col in enumerate(coli2cols[i])},axis=0,names=['variable']).reset_index(level=0)
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

@pd.api.extensions.register_dataframe_accessor("log")
class log:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj        
    def dropna(self,**kws):
        from rohan.dandage.io_dfs import log_apply
        return log_apply(self._obj,fun='dropna',**kws)
    def drop_duplicates(self,**kws):
        from rohan.dandage.io_dfs import log_apply
        return log_apply(self._obj,fun='drop_duplicates',**kws)    
    def drop(self,**kws):
        from rohan.dandage.io_dfs import log_apply
        return log_apply(self._obj,fun='drop',**kws)    
    def pivot(self,**kws):
        from rohan.dandage.io_dfs import log_apply
        return log_apply(self._obj,fun='pivot',**kws)
    def pivot_table(self,**kws):
        from rohan.dandage.io_dfs import log_apply
        return log_apply(self._obj,fun='pivot_table',**kws)    
    def melt(self,**kws):
        from rohan.dandage.io_dfs import log_apply
        return log_apply(self._obj,fun='melt',**kws)
    def stack(self,**kws):
        from rohan.dandage.io_dfs import log_apply
        return log_apply(self._obj,fun='stack',**kws)
    def unstack(self,**kws):
        from rohan.dandage.io_dfs import log_apply
        return log_apply(self._obj,fun='unstack',**kws)
