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
from rohan.lib.io_df import *
from rohan.lib import to_class,rd
        
def filter_dfs(dfs,cols,how='inner'):
    """
    
    """
    def apply_(dfs,col,how):
        from rohan.lib.io_sets import list2intersection,list2union
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

@to_class(rd)
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
    else:
        logging.warning(f"contains potentially redundant columns e.g. {cols_del[0]}")
        if how=='left':
            logging.info("maybe incomplete merge; try how='inner'")
    d1['to  ']=dfpair_merge2.shape        
    if verb and d1['from']!=d1['to  ']:
        for k in d1:
            logging.info(f'shape changed {k} {d1[k]}')
    return dfpair_merge2

### alias to be deprecated   
merge_dfpairwithdf=merge_paired
merge_dfs_paired_with_unpaireds=merge_paired

## append
def append_dfs(dfs,cols_index=None,cols_value=None):
    from rohan.lib.io_sets import list2intersection,unique
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
    logging.info(f"merge_dfs: shape changed from : dfs shape={[df.shape for df in dfs]}")
    df3=reduce(lambda df1,df2: pd.merge(df1,df2,**params_merge), dfs)
    logging.info(f"merge_dfs: shape changed to   : {df3.shape}")
    return df3

def merge_dfs_auto(dfs,how='left',suffixes=['','_'],
              test=False,fast=False,drop_duplicates=True,
              sort=True,
              **params_merge):
    """
    
    """
    from rohan.lib.io_sets import list2intersection,flatten
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
        from rohan.lib.io_dict import sort_dict
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


