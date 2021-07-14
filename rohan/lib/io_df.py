"""
io_df -> io_dfs -> io_files
"""
import pandas as pd
import numpy as np
import logging
from icecream import ic

from rohan.lib import to_class,rd

## log
def log_apply(df, fun, *args, **kwargs):
    """
    """
    d1={}
    d1['from']=df.shape
    df = getattr(df, fun)(*args, **kwargs)
    d1['to  ']=df.shape
    if d1['from']!=d1['to  ']:
        for k in d1:
            logging.info(f'{fun}: shape changed {k} {d1[k]}')
    return df

## delete unneeded columns
@to_class(rd)
def drop_unnamedcol(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'Unnamed' in c]
    return df.drop(cols_del,axis=1)
### alias
delunnamedcol=drop_unnamedcol

@to_class(rd)
def drop_levelcol(df):
    """
    Deletes all the temporary columns names "level".

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'level' in c]
    return df.drop(cols_del,axis=1)
### alias
dellevelcol=drop_levelcol

@to_class(rd)
def flatten_columns(df):
    df.columns=coltuples2str(df.columns)
    return df

@to_class(rd)
def renameby_replace(df,replaces,ignore=True,**kws):
    from rohan.lib.io_strs import replacemany
    df.columns=[replacemany(c,replaces,ignore=ignore,**kws) for c in df]
    return df

@to_class(rd)
def drop_constants(df):
    cols_del=df.nunique().loc[lambda x: x==1].index.tolist()
    logging.warning(f"dropped columns: {', '.join(cols_del)}")
    return df.drop(cols_del,axis=1)

@to_class(rd)
def dropby_patterns(df1,l1,test=False):
    s0='|'.join(l1).replace('(','\(').replace(')','\)')
    s1=f"^.*({s0}).*$"
    cols=df1.filter(regex=s1).columns.tolist()
    if test: info(s1)
    assert(len(cols)!=0)
    logging.info('columns dropped:'+','.join(cols))
    return df1.log.drop(labels=cols,axis=1)

@to_class(rd)
def clean(df,cols=[],
          drop_constants=False,
          drop_unnamed=True,
          verb=True,
         ):
    """
    Deletes temporary columns
    :param df: pandas dataframe
    """
    cols_del=df.filter(regex="^(?:index|level|Unnamed|chunk|_).*$").columns.tolist()+df.filter(regex="^.*(?:\.1)$").columns.tolist()+cols
    if drop_constants:
        df=df.rd.drop_constants()
    if not drop_unnamed:
        cols_del=[c for c in cols_del if not c.startswith('Unnamed')]
    if any(df.columns.duplicated()):
#         from rohan.lib.io_sets import unique
        if verb: logging.warning(f"duplicate column/s dropped:{df.loc[:,df.columns.duplicated()].columns.tolist()}")
        df=df.loc[:,~(df.columns.duplicated())]
    if len(cols_del)!=0:
        if verb: logging.warning(f"dropped columns: {', '.join(cols_del)}")
        return df.drop(cols_del,axis=1)
    else:
        return df
@to_class(rd)
def compress(df1,coff_categories=20,test=False):
    if test: ini=df1.memory_usage().sum()
    ds=df1.select_dtypes('object').nunique()
    for c in ds[ds<=coff_categories].index:
        df1[c]=df1[c].astype('category')
    if test: logging.info(f"compression={((ini-df1.memory_usage().sum())/ini)*100:.1f}%")
    return df1

@to_class(rd)
def clean_compress(df,**kws_compress): return df.rd.clean().rd.compress(**kws_compress)

## nans:
@to_class(rd)
def check_na_percentage(df,cols=None):
    """
    prefer check_na
    """
    if cols is None:
        cols=df.columns.tolist()
    return (df.loc[:,cols].isnull().sum()/df.loc[:,cols].agg(len))*100

@to_class(rd)
def check_na(df,cols=None):
    if cols is None:
        cols=df.columns.tolist()
    return (df.loc[:,cols].isnull().sum()/df.loc[:,cols].agg(len))*100

## nunique:
@to_class(rd)
def check_nunique(df,cols=None,):
    if cols is None:
        cols=df.select_dtypes(object).columns.tolist()
    return df.loc[:,cols].nunique()

## nunique:
@to_class(rd)
def check_inflation(df1,cols=None,):
    if cols is None:
        cols=df1.columns.tolist()
    return df1.loc[:,cols].apply(lambda x: (x.value_counts().values[0]/len(df1))*100)
#     df.loc[:,cols].nunique()
    
## duplicates:
@to_class(rd)
def check_duplicated(df,cols=None,subset=None):
    if not cols is None and not subset is None: logging.error(f"cols and subset are alias, both cannot be used.")        
    if cols is None and not subset is None: cols=subset        
    if cols is None:
        cols=df.columns
    if df.duplicated(subset=cols).any():
        logging.error('duplicates in the table!')  
        return True
    else:
        return False

## mappings    
@to_class(rd)        
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

@to_class(rd)
def get_mappings(df1,cols=None,keep='1:1',clean=False):
    """
    validate by df1.rd.check_mappings(cols)
    """
    if cols is None:
        cols=df1.columns.tolist()
    if df1.rd.check_duplicated(cols):
        df1=df1.loc[:,cols].log.drop_duplicates()
    if len(cols)==2:
        from rohan.lib.io_sets import get_alt
        df2=df1.copy()
        cols2=[]
        for c in cols:
            d1=df2.groupby(c)[get_alt(cols,c)].nunique().to_dict()
            c2=f"{c}:{get_alt(cols,c)}"
            df2[c2]=df2[c].map(d1)
            cols2.append(c2)
        df2['mapping']=df2.loc[:,cols2].apply(lambda x: ':'.join(["1" if i==1 else 'm' for i in x]),axis=1)
        if clean:
            df2=df2.drop(cols2,axis=1)
        if keep=='1:1':
            return df2.rd.filter_rows({'mapping':'1:1'})
        else:
            logging.info(df2['mapping'].value_counts())
            return df2
    else:
        d1={'1:1':df1.copy(),
           }
        if keep!='1:1':
            d1['not']=pd.DataFrame()
        import itertools
    #     for t in list(itertools.permutations(cols,2)):
        for c in cols:
            d1['1:1']=d1['1:1'].groupby(c).filter(lambda df: len(df)==1)
            if keep!='1:1':
                d1['not']=d1['not'].append(df1.copy().groupby(c).filter(lambda df: len(df)!=1))
        if keep=='1:1':
            logging.info(df1.shape)
            logging.info(d1['1:1'].shape)
            return d1['1:1']
        else:
            assert(len(df1)==len(d1['1:1'])+len(d1['not']))
            return pd.concat(d1,axis=0,names=['mapping']).reset_index()

@to_class(rd)
def filterby_mappings(df1,cols=None,maps=['1:1'],test=False):
    """
    :cols :
    """
    d1={}
    d1['from']=df1.shape
    
    if cols is None:
        cols=df1.columns.tolist()
    assert(len(cols)==2)
    if df1.rd.check_duplicated(cols):
        df1=df1.loc[:,cols].log.drop_duplicates()
    if isinstance(maps,str):
        maps=[maps]
    if '1:m' in maps or '1:1' in maps:
        df1=df1.loc[(df1[cols[0]].isin(df1[cols[0]].value_counts().loc[lambda x: x==1].index)),:]
    if 'm:1' in maps or '1:1' in maps:
        df1=df1.loc[(df1[cols[1]].isin(df1[cols[1]].value_counts().loc[lambda x: x==1].index)),:]
    if test: logging.info(df1.rd.check_mappings())

    d1['to  ']=df1.shape
    if d1['from']!=d1['to  ']:
        for k in d1:
            logging.info(f'shape changed {k} {d1[k]}')        
    return df1
# def get_rate(df1,cols1,cols2):
#     return df1.groupby(cols1).apply(lambda df: len(df.loc[:,cols2].drop_duplicates()))
# def get_rate_bothways(df,cols1,cols2,stat=None):
#     if isinstance(cols1,str):
#         cols1=[cols1]
#     if isinstance(cols2,str):
#         cols1=[cols2]
#     if not stat is None:
#         return pd.Series({f"{' '.join(cs2)} count {stat} per {' '.join(cs1)}" :getattr(get_rate(df,cols1=cs1,cols2=cs2,),stat)() for cs1, cs2 in [[cols1,cols2],[cols2,cols1]]})
#     else:
#         return pd.concat({f"{' '.join(cs2)} count per {' '.join(cs1)}" :get_rate(df,cols1=cs1,cols2=cs2,) for cs1, cs2 in [[cols1,cols2],[cols2,cols1]]},
#                          names=['rate'],
#                         axis=0)
    
@to_class(rd)
def to_map_binary(df,colgroupby=None,colvalue=None):
    """
    linear mappings to binary map
    no mappings -> False
    """
    colgroupby=[colgroupby] if isinstance(colgroupby,str) else colgroupby
    colvalue=[colvalue] if isinstance(colvalue,str) else colvalue
    if df.rd.check_duplicated(colgroupby+colvalue):
        df=df.log.drop_duplicates(subset=colgroupby+colvalue)
    df['_value']=True
    df1=df.pivot(index=colvalue,columns=colgroupby,values='_value').fillna(False)
    return df1

@to_class(rd)        
def check_intersections(df,
                        colindex=None, # 'paralog pair'
                        colgroupby=None, # 'variable'
                        plot=False,**kws_plot):
    """
    'variable',
    lin -> map -> groupby (ds)
    """
    def map2groupby(df):
        assert(all(df.dtypes==bool))
        return df.groupby(df.columns.tolist()).size()
        
    if isinstance(df,pd.DataFrame):
        if not (colgroupby is None or colindex is None) :
            if isinstance(colgroupby,str):
                # lin
                df1=to_map_binary(df,colgroupby=colgroupby,colvalue=colindex)
                ds=df1.groupby(df1.columns.to_list()).size()
            elif isinstance(colgroupby,list):
                assert(not df.rd.check_duplicated([colindex]+colgroupby))
                # map
                df1=df.set_index(colindex).loc[:,colgroupby] 
            else:
                logging.error('colgroupby should be a str or list')
        else:
            # map
            ds=map2groupby(df)
    elif isinstance(df,pd.Series):
        ds=df
    elif isinstance(df,dict):
        ds=dict2df(d1).rd.check_intersections(colindex='value',colgroupby='key')
    else:
        ValueError()
    if plot:
        from rohan.lib.plot.bar import plot_bar_intersections
        return plot_bar_intersections(dplot=ds,colvalue=colindex,**kws_plot)
    else:
        return ds

#filter df
@to_class(rd)
def filter_rows(df,d,sign='==',logic='and',
                drop_constants=False,
                test=False,
                verb=True,
               ):
    if verb: logging.info(df.shape)    
    assert(all([isinstance(d[k],(str,list)) for k in d]))
    qry = f" {logic} ".join([f"`{k}` {sign} "+(f"'{v}'" if isinstance(v,str) else f"{v}") for k,v in d.items()])
    df1=df.query(qry)
    if test:
        logging.info(df1.loc[:,list(d.keys())].drop_duplicates())
        logging.warning('may be some column names are wrong..')
        logging.warning([k for k in d if not k in df])
    if verb: logging.info(df1.shape)
    if drop_constants:
        df1=df1.rd.drop_constants()
    return df1

filter_rows_bydict=filter_rows

## conversion to type
@to_class(rd)
def to_dict(df,cols,drop_duplicates=False):
    df=df.log.dropna(subset=cols)
    if drop_duplicates:
        df=df.loc[:,cols].drop_duplicates()
    if not df[cols[0]].duplicated().any():
        return df.set_index(cols[0])[cols[1]].to_dict()
    else:
        logging.warning('format: {key:list}')
        return df.groupby(cols[0])[cols[1]].unique().to_dict()        

## conversion
@to_class(rd)
def get_bools(df,cols,drop=False):
    """
    """
    for c in cols:
        df_=pd.get_dummies(df[c],
                                  prefix=c,
                                  prefix_sep=": ",
                                  dummy_na=False)
        df_=df_.replace(1,True).replace(0,False)
        df=df.join(df_)
        if drop:
            df=df.drop([c],axis=1)
    return df

@to_class(rd)
def agg_bools(df1,cols):
    """
    reverse pd.get_dummies
    
    :param df1: bools
    """
    col='+'.join(cols)
#     print(df1.loc[:,cols].T.sum())
    assert(all(df1.loc[:,cols].T.sum()==1))
    for c in cols:
        df1.loc[df1[c],col]=c
    return df1[col]     

## paired dfs
@to_class(rd)
def melt_paired(df,
                cols_index=None,
                suffixes=None,
                ):
    """
    TODO: assert whether melted
    """
    if suffixes is None and not cols_index is None:
        from rohan.lib.io_strs import get_suffix
        suffixes=get_suffix(*cols_index)
    # both suffixes should not be in any column name
    assert(not any([all([s in c for s in suffixes]) for c in df]))
    cols_common=[c for c in df if not any([s in c for s in suffixes])]
    dn2df={}
    for s in suffixes:
        cols=[c for c in df if s in c]
        dn2df[s]=df.loc[:,cols_common+cols].rename(columns={c:c.replace(s,'') for c in cols},
                                                  errors='raise')
    df1=pd.concat(dn2df,axis=0,names=['suffix']).reset_index(0)
    df2=df1.rename(columns={c: c[:-1] if c.endswith(' ') else c[1:] if c.startswith(' ') else c for c in df1},
                     errors='raise')
    if '' in df2:
        df2=df2.rename(columns={'':'id'},
               errors='raise')
    return df2

@to_class(rd)
def get_chunks(df1,colindex,colvalue,bins=None,value='right'):
    """
    based on other df
    
    :param colvalue: value within [0-100]
    """
    from rohan.lib.io_sets import unique,nunique
    if bins==0:
        df1['chunk']=bins
        logging.warning("bins=0, so chunks=1")
        return df1['chunk']
    elif bins is None:
        bins=int(np.ceil(df1.memory_usage().sum()/1e9))
    df2=df1.loc[:,[colindex,colvalue]].drop_duplicates()
    from rohan.lib.stat.transform import get_bins
    d1=get_bins(df2.set_index(colindex)[colvalue],
                 bins=bins,
                 value=value,
                ignore=True)
    ## number bins
    d_={k:f"chunk{ki+1:08d}_upto{int(k):03d}" for ki,k in enumerate(sorted(np.unique(list(d1.values()))))}
    ## rename bins
    d2={k:d_[d1[k]] for k in d1}
    assert(nunique(d1.values())==nunique(d2.values()))
    df1['chunk']=df1[colindex].map(d2)
    return df1['chunk']

### alias
# @to_class(rd)
# unpair_df=melt_paired

## symmetric dfs eg. submaps    
@to_class(rd)
def dfmap2symmcolidx(df,test=False):
    geneids=set(df.index).union(set(df.columns))
    df_symm=pd.DataFrame(columns=geneids,index=geneids)
    df_symm.loc[df.index,:]=df.loc[df.index,:]
    df_symm.loc[:,df.columns]=df.loc[:,df.columns]
    if test:
        logging.debug(df_symm.shape)
    return df_symm

@to_class(rd)
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

@to_class(rd)
def get_diagonalvals(df):
    if df.shape[0]!=df.shape[1]:
        logging.warning('df not symmetric')
#     ids=set(df.columns).intersection(df.index)
    ds=pd.Series(np.diag(df), index=[df.index, df.columns])
#     id2val={}
#     for i,c in zip(ids,ids):
#         id2val[i]=df.loc[i,c]
    return pd.DataFrame(ds,columns=['diagonal value']).reset_index()

@to_class(rd)
def fill_symmetricdf_indices(dsubmap,vals=None):
    if vals is None:
        vals=np.unique(dsubmap.index.tolist()+dsubmap.columns.tolist())
    for v in vals: 
        if not v in dsubmap.columns:            
            dsubmap[v]=np.nan
        if not v in dsubmap.index:
            dsubmap.loc[v,:]=np.nan
    return dsubmap.loc[vals,vals]

@to_class(rd)
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

@to_class(rd)
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
        
@to_class(rd)
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
    from rohan.lib.io_strs import tuple2str
    cols_str=[]
    for col in cols:
        cols_str.append(tuple2str(col,sep=sep))
    return cols_str

@to_class(rd)
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
@to_class(rd)
def colobj2str(df,test=False):
    cols_obj=df.dtypes[df.dtypes=='object'].index.tolist()
    if test:
        logging.info(cols_obj)
    for c in cols_obj:
        df[c]=df[c].astype('|S80')
    return df

@to_class(rd)
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
@to_class(rd)
def apply_as_map(df,index,columns,values,
                 fun,**kws):
    """
    :param fun: map to map
    """
    df1=df.pivot(index=index,columns=columns,values=values)
    df2=fun(df1,**kws)
    return df2.melt(ignore_index=False,value_name=values).reset_index()

@to_class(rd)
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
@to_class(rd)
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
@to_class(rd)
def sort_valuesby_list(df1,by,l1,**kws):return df1.sort_values(by=by, key=lambda x: pd.Series(l1),**kws)

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

@to_class(rd)
def groupby_agg_merge(df,col_groupby,col_aggby,
                      funs=['mean'],
                      ascending=True):
    if subset is None:
        df1=gs.agg({col_sortby:getattr(np,func)}).reset_index()
        df2=df.merge(df1,
                on=col_groupby,how='inner',suffixes=['',f' per {col_groupby}'])
        logging.warning(f'column added to df: {col_sortby} per {col_groupby}')
        return df2.sort_values(f'{col_sortby} per {col_groupby}',ascending=ascending)

@to_class(rd)
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

@to_class(rd)
def swap_paired_cols(df_,suffixes=['gene1', 'gene2']):
    rename={c:c.replace(suffixes[0],suffixes[1]) if (suffixes[0] in c) else c.replace(suffixes[1],suffixes[0]) if (suffixes[1] in c) else c for c in df_}
    return df_.rename(columns=rename,errors='raise')

@to_class(rd)
def sort_columns_by_values(df,cols_sortby=['mutation gene1','mutation gene2'],
                            suffixes=['gene1','gene2'], # no spaces
                            ):
    """
    sorts in ascending order. 
    `sorted` means values are sorted because gene1>gene2. 
    """
    assert((df.rd.check_na(cols=cols_sortby)==0).all())
    suffixes=[s.replace(' ','') for s in suffixes]
    dn2df={}
    # keys: (equal, to be sorted)
    dn2df[(False,False)]=df.loc[(df[cols_sortby[0]]<df[cols_sortby[1]]),:]
    dn2df[(False,True)]=df.loc[(df[cols_sortby[0]]>df[cols_sortby[1]]),:]
    dn2df[(True,False)]=df.loc[(df[cols_sortby[0]]==df[cols_sortby[1]]),:]
    dn2df[(True,True)]=df.loc[(df[cols_sortby[0]]==df[cols_sortby[1]]),:]
    ## rename columns of of to be sorted
    ## TODO: use swap_paired_cols
    rename={c:c.replace(suffixes[0],suffixes[1]) if (suffixes[0] in c) else c.replace(suffixes[1],suffixes[0]) if (suffixes[1] in c) else c for c in df}
    for k in [True, False]:
        dn2df[(k,True)]=dn2df[(k,True)].rename(columns=rename,
                                              errors='raise')
        
    df1=pd.concat(dn2df,names=['equal','sorted']).reset_index([0,1])
    logging.info(df1.groupby(['equal','sorted']).size())
    return df1         
    
# quantile bins
@to_class(rd)
def aggcol_by_qbins(df,colx,coly,colgroupby=None,bins=10):
    """
    get_stats_by_bins(df,colx,coly,fun,bins=4)
    """
    from rohan.lib.stat.transform import get_qbins
    df[f"{colx} qbin midpoint"]=get_qbins(ds=df[colx],
                                          bins=bins,
                                          value='mid')
#     qcut(df[colx],bins,duplicates='drop')    
    if colgroupby is None:
        colgroupby='del'
        df[colgroupby]='del'
    from rohan.lib.stat.variance import confidence_interval_95
    dplot=df.groupby([f"{colx} qbin",colgroupby]).agg({coly:[np.mean,confidence_interval_95],})
    dplot.columns=coltuples2str(dplot.columns)
    dplot=dplot.reset_index()
    dplot[f"{colx} qbin midpoint"]=dplot[f"{colx} qbin"].apply(lambda x:x.mid).astype(float)
    dplot[f"{colx} qbin midpoint"]=dplot[f"{colx} qbin midpoint"].apply(float)
    if 'del' in dplot:
        dplot=dplot.drop(['del'],axis=1)
    return dplot

# subsets
from rohan.lib.io_sets import dropna
@to_class(rd)
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
    from rohan.lib.stat.transform import get_qbins
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


@to_class(rd)
def get_colsubset2stats(dannot,colssubset=None):
    if colssubset is None:
        colssubset=dannot_stats.columns
    dannot_stats=dannot.loc[:,colssubset].apply(pd.Series.value_counts)

    colsubset2classes=dannot_stats.apply(lambda x: x.index,axis=0)[dannot_stats.apply(lambda x: ~pd.isnull(x),axis=0)].apply(lambda x: dropna(x),axis=0).to_dict()
    colsubset2classns=dannot_stats[dannot_stats.apply(lambda x: ~pd.isnull(x),axis=0)].apply(lambda x: dropna(x),axis=0).to_dict()
    colsubset2classns={k:[int(i) for i in colsubset2classns[k]] for k in colsubset2classns}
    return dannot_stats,colsubset2classes,colsubset2classns
# 2d subsets
@to_class(rd)
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
@to_class(rd)
def make_ids_sorted(df,cols,ids_have_equal_length):
    if ids_have_equal_length:
        return np.apply_along_axis(get_ids_sorted, 1, df.loc[:,cols].values)
    else:
        return df.loc[:,cols].agg(lambda x: '--'.join(sorted(x)),axis=1)
def get_alt_id(s1='A--B',s2='A'): return [s for s in s1.split('--') if s!=s2][0]

@to_class(rd)    
def split_ids(df1,col,sep='--'):
    df=df1[col].str.split(sep,expand=True)
    for i in range(len(df.columns)):
        df1[f"{col} {i+1}"]=df[i]
    return df1

reverse_ids_=lambda x: '--'.join(x.split('--')[::-1])                                 
@to_class(rd)
def reverse_ids(df,col,colonly=None,fast=False):
    """
    :param col: ids
    :param colonly: e.g. sorted
    """
    if colonly is None:
        return getattr(df[col],f"{'progress' if not fast else 'parallel'}_apply")(reverse_ids_)
    else:
        return getattr(df,f"{'progress' if not fast else 'parallel'}_apply")(lambda x: reverse_ids_(x[col]) if x[colonly] else x[col], axis=1)
                                 
## merge/map ids
@to_class(rd)
def map_ids(df,df2,colgroupby,col_mergeon,order_subsets=None,**kws_merge):
    """
    :param df: target
    :param df2: sources. labels in colgroupby 
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

@pd.api.extensions.register_dataframe_accessor("log")
class log:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj        
    def dropna(self,**kws):
        from rohan.lib.io_df import log_apply
        return log_apply(self._obj,fun='dropna',**kws)
    def drop_duplicates(self,**kws):
        from rohan.lib.io_df import log_apply
        return log_apply(self._obj,fun='drop_duplicates',**kws)    
    def drop(self,**kws):
        from rohan.lib.io_df import log_apply
        return log_apply(self._obj,fun='drop',**kws)    
    def pivot(self,**kws):
        from rohan.lib.io_df import log_apply
        return log_apply(self._obj,fun='pivot',**kws)
    def pivot_table(self,**kws):
        from rohan.lib.io_df import log_apply
        return log_apply(self._obj,fun='pivot_table',**kws)    
    def melt(self,**kws):
        from rohan.lib.io_df import log_apply
        return log_apply(self._obj,fun='melt',**kws)
    def stack(self,**kws):
        from rohan.lib.io_df import log_apply
        return log_apply(self._obj,fun='stack',**kws)
    def unstack(self,**kws):
        from rohan.lib.io_df import log_apply
        return log_apply(self._obj,fun='unstack',**kws)
    def merge(self,**kws):
        from rohan.lib.io_df import log_apply
        return log_apply(self._obj,fun='merge',**kws)
    def join(self,**kws):
        from rohan.lib.io_df import log_apply
        return log_apply(self._obj,fun='join',**kws)    