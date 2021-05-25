"""
io_df -> io_dfs -> io_files
"""
import pandas as pd
import numpy as np
import logging
from icecream import ic

from rohan.dandage import add_method_to_class
from rohan.global_imports import rd

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
def renameby_replace(df,replaces,ignore=True,**kws):
    from rohan.dandage.io_strs import replacemany
    df.columns=[replacemany(c,replaces,ignore=ignore,**kws) for c in df]
    return df

@add_method_to_class(rd)
def drop_constants(df):
    cols_del=df.nunique().loc[lambda x: x==1].index.tolist()
    logging.warning(f"dropped columns: {', '.join(cols_del)}")
    return df.drop(cols_del,axis=1)

@add_method_to_class(rd)
def clean(df,cols=[],
          drop_constants=False,
          drop_unnamed=True,
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

## nans:
@add_method_to_class(rd)
def check_na_percentage(df,cols=None):
    if cols is None:
        cols=df.columns.tolist()
    return (df.loc[:,cols].isnull().sum()/df.loc[:,cols].agg(len))*100

## duplicates:
@add_method_to_class(rd)
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
def get_mappings(df1,cols=None,keep='1:1'):
    """
    validate by df1.rd.check_mappings(cols)
    """
    if cols is None:
        cols=df1.columns.tolist()
    if df1.rd.check_duplicated(cols):
        df1=df1.loc[:,cols].log.drop_duplicates()
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

@add_method_to_class(rd)
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
    
@add_method_to_class(rd)
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

@add_method_to_class(rd)        
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
    if plot:
        from rohan.dandage.plot.bar import plot_bar_intersections
        return plot_bar_intersections(dplot=ds,colvalue=colindex,**kws_plot)
    else:
        return ds

#filter df
@add_method_to_class(rd)
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
@add_method_to_class(rd)
def to_dict(df,cols,drop_duplicates=False):
    df=df.log.dropna(subset=cols)
    if drop_duplicates:
        df=df.loc[:,cols].drop_duplicates()
    if not df.rd.check_duplicated([cols[0]]):
        return df.set_index(cols[0])[cols[1]].to_dict()
    else:
        return df.groupby(cols[0])[cols[1]].unique().to_dict()        

## conversion
@add_method_to_class(rd)
def get_bools(df,cols,drop=False):
    """
    prefer pd.get_dummies
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

@add_method_to_class(rd)
def agg_bools(df1,cols):
    """
    reverse pd.get_dummies
    
    :params df1: bools
    """
    col='+'.join(cols)
#     print(df1.loc[:,cols].T.sum())
    assert(all(df1.loc[:,cols].T.sum()==1))
    for c in cols:
        df1.loc[df1[c],col]=c
    return df1[col]     

## paired dfs
@add_method_to_class(rd)
def melt_paired(df,
                cols_index=None,
                suffixes=None,
                ):
    """
    TODO: assert whether melted
    """
    if suffixes is None and not cols_index is None:
        from rohan.dandage.io_strs import get_suffix
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

@add_method_to_class(rd)
def get_chunks(df1,colindex,colvalue,bins=None,value='right'):
    """
    based on other df
    
    :params colvalue: value within [0-100]
    """
    from rohan.dandage.io_sets import unique,nunique
    if bins==0:
        df1['chunk']=bins
        logging.warning("bins=0, so chunks=1")
        return df1['chunk']
    elif bins is None:
        bins=int(np.ceil(df1.memory_usage().sum()/1e9))
    df2=df1.loc[:,[colindex,colvalue]].drop_duplicates()
    from rohan.dandage.stat.transform import get_bins
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
# @add_method_to_class(rd)
# unpair_df=melt_paired

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
    def merge(self,**kws):
        from rohan.dandage.io_dfs import log_apply
        return log_apply(self._obj,fun='merge',**kws)