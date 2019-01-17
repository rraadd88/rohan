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



from os.path import basename,dirname,exists
from os import makedirs
import pandas as pd
import numpy as np

import logging

def df2info(df,col_searches=None):
    if col_searches is None:
        if len(df.columns)>5:
            print('**COLS**: ',df.columns.tolist())
        print('**HEAD**: ',df.loc[:,df.columns[:5]].head())
        print('**SHAPE**: ',df.shape)
    else:
        cols_searched=[c2 for c1 in col_searches for c2 in df if c1 in c2]
        print('**SEARCHEDCOLS**:\n',cols_searched)
        print('**HEAD**: ',df.loc[:,cols_searched].head())
        
def del_Unnamed(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'Unnamed' in c]
    return df.drop(cols_del,axis=1)

def set_index(data,col_index):
    """
    Sets the index if the index is not present

    :param data: pandas table 
    :param col_index: column name which will be assigned as a index
    """
    if col_index in data:
        data=data.reset_index().set_index(col_index)
        if 'index' in data:
            del data['index']
        return data
    elif data.index.name==col_index:
        return data
    else:
        logging.error("set_index: something's wrong with the df")
        df2info(data)

#tsv io
def read_table(p):
    if p.endswith('.tsv') or p.endswith('.tab'):
        return del_Unnamed(pd.read_table(p))
    elif p.endswith('.csv'):
        return del_Unnamed(pd.read_csv(p,sep=','))
    elif p.endswith('.pqt') or p.endswith('.parquet'):
        return del_Unnamed(read_table_pqt(p))
    else: 
        logging.error(f'unknown extension {p}')
def read_table_pqt(p):
    return del_Unnamed(pd.read_parquet(p,engine='fastparquet'))
    
def to_table(df,p):
    if p.endswith('.tsv') or p.endswith('.tab'):
        if not exists(dirname(p)) and dirname(p)!='':
            makedirs(dirname(p),exist_ok=True)
        df.to_csv(p,sep='\t')
    elif p.endswith('.pqt') or p.endswith('.parquet'):
        to_table_pqt(df,p)
    else: 
        logging.error(f'unknown extension {p}')        
def to_table_pqt(df,p):
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    df.to_parquet(p,engine='fastparquet',compression='gzip',)

def tsv2pqt(p):
    to_table_pqt(pd.read_table(p,low_memory=False),f"{p}.pqt")
    
def read_excel(p,sheet_name=None,):
    xl = pd.ExcelFile(p)
    xl.sheet_names  # see all sheet names
    if sheet_name is None:
        sheet_name=input(', '.join(xl.sheet_names))
    return xl.parse(sheet_name) 
    
#slice     
def dfvslicebysstr(df,sstr,include=True,how='and',outcols=False):
    if isinstance(sstr,str):        
        if include:
            cols=[c for c in df if sstr in c]
        else:
            cols=[c for c in df if not sstr in c]            
    elif isinstance(sstr,list):
        cols=[]
        for s in sstr:
            if include:
                cols.append([c for c in df if s in c])
            else:
                cols.append([c for c in df if not s in c])
#         print()
        from rohan.dandage.io_sets import list2union,list2intersection
        if how=='or':
            cols=list2union(cols)
        elif how=='and':
            cols=list2intersection(cols)
    else:
        logging.error('sstr should be str or list')
    if outcols:
        return cols
    else:
        return df.loc[:,cols]

# multi dfs
def concat_cols(df1,df2,idx_col,df1_cols,df2_cols,
                df1_suffix,df2_suffix,wc_cols=[],suffix_all=False):
    """
    Concatenates two pandas tables 

    :param df1: dataframe 1
    :param df2: dataframe 2
    :param idx_col: column name which will be used as a common index 
    """

    df1=df1.set_index(idx_col)
    df2=df2.set_index(idx_col)    
    if not len(wc_cols)==0:
        for wc in wc_cols:
            df1_cols=df1_cols+[c for c in df1.columns if wc in c]
            df2_cols=df2_cols+[c for c in df2.columns if wc in c]
    combo=pd.concat([df1.loc[:,df1_cols],df2.loc[:,df2_cols]],axis=1)
    # find common columns and rename them
    # print df1_cols
    # print df2_cols    
    if suffix_all:
        df1_cols=["%s%s" % (c,df1_suffix) for c in df1_cols]
        df2_cols=["%s%s" % (c,df2_suffix) for c in df2_cols]
        # df1_cols[df1_cols.index(col)]="%s%s" % (col,df1_suffix)
        # df2_cols[df2_cols.index(col)]="%s%s" % (col,df2_suffix)
    else:
        common_cols=[col for col in df1_cols if col in df2_cols]
        for col in common_cols:
            df1_cols[df1_cols.index(col)]="%s%s" % (col,df1_suffix)
            df2_cols[df2_cols.index(col)]="%s%s" % (col,df2_suffix)
    combo.columns=df1_cols+df2_cols
    combo.index.name=idx_col
    return combo     

def get_colmin(data):
    """
    Get rowwise column names with minimum values

    :param data: pandas dataframe
    """
    data=data.T
    colmins=[]
    for col in data:
        colmins.append(data[col].idxmin())
    return colmins

def fhs2data_combo(fhs,cols,index,labels=None,col_sep=': '):
    """
    Collates data from multiple csv files

    :param fhs: list of paths to csv files
    :param cols: list of column names to concatenate
    :param index: name of the column name to be used as the common index of the output pandas table 
    """

    if labels is None:
        labels=[basename(fh) for fh in fhs]
    if len(fhs)>0:
        for fhi,fh in enumerate(fhs):
            label=labels[fhi]
            data=pd.read_csv(fh).set_index(index)
            if fhi==0:
                data_combo=pd.DataFrame(index=data.index)
                for col in cols:
                    data_combo.loc[:,'%s%s%s' % (label,col_sep,col)]=data.loc[:,col]
            else:
                for col in cols:
                    data_combo.loc[:,'%s%s%s' % (label,col_sep,col)]=data.loc[:,col]    
        return del_Unnamed(data_combo)
    else:
        logging.error('no fhs found: len(fhs)=0')

def fhs2data_combo_appended(fhs, cols=None,labels=None,labels_coln='labels',sep=',',
                           error_bad_lines=True):
    """
    Collates data from multiple csv files vertically

    :param fhs: list of paths to csv files
    :param cols: list of column names to concatenate
    """    
    if labels is None:
        labels=[basename(fh) for fh in fhs]
    if len(fhs)>0:
        data_all=pd.DataFrame(columns=cols)
        for fhi,fh in enumerate(fhs):
            label=labels[fhi]
            try:
                data=pd.read_csv(fh,sep=sep,error_bad_lines=error_bad_lines)
            except:
                raise ValueError(f"something wrong with file pd.read_csv({fh},sep={sep})")
            if len(data)!=0:
                data.loc[:,labels_coln]=label
                if not cols is None:
                    data=data.loc[:,cols]
                data_all=data_all.append(data,sort=True)
        return del_Unnamed(data_all)

def rename_cols(df,names,renames=None,prefix=None,suffix=None):
    """
    rename columns of a pandas table

    :param df: pandas dataframe
    :param names: list of new column names
    """
    if not prefix is None:
        renames=[ "%s%s" % (prefix,s) for s in names]
    if not suffix is None:    
        renames=[ "%s%s" % (s,suffix) for s in names]
    if not renames is None:
        for i,name in enumerate(names):
#             names=[renames[i] if s==names[i] else s for s in names]    
            rename=renames[i]    
            df.loc[:,rename]=df.loc[:,name]
        df=df.drop(names,axis=1)
        return df 


def reorderbydf(df2,df1):
    """
    Reorder rows of a dataframe by other dataframe

    :param df2: input dataframe
    :param df1: template dataframe 
    """
    df3=pd.DataFrame()
    for idx,row in df1.iterrows():
        df3=df3.append(df2.loc[idx,:])
    return df3

def dfswapcols(df,cols):
    df[f"_{cols[0]}"]=df[cols[0]].copy()
    df[cols[0]]=df[cols[1]].copy()
    df[cols[1]]=df[f"_{cols[0]}"].copy()
    df=df.drop([f"_{cols[0]}"],axis=1)
    return df

def df2unstack(df,coln='columns',idxn='index',col='value'):
    if (df.columns.name is None) or (coln!='columns'):
        df.columns.name=coln
    if (df.index.name is None) or (idxn!='columns'):
        df.index.name=idxn
    df=df.unstack()
    df.name=col
    return pd.DataFrame(df).reset_index()

from rohan.dandage.io_strs import replacelist
def lin_dfpair(df,df1cols,df2cols,cols_common,replace_suffix):
    dfs=[]
    for cols in [df1cols,df2cols]:
        df_=df[cols+cols_common]
        df_=df_.rename(columns=dict(zip(df_.columns,replacelist(df_.columns,replace_suffix))))
        dfs.append(df_)
    dfout=dfs[0].append(dfs[1])
    return dfout

def lambda2cols(df,lambdaf,in_coln,to_colns):
    df_=df.apply(lambda x: lambdaf(x[in_coln]),
                 axis=1).apply(pd.Series)
    df_.columns=to_colns
    df=df.join(df_)        
    return df

def df2chucks(din,chunksize,outd,fn,return_fmt='\t',force=False):
    """
    :param return_fmt: '\t': tab-sep file, lly, '.', 'list': returns a list
    """
    from os.path import exists#,splitext,dirname,splitext,basename,realpath
    from os import makedirs
    din.index=range(0,len(din),1)

    chunkrange=list(np.arange(0,len(din),chunksize))
    chunkrange=list(zip([c+1 if ci!=0 else 0 for ci,c in enumerate(chunkrange)],chunkrange[1:]+[len(din)-1]))

    chunk2range={}
    for ri,r in enumerate(chunkrange):    
        chunk2range[ri+1]=r

    if not exists(outd):
        makedirs(outd)
    chunks=[]
    chunkps=[]
    for chunk in chunk2range:
        chunkp='{}/{}_chunk{:08d}.tsv'.format(outd,fn,chunk)
        rnge=chunk2range[chunk]
        din_=din.loc[rnge[0]:rnge[1],:]
        if not exists(chunkp) or force:
            if return_fmt=='list':
                chunks.append(din_)
            else:
                din_.to_csv(chunkp,sep=return_fmt)
            del din_
        chunkps.append(chunkp)
    if return_fmt=='list':
        return chunks
    else:
        return chunkps        

## symmetric dfs eg. submaps    
    
def filldiagonal(df,cols,filler=None):
    try:
        d=df.loc[cols,cols]
    except:
        logging.error('cols should be in cols and idxs')
    if filler is None:
        filler=np.nan
    for r,c in zip(cols,cols):
        df.loc[r,c]=filler
    return df

def df2submap(df,col,idx,aggfunc='sum',binary=False,binaryby='nan'):
    df['#']=1
    dsubmap=df.pivot_table(columns=col,index=idx,values='#',
                       aggfunc=aggfunc)
    if binary:
        if binaryby=='nan':
            dsubmap=~pd.isnull(dsubmap)
        else:
            dsubmap=dsubmap!=binaryby
    return dsubmap

def completesubmap(dsubmap,fmt,
    fmt2vals={'aminoacid':["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y","*"], 
    'aminoacid_3letter':['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'],
    'codon':["TTT",    "TTC",    "TTA",  "TTG",  "TCT",  "TCC",  "TCA",  "TCG",  "TAT",  "TAC",  "TAA",  "TAG",  "TGT",  "TGC",  "TGA",  "TGG",  "CTT",  "CTC",  "CTA",  "CTG",  "CCT",  "CCC",  "CCA",  "CCG",  "CAT",  "CAC",  "CAA",  "CAG",  "CGT",  "CGC",  "CGA",  "CGG",  "ATT",  "ATC",  "ATA",  "ATG",  "ACT",  "ACC",  "ACA",  "ACG",  "AAT",  "AAC",  "AAA",  "AAG",  "AGT",  "AGC",  "AGA",  "AGG",  "GTT",  "GTC",  "GTA",  "GTG",  "GCT",  "GCC",  "GCA",  "GCG",  "GAT",  "GAC",  "GAA",  "GAG",  "GGT",  "GGC",  "GGA",  "GGG"],
    'nucleotide': ['A','T','G','C'],}):
    
    vals=fmt2vals[fmt]
    for v in vals: 
        if not v in dsubmap.columns:            
            dsubmap[v]=np.nan
        if not v in dsubmap.index:
            dsubmap.loc[v,:]=np.nan
    return dsubmap.loc[vals,vals]

def get_offdiagonal_values(dcorr,take_diag=False,replace=np.nan):
    for ii,i in enumerate(dcorr.index):
        for ci,c in enumerate(dcorr.columns):
            if ci>ii:
                dcorr.loc[i,c]=replace
            if not take_diag:
                if ci==ii:
                    dcorr.loc[i,c]=replace
    return dcorr

def get_offdiag_vals(dcorr):
    """
    for lin dcorr i guess
    """
    del_indexes=[]
    for spc1 in np.unique(dcorr.index.get_level_values(0)):
        for spc2 in np.unique(dcorr.index.get_level_values(0)):
            if (not (spc1,spc2) in del_indexes) and (not (spc2,spc1) in del_indexes):
                del_indexes.append((spc1,spc2))
    #         break
    for spc1 in np.unique(dcorr.index.get_level_values(0)):
        for spc2 in np.unique(dcorr.index.get_level_values(0)):
            if spc1==spc2:
                del_indexes.append((spc1,spc2))
    
    return dcorr.drop(del_indexes)

# aggregate dataframes

def dfaggregate_unique(df,colgroupby,colaggs):
    for colaggi,colagg in enumerate(colaggs):  
        ds=df.groupby(colgroupby)[colagg].apply(list).apply(pd.Series).apply(lambda x: x.dropna().unique(),axis=1)
        colaggname=f"{colagg}: list"
        if colaggname in df:
            colaggname=f"{colagg}: list: list"            
        ds.name=colaggname
        if all((ds.apply(len)==1) | (ds.apply(len)==0)):
            ds=ds.apply(lambda x : x[0] if len(x)>0 else np.nan)        
            ds.name=colagg       
        if colaggi==0:
            df_=pd.DataFrame(ds)
        else:
            df_=df_.join(ds)
    if df_.index.name in df_:
        df_.index.name=f"{colgroupby} groupby"
    return df_.reset_index()
        
        
def dropna_by_subset(df,colgroupby,colaggs,colval,colvar,test=False):
    df_agg=dfaggregate_unique(df,colgroupby,colaggs)
    df_agg['has values']=df_agg.apply(lambda x : len(x[f'{colval}: list'])!=0,axis=1)
    varswithvals=df_agg.loc[(df_agg['has values']),colvar].tolist()
    if test:
        df2info(df_agg)
    df=df.loc[df[colvar].isin(varswithvals),:] 
    return df

def df2colwise_unique_counts(df,cols,out=False):
    col2uniquec={}
    for col in cols:
        if col in df:
            col2uniquec[col]=len(df.loc[:,col].unique())
        else:
            col2uniquec[col]='column not found'
    dcol2uniquec=pd.Series(col2uniquec)
    if out:
        return dcol2uniquec
    else:
        print(dcol2uniquec)
        
def dfdupval2unique(df,coldupval,preffix_unique='variant'):  
    dups=df[coldupval].value_counts()[(df[coldupval].value_counts()>1)].index

    for dup in dups:
        ddup=df.loc[(df[coldupval]==dup)]
        df.loc[(df[coldupval]==dup),preffix_unique]=range(1, len(ddup)+1)
    #     break
    df[coldupval]=df.apply(lambda x : x[coldupval] if pd.isnull(x[preffix_unique]) else f"{x[coldupval]}: {preffix_unique} {int(x[preffix_unique])}",axis=1)
    return df

def dfliststr2dflist(df,colliststrs,colfmt='list'):
    import ast
    for c in colliststrs:
#         print(c)
        if colfmt=='list' or df[c].apply(lambda x : (('[' in x) and (']' in x))).all(): #is list
            df[c]=df.apply(lambda x : ast.literal_eval(x[c].replace("nan","''")) if not isinstance(x[c], (list)) else x[c],axis=1)
        elif colfmt=='tuple' or df[c].apply(lambda x : (('(' in x) and (')' in x))).all(): #is tuple        
            df[c]=df.apply(lambda x : ast.literal_eval(x[c]) if not isinstance(x[c], (tuple)) else x[c],axis=1)
        else:
            df[c]=df.apply(lambda x : x[c].replace("nan","").split(',') if not isinstance(x[c], (list)) else x[c],axis=1)
    return df

# multiindex
def coltuples2str(cols):
    from rohan.dandage.io_strs import tuple2str
    cols_str=[]
    for col in cols:
        cols_str.append(tuple2str(col))
    return cols_str

def colobj2str(df,test=False):
    cols_obj=df.dtypes[df.dtypes=='object'].index.tolist()
    if test:
        print(cols_obj)
    for c in cols_obj:
        df[c]=df[c].astype('|S80')
    return df

#merge
##fix for merge of object cols + non-object cols
def pd_merge_dfwithobjcols(df1,df2,left_on=None,right_on=None,on=None,
                          how='inner'):
    if not on is None:
        left_on=on
        right_on=on
    df1=set_index(df1,left_on)
    df2=set_index(df2,right_on)
    df=pd.concat([df1,df2], axis=1, join=how)
    return df.reset_index()

import logging
def merge_dn2df(dn2df,on,how='left',
               test=False):
    dn2dflen=dict(zip([len(dn2df[dn].drop_duplicates(subset=on)) for dn in dn2df.keys()],dn2df.keys()))
    if test:
        print(dn2dflen)
    for dni,dflen in enumerate(sorted(dn2dflen,reverse=True)):
        dn=dn2dflen[dflen]
        df=dn2df[dn]
        df_ddup=df.drop_duplicates(subset=on)
        if len(df)!=len(df_ddup):
            df=df_ddup.copy()
            logging.warning(f'{dn}: dropped duplicates. size drop from {len(df)} to {len(df_ddup)}')
        if dni==0:
            cols=[c for c in df.columns.tolist() if (not ((c in on) or (c==on))) and (c in df)]
            df=df.rename(columns=dict(zip(cols,[f"{c} {dn}" for c in cols])))
            dfmerged=df.copy()
        else:
            cols=[c for c in df.columns.tolist() if (not ((c in on) or (c==on))) and (c in df)]
            if test:
                print(dn,cols)
                print(dict(zip(cols,[f"{c} {dn}" for c in cols])))
            df=df.rename(columns=dict(zip(cols,[f"{c} {dn}" for c in cols])))
            dfmerged=dfmerged.merge(df,on=on,how=how,
#                                     suffixes=['',f" {dn}"],
                                   )
            if test:
                print(f" {dn}",dfmerged.columns.tolist(),df.columns.tolist())
        del df
    return dfmerged

# def dfsyn2appended(df,colsyn):
#     """
#     for merging dfs with names with df with synonymns
#     param colsyn: col containing tuples of synonymns 
#     """
#     colsynappended=colsyn+' appended'
#     df.index=range(len(df))
#     #make duplicated row for each synonymn
#     dfdup=pd.DataFrame(columns=df.columns.tolist()+[colsynappended])
#     for i in df.index:
#         for syn in df.loc[i,colsyn]:
#             ds=df.loc[i,:]
#             ds[colsynappended]=syn
#             dfdup=dfdup.append(ds,ignore_index=True)
#             del ds
#         del i
#     #         break
#     #     break
#     return dfdup

def dfsyn2appended(df,colsyn):
    """
    for merging dfs with names with df with synonymns
    param colsyn: col containing tuples of synonymns 
    """
    colsynappended=colsyn+' appended'
    df.index=range(len(df))
    #make duplicated row for each synonymn
    dfsynappended=df[colsyn].apply(pd.Series).unstack().reset_index().drop('level_0',axis=1).set_index('level_1')
    dfsynappended.columns=[colsynappended]
    dfsynappended=dfsynappended.dropna()
    return dfsynappended.join(df,how='left')

## drop duplicates by aggregating the dups
def drop_duplicates_agg(df,colsgroupby,cols2aggf,test=False):
    """
    colsgroupby: unique names ~index
    cols2aggf: rest of the cols `unique_dropna_str` for categories
    """
    if test:
        print(df.shape)
        print(df.drop_duplicates(subset=colsgroupby).shape)
    #ddup aggregated
    dfdupagg=df.loc[(df.duplicated(subset=colsgroupby,keep=False)),:].groupby(colsgroupby).agg(cols2aggf)
    #drop duplicates all
    df_=df.drop_duplicates(subset=colsgroupby,keep=False)
    if test:
        print(df_.shape)
    #append ddup aggregated
    dfout=df_.append(dfdupagg,sort=True)
    if test:
        print(dfout.shape)
    return dfout

## sorting

def dfsortbybins(df, col):
    d=dict(zip(bins,[float(s.split(',')[0].split('(')[1]) for s in bins]))
    df[f'{col} dfrankbybins']=df.apply(lambda x : d[x[col]] if not pd.isnull(x[col]) else x[col], axis=1)
    df=df.sort_values(f'{col} dfrankbybins').drop(f'{col} dfrankbybins',axis=1)
    return df

def sortdfcolbylist(df, col,l):
    df[col]=pd.Categorical(df[col],categories=l, ordered=True)
    return df.sort_values(col)

from rohan.dandage.io_sets import dropna
def get_intersectionsbysubsets(df,cols_fracby2vals,cols_subset,col_ids):
    """
    cols_fracby:
    cols_subset:
    """
    for col_fracby in cols_fracby2vals:
        val=cols_fracby2vals[col_fracby]
        ids=df.loc[(df[col_fracby]==val),col_ids].dropna().unique()
        for col_subset in cols_subset:
            for subset in dropna(df[col_subset].unique()):
                ids_subset=df.loc[(df[col_subset]==subset),col_ids].dropna().unique()
                df.loc[(df[col_subset]==subset),f'P {col_fracby} {col_subset}']=len(set(ids_subset).intersection(ids))/len(ids_subset)
    return df

# import from stat
from rohan.dandage.stat.transform import dflogcol