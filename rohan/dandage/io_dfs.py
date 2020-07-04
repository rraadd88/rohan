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
        

def delunnamedcol(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'Unnamed' in c]
    return df.drop(cols_del,axis=1)

def dellevelcol(df):
    """
    Deletes all the unnamed columns

    :param df: pandas dataframe
    """
    cols_del=[c for c in df.columns if 'level' in c]
    return df.drop(cols_del,axis=1)

def del_Unnamed(df):
    """
    to be deprecated
    """
    return delunnamedcol(df)
    
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

def reset_index(df):
    if df.index.name in df:
        df.index.name=df.index.name+' index'
    return df.reset_index()

#tsv io
def read_table(p,params_read_csv={}):
    """
    'decimal':'.'
    """
    if len(params_read_csv.keys())!=0:
        return del_Unnamed(pd.read_csv(p,**params_read_csv))        
    else:
        if p.endswith('.tsv') or p.endswith('.tab'):
            return del_Unnamed(pd.read_csv(p,sep='\t'))
        elif p.endswith('.csv'):
            return del_Unnamed(pd.read_csv(p,sep=','))
        elif p.endswith('.pqt') or p.endswith('.parquet'):
            return del_Unnamed(read_table_pqt(p))
        else: 
            logging.error(f'unknown extension {p}')
def read_table_pqt(p):
    return del_Unnamed(pd.read_parquet(p,engine='fastparquet'))

def read_manytables(ps,axis,collabel='label',labels=[],cols=[],params_read_csv={},params_concat={},
                   to_dict=False):
    if isinstance(ps,str):
        ps=glob(ps)
    if len(labels)!=0:
        if len(labels)!=len(ps):
            ValueError('len(labels)!=len(ps)')
    dn2df={}
    for pi,p in enumerate(ps):
        if len(labels)!=0:
            label=labels[pi]
        else:
            label=basenamenoext(p)
        df=read_table(p,params_read_csv)
        if len(cols)!=0:
            df=df.loc[:,cols]
        dn2df[label]=df        
    if not to_dict:
        return delunnamedcol(pd.concat(dn2df,names=[collabel,'Unnamed'],axis=axis,**params_concat).reset_index())
    else:
        return dn2df

## save table
def to_table(df,p):
#     from rohan.dandage.io_strs import make_pathable_string
#     p=make_pathable_string(p)
    if not 'My Drive' in p:
        p=p.replace(' ','_')
    else:
        logging.warning('probably working on google drive; space/s left in the path.')
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    if p.endswith('.tsv') or p.endswith('.tab'):
        df.to_csv(p,sep='\t')
        if is_interactive_notebook():
            print(p)
    elif p.endswith('.pqt') or p.endswith('.parquet'):
        to_table_pqt(df,p)
        if is_interactive_notebook():
            print(p)
    else: 
        logging.error(f'unknown extension {p}')
def to_table_pqt(df,p):
    if len(df.index.names)>1:
        df=df.reset_index()    
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    df.to_parquet(p,engine='fastparquet',compression='gzip',)

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
    to be deprecated
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
    to be deprecated
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

## reshape df

def dmap2lin(df,idxn='index',coln='column',colvalue_name='value'):
    """
    dmap: ids in index and columns 
    idxn,coln of dmap -> 1st and 2nd col
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
     
def df2unstack(df,coln='columns',idxn='index',col='value'):
    """
    will be deprecated
    """
    return dmap2lin(df,idxn=idxn,coln=coln,colvalue_name=col)


def pivot_table_str(df,index,columns,values):
    def apply_(x):
        zx=list()
        if len(x)>1:
            logging.warning('more than 1 str value encountered, returning list')
            rerturn x
        else:
            return x[0]
    return df.pivot_table(index=index,columns=columns,values=values,aggfunc=apply_)

## paired dfs
from rohan.dandage.io_strs import replacelist
def unpair_df(df,cols_df1,cols_df2,cols_common,replace_suffix):
    dfs=[]
    for cols in [cols_df1,cols_df2]:
        df_=df[cols+cols_common]
        df_=df_.rename(columns=dict(zip(df_.columns,replacelist(df_.columns,replace_suffix))))
        dfs.append(df_)
    dfout=dfs[0].append(dfs[1])
    return dfout

lin_dfpair=unpair_df

def merge_dfs_paired_with_unpaireds(dfpair,df,
                        left_ons=['gene1 name','gene2 name'],
                        right_on='gene name',right_ons_common=[],
                        suffixes=[' gene1',' gene2'],how='left',dryrun=False, test=False):
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
        print('> merge1_left_on');print(merge1_left_on)
        print('> merge1_right_on');print(merge1_right_on)
        print('> merge2_left_on');print(merge2_left_on)
        print('> merge2_right_on');print(merge2_right_on)
        print('> dfpair');print(dfpair.columns.tolist())
        print('> df1');print(df1.columns.tolist())
        print('> df2');print(df2.columns.tolist())
                    
    if not dryrun:
        dfpair_merge1=dfpair.merge(df1,
                        left_on=merge1_left_on,
                        right_on=merge1_right_on,
                        how=how)
        if test:
            print('> dfpair_merge1 columns'); print(dfpair_merge1.columns.tolist())
        dfpair_merge2=dfpair_merge1.merge(df2,
                    left_on=merge2_left_on,
                    right_on=merge2_right_on,
                    how=how)
        if test:
            print('> dfpair_merge2 columns');print(dfpair_merge2.columns.tolist())
        return dfpair_merge2
    
merge_dfpairwithdf=merge_dfs_paired_with_unpaireds

def column_suffixes2multiindex(df,suffixes,test=False):
    cols=[c for c in df if c.endswith(f' {suffixes[0]}') or c.endswith(f' {suffixes[1]}')]
    if test:
        print(cols)
    df=df.loc[:,cols]
    df=df.rename(columns={c: (s,c.replace(f' {s}','')) for s in suffixes for c in df if c.endswith(f' {s}')})
    df.columns=pd.MultiIndex.from_tuples(df.columns)
    return df

## chucking

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
def dfmap2symmcolidx(df,test=False):
    geneids=set(df.index).union(set(df.columns))
    df_symm=pd.DataFrame(columns=geneids,index=geneids)
    df_symm.loc[df.index,:]=df.loc[df.index,:]
    df_symm.loc[:,df.columns]=df.loc[:,df.columns]
    if test:
        logging.debug(df_symm.shape)
    return df_symm

def dflin2dfbinarymap(dflin,col1,col2,params_df2submap={'aggfunc':'sum','binary':True,'binaryby':'nan'},test=False):
    """
    if not binary:
        dropna the df by value [col index and value] column
    """
    # get the submap ready
    df_map=df2submap(df=dflin,
              col=col1,idx=col2,**params_df2submap)
    if test:           
        logging.debug(df_map.unstack().unique())
    # make columns and index symmetric
    df_map_symm=dfmap2symmcolidx(df_map)
    df_map_symm=df_map_symm.fillna(False)
    if test:           
        logging.debug(df_map_symm.unstack().unique())
    df_map_symm=(df_map_symm+df_map_symm.T)/2
    if test:           
        logging.debug(df_map_symm.unstack().unique())
    df_map_symm=df_map_symm!=0
    if test:           
        logging.debug(df_map_symm.unstack().unique())
    return df_map_symm

def symmetric_dflin(df,col1,col2,colval,sort=False,dropna=False):
    """
    col1 becomes columns
    col2 becomes index    
    """
    
    ds=df.set_index([col1,col2])[colval].unstack()
    for i in set(ds.index.tolist()).difference(ds.columns.tolist()):
        ds.loc[:,i]=np.nan
    for i in set(ds.columns.tolist()).difference(ds.index.tolist()):
        ds.loc[i,:]=np.nan
    arr=ds.values
    arr2=np.triu(arr) + np.triu(arr,1).T
    dmap=pd.DataFrame(arr2,columns=ds.columns,index=ds.columns)
    if sort:
        index=dmap.mean().sort_values(ascending=False).index
        dmap=dmap.loc[index,index]
    if dropna:
        print(dmap.shape,end='')
        dmap=dmap.dropna(axis=1,how='all').dropna(axis=0,how='all')
        print(dmap.shape)        
    return dmap


def filldiagonal(df,filler=None):
    if df.shape[0]!=df.shape[1]:
        logging.warning('df not symmetric')      
#     ids=set(df.columns).intersection(df.index)
    if filler is None:
        filler=np.nan
    if str(df.dtypes.unique()[0])=='bool' and (pd.isnull(filler)) :
        print('warning diagonal is replaced by True rather than nan')
    np.fill_diagonal(df.values, filler)        
    return df

def getdiagonalvals(df):
    if df.shape[0]!=df.shape[1]:
        logging.warning('df not symmetric')
#     ids=set(df.columns).intersection(df.index)
    ds=pd.Series(np.diag(df), index=[df.index, df.columns])
#     id2val={}
#     for i,c in zip(ids,ids):
#         id2val[i]=df.loc[i,c]
    return pd.DataFrame(ds,columns=['diagonal value']).reset_index()

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

# def get_offdiag_vals(dcorr):
#     """
#     for lin dcorr i guess
#     """
#     del_indexes=[]
#     for spc1 in np.unique(dcorr.index.get_level_values(0)):
#         for spc2 in np.unique(dcorr.index.get_level_values(0)):
#             if (not (spc1,spc2) in del_indexes) and (not (spc2,spc1) in del_indexes):
#                 del_indexes.append((spc1,spc2))
#     #         break
#     for spc1 in np.unique(dcorr.index.get_level_values(0)):
#         for spc2 in np.unique(dcorr.index.get_level_values(0)):
#             if spc1==spc2:
#                 del_indexes.append((spc1,spc2))
    
#     return dcorr.drop(del_indexes)


def make_symmetric_across_diagonal(df,fill='lower'):
    for c1i,c1 in enumerate(df.columns):
        for c2i,c2 in enumerate(df.columns):
            if c1i>c2i:
                if fill=='lower': 
                    df.loc[c1,c2]=df.loc[c2,c1]
                elif fill=='upper':
                    df.loc[c2,c1]=df.loc[c1,c2]                
    return df

# aggregate dataframes

def get_group(groups,i=0):return groups.get_group(list(groups.groups.keys())[i])

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

## df stats
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

def percentiles(ds):
    return [(f"{per:.2f}",ds.quantile(per)) for per in np.arange(0,1.1,0.1)]        

## dedup

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
def coltuples2str(cols,sep=' '):
    from rohan.dandage.io_strs import tuple2str
    cols_str=[]
    for col in cols:
        cols_str.append(tuple2str(col,sep=sep))
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
# def pd_merge_dfwithobjcols(df1,df2,left_on=None,right_on=None,on=None,
#                           how='inner'):
#     if not on is None:
#         left_on=on
#         right_on=on
#     df1=set_index(df1,left_on)
#     df2=set_index(df2,right_on)
#     df=pd.concat([df1,df2], axis=1, join=how)
#     return df.reset_index()

# import logging
def merge_dfs(dfs,how='left',suffixes=['','_'],
              test=False,fast=False,
              **params_merge):
    from rohan.dandage.io_sets import list2intersection,flatten
    from rohan.dandage.io_dict import sort_dict
    if all([isinstance(df,str) for df in dfs]):
        dfs=[read_table(p) for p in dfs]
    if not 'on' in params_merge:
        params_merge['on']=list(list2intersection([df.columns for df in dfs]))
        if len(params_merge['on'])==0:
            logging.error('no common columns found for infer params_merge[on]')
            return
    params_merge['how']=how
    params_merge['suffixes']=suffixes
    # sort largest first
    if test:
        print(params_merge)
        print('size',{dfi:[len(df)] for dfi,df in enumerate(dfs)})
    dfi2cols_value={dfi:df.select_dtypes([int,float]).columns.tolist() for dfi,df in enumerate(dfs)}
    cols_common=list(np.unique(params_merge['on']+list(list2intersection(dfi2cols_value.values()))))
    dfi2cols_value={k:list(set(dfi2cols_value[k]).difference(cols_common)) for k in dfi2cols_value}
    if test:
        print('cols_common',cols_common)
        print('dfi2cols_value',dfi2cols_value)
    dfs=[drop_duplicates_by_agg(dfs[dfi],cols_common,dfi2cols_value[dfi],fast=fast) for dfi in dfi2cols_value]
    print('size agg',{dfi:[len(df)] for dfi,df in enumerate(dfs)})
    sorted_indices_by_size=sort_dict({dfi:[len(df.drop_duplicates(params_merge['on']))] for dfi,df in enumerate(dfs)},0)
    print('size dedup',sorted_indices_by_size)
    sorted_indices_by_size=list(sorted_indices_by_size.keys())[::-1]
    dfs=[dfs[i] for i in sorted_indices_by_size]
    for dfi,df in enumerate(dfs):
        if dfi==0:
            df1=df.copy()
        else:
            if test:
                print(df1.columns)
                print(df.columns)
            df1=pd.merge(df1, df, **params_merge)
        print(dfi,':',df1.shape,'; ',end='')
    print('')
    cols_std=[f"{c} var" for c in flatten(list(dfi2cols_value.values()))]
    cols_del=[c for c in cols_std if df1[c].isnull().all()]
    df1=df1.drop(cols_del,axis=1)
    return df1

# def merge_dn2df(dn2df,on,how='left',
#                test=False):
#     dn2dflen=dict(zip([len(dn2df[dn].drop_duplicates(subset=on)) for dn in dn2df.keys()],dn2df.keys()))
#     if test:
#         print(dn2dflen)
#     for dni,dflen in enumerate(sorted(dn2dflen,reverse=True)):
#         dn=dn2dflen[dflen]
#         df=dn2df[dn]
#         df_ddup=df.drop_duplicates(subset=on)
#         if len(df)!=len(df_ddup):
#             df=df_ddup.copy()
#             logging.warning(f'{dn}: dropped duplicates. size drop from {len(df)} to {len(df_ddup)}')
#         if dni==0:
#             cols=[c for c in df.columns.tolist() if (not ((c in on) or (c==on))) and (c in df)]
#             df=df.rename(columns=dict(zip(cols,[f"{c} {dn}" for c in cols])))
#             dfmerged=df.copy()
#         else:
#             cols=[c for c in df.columns.tolist() if (not ((c in on) or (c==on))) and (c in df)]
#             if test:
#                 print(dn,cols)
#                 print(dict(zip(cols,[f"{c} {dn}" for c in cols])))
#             df=df.rename(columns=dict(zip(cols,[f"{c} {dn}" for c in cols])))
#             dfmerged=dfmerged.merge(df,on=on,how=how,
# #                                     suffixes=['',f" {dn}"],
#                                    )
#             if test:
#                 print(f" {dn}",dfmerged.columns.tolist(),df.columns.tolist())
#         del df
#     return dfmerged


def split_rows(df,collist,rowsep=None):
    """
    for merging dfs with names with df with synonymns
    param colsyn: col containing tuples of synonymns 
    """
    if not rowsep is None:
        df.loc[:,colsyn]=df.loc[:,colsyn].apply(lambda x : x.split(rowsep))
    return dellevelcol(df.set_index([c for c in df if c!=collist])[collist].apply(pd.Series).stack().reset_index().rename(columns={0:collist}))        
#     #make duplicated row for each synonymn
#     dfsynappended=df[colsyn].apply(pd.Series).unstack().reset_index().drop('level_0',axis=1).set_index('level_1')
#     dfsynappended.columns=[colsynappended]
#     dfsynappended=dfsynappended.dropna()
#     return dfsynappended.join(df,how='left')

dfsyn2appended=split_rows

# def split_lists(ds):
#     """
#     """
#     return dmap2lin(ds.apply(pd.Series),colvalue_name=ds.name).drop(['column'],axis=1).set_index(ds.index.names).dropna()

def meltlistvalues(df,value_vars,colsynfmt='str',colsynstrsep=';'):
    return dfsyn2appended(df,colsyn,colsynfmt=colsynfmt,colsynstrsep=colsynstrsep)
## drop duplicates by aggregating the dups
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
#     print(cols_groupby)
#     print(col2aggfunc)
#     print(df.columns.tolist())
    df1=getattr(df.groupby(cols_groupby),f"{'progress' if not fast else 'parallel'}_apply")(lambda x: agg(x,col2aggfunc))
#     df1.columns=[c.replace(f' {aggfunc}','') for c in coltuples2str(df1.columns)]
    return df1.reset_index()
# def drop_duplicates_agg(df,colsgroupby,cols2aggf,test=False):
#     """
#     colsgroupby: unique names ~index
#     cols2aggf: rest of the cols `unique_dropna_str` for categories
#     """
#     if test:
#         print(df.shape)
#         print(df.drop_duplicates(subset=colsgroupby).shape)
#     #ddup aggregated
#     dfdupagg=df.loc[(df.duplicated(subset=colsgroupby,keep=False)),:].groupby(colsgroupby).agg(cols2aggf)
#     #drop duplicates all
#     df_=df.drop_duplicates(subset=colsgroupby,keep=False)
#     if test:
#         print(df_.shape)
#     #append ddup aggregated
#     dfout=df_.append(dfdupagg,sort=True)
#     if test:
#         print(dfout.shape)
#     return dfout

## sorting

def dfsortbybins(df, col):
    d=dict(zip(bins,[float(s.split(',')[0].split('(')[1]) for s in bins]))
    df[f'{col} dfrankbybins']=df.apply(lambda x : d[x[col]] if not pd.isnull(x[col]) else x[col], axis=1)
    df=df.sort_values(f'{col} dfrankbybins').drop(f'{col} dfrankbybins',axis=1)
    return df

def sort_col_by_list(df, col,l):
    df[col]=pd.Categorical(df[col],categories=l, ordered=True)
    return df.sort_values(col)
def sort_binnnedcol(df,col):
    df[f'_{col}']=df[col].apply(lambda s : float(s.split('(')[1].split(',')[0]))
    df=df.sort_values(by=f'_{col}')
    df=df.drop([f'_{col}'],axis=1)
    return df

def groupby_sort(df,col_groupby,col_sortby,func='mean',ascending=True):
    df1=df.groupby(col_groupby).agg({col_sortby:getattr(np,func)}).reset_index()
    df2=df.merge(df1,
            on=col_groupby,how='inner',suffixes=['',f' per {col_groupby}'])
    return df2.sort_values(f'{col_sortby} per {col_groupby}',ascending=ascending)
     
def sorted_column_pair(x,colvalue,suffixes,categories=None,how='all',
                                    test=False):
    """
    Checks if values in pair of columns is sorted.
    
    Numbers are sorted in ascending order.
    
    :returns : True if sorted else 
    """
    if categories is None:
        if x[f'{colvalue} {suffixes[0]}'] < x[f'{colvalue} {suffixes[1]}']:
            return True
        else:
            return False            
    else:
        if test:
            print([x[f'{colvalue} {suffixes[0]}'],x[f'{colvalue} {suffixes[1]}']],
              categories,
              getattr(np,how)([x[f'{colvalue} {suffixes[0]}']==categories[0],
                            x[f'{colvalue} {suffixes[1]}']==categories[1]]),
              getattr(np,how)([x[f'{colvalue} {suffixes[0]}']==categories[1],
                            x[f'{colvalue} {suffixes[1]}']==categories[0]]))
        if getattr(np,how)([x[f'{colvalue} {suffixes[0]}']==categories[0],
                            x[f'{colvalue} {suffixes[1]}']==categories[1]]):
            return True
        elif getattr(np,how)([x[f'{colvalue} {suffixes[0]}']==categories[1],
                              x[f'{colvalue} {suffixes[1]}']==categories[0]]):
            return False
        else:
            return np.nan        
        
def sort_column_pair(df,colvalue,suffixes,categories=None,how='all',test=False,fast=True): 
    suffix2cols={s:sorted(df.filter(like=s).columns.tolist()) for s in suffixes}
    if len(suffix2cols[suffixes[0]])!=len(suffix2cols[suffixes[1]]):
        logging.error("df should contain paired columns")
        print(suffix2cols)
        return 
    from rohan.dandage.io_sets import list2intersection
    if len(list2intersection(list(suffix2cols.values())))!=0:
        logging.error("df should contain non-overlapping paired columns")
        print(suffix2cols)
        return 
    df['sorted']=getattr(df,'parallel_apply' if fast else 'apply')(lambda x: sorted_column_pair(x,colvalue=colvalue,suffixes=suffixes,
                                                                                                                       categories=categories,how=how,test=test),axis=1)
    if test:
        print(df.shape,end=' ')
    df=df.dropna(subset=['sorted'])
    df['sorted']=df['sorted'].astype(bool)
    if test:
        print(df.shape)
    df1,df2=df.loc[df['sorted'],:],df.loc[~df['sorted'],:]
    # rename cols of df2 (not sorted) 
    rename=dict(zip(suffix2cols[suffixes[0]]+suffix2cols[suffixes[1]],
             suffix2cols[suffixes[1]]+suffix2cols[suffixes[0]]))
    if test:
        print(rename)
    df2=df2.rename(columns=rename)
    df3=df1.append(df2)
    if test:
        print(df1.shape,df2.shape,df3.shape)
    return df3.drop(['sorted'],axis=1)


def is_col_numeric(ds):
    return np.issubdtype(ds.dtype, np.number)

# 2d subsets
def subset_cols_by_cutoffs(df,col2cutoffs,quantile=False,outdf=False,test=False):    
    for col in col2cutoffs:
        if isinstance(col2cutoffs[col],float):
            cutoffs=[col2cutoffs[col],1-col2cutoffs[col]]
            if quantile:
                col2cutoffs[col]=[df[col].quantile(c) for c in cutoffs]
            else:
                col2cutoffs[col]=cutoffs
        elif not isinstance(col2cutoffs[col],list):
            logging.error("cutoff should be float or list")
        df[f"{col} (high or low)"]=df[col].apply(lambda x: 'low' if x<col2cutoffs[col][0] else 'high' if x>col2cutoffs[col][1] else np.nan)
    colout=f"{'-'.join(list(col2cutoffs.keys()))} (low or high)"
    def get_subsetname(x):
        l=[x[f"{col} (high or low)"] for col in col2cutoffs]
        if not any([pd.isnull(i) for i in l]):
            return '-'.join(l)
        else:
            return np.nan
    df[colout]=df.apply(lambda x:  get_subsetname(x),axis=1)
    if test:
        print(col2cutoffs)
        if len(col2cutoffs.keys())==2:
            element2color={'high-high':'r',
                   'high-low':'g',
                   'low-high':'b',
                   'low-low':'k',
                  }
            import matplotlib.pyplot as plt
            ax=plt.subplot()
            df.groupby(colout).apply(lambda x: x.plot.scatter(x=list(col2cutoffs.keys())[0],
                                                              y=list(col2cutoffs.keys())[0],
                                                              alpha=1,ax=ax,label=x.name,color=element2color[x.name]))
    if not outdf:
        return df[colout]
    else:
        return df
    
from rohan.dandage.io_sets import dropna
def get_intersectionsbysubsets(df,cols_fracby2vals,cols_subset,col_ids,params_qcut={'q':10,'duplicates':'drop'}):
    """
    cols_fracby:
    cols_subset:
    """
    for coli,col in enumerate(cols_subset):
        if is_col_numeric(df[col]):
            try:
                df[f"{col} bin"]=pd.qcut(df[col],**params_qcut)
            except:
                print(col)
            cols_subset[coli]=f"{col} bin"
    for col_fracby in cols_fracby2vals:
        val=cols_fracby2vals[col_fracby]
        ids=df.loc[(df[col_fracby]==val),col_ids].dropna().unique()
        for col_subset in cols_subset:
            for subset in dropna(df[col_subset].unique()):
                ids_subset=df.loc[(df[col_subset]==subset),col_ids].dropna().unique()
                df.loc[(df[col_subset]==subset),f'P {col_fracby} {col_subset}']=len(set(ids_subset).intersection(ids))/len(ids_subset)
    return df
#filter df
def filter_rows_bydict(df,d,sign='==',logic='and',test=False):
    qry = f' {logic} '.join([f"`{k}` {sign} '{v}'" for k,v in d.items()])
    df1=df.query(qry)
    if test:
        print(df1.loc[:,list(d.keys())].drop_duplicates())
    if len(df1)==0:
        logging.warning('may be some column names are wrong..')
        logging.warning([k for k in d if not k in df])
    return df1

# import from stat
from rohan.dandage.stat.transform import dflogcol

# filter with stats
import logging
def filterstats(df,boolcol):
    df_=df.loc[boolcol,:]
    logging.info(df,df_)
    return df_

# get numbers from annot 
def get_colsubset2stats(dannot,colssubset=None):
    if colssubset is None:
        colssubset=dannot_stats.columns
    dannot_stats=dannot.loc[:,colssubset].apply(pd.Series.value_counts)

    colsubset2classes=dannot_stats.apply(lambda x: x.index,axis=0)[dannot_stats.apply(lambda x: ~pd.isnull(x),axis=0)].apply(lambda x: dropna(x),axis=0).to_dict()
    colsubset2classns=dannot_stats[dannot_stats.apply(lambda x: ~pd.isnull(x),axis=0)].apply(lambda x: dropna(x),axis=0).to_dict()
    colsubset2classns={k:[int(i) for i in colsubset2classns[k]] for k in colsubset2classns}
    return dannot_stats,colsubset2classes,colsubset2classns

def append_similar_cols(df,suffixes=None,prefixes=None,ffixes=None,test=False):
    import re
    if prefixes is None and suffixes is None and ffixes is None:
        logging.error('provide either prefixes or suffixes or ffixes')
    dn2df={}
    for f in (suffixes if not suffixes is None else prefixes if not prefixes is None else ffixes): 
        reg=f"{'$' if suffixes is None else '.*'}{f}{'$' if prefixes is None else '.*'}"
        df_=df.filter(regex=reg,axis=1)
        dn2df[f]=df_.rename(columns={c:c.replace(f,'') if not c==f else 'ffix' for c in df_})
    if test:
        print({k:dn2df[k].columns.tolist() for k in dn2df})
    return pd.concat(dn2df,axis=0)


def dict2df(d,colkey='key',colvalue='value'):
    return pd.DataFrame(pd.concat({k:pd.Series(d[k]) for k in d})).droplevel(1).reset_index().rename(columns={'index':colkey,0:colvalue})


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