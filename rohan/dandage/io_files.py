# paths
from glob import glob,iglob
import os
from os import makedirs
from os.path import exists,basename,dirname,abspath,realpath

from rohan.dandage.io_dfs import *
from rohan.dandage.io_sys import is_interactive_notebook
from rohan.dandage.io_strs import make_pathable_string,replacemany

from shutil import copyfile
import logging
# def copy(src, dst):copyfile(src, dst)
# def cp(src, dst):copy(src, dst)

# walker
def get_all_subpaths(d='.',include_directories=False): 
    """
    Get all the subpaths (folders and files) from a path.
    """
    from glob import glob
    import os
    paths=[]
    for root, dirs, files in os.walk(d):
        if include_directories:
            for d in dirs:
                path=os.path.relpath(os.path.join(root, d), ".")        
                paths.append(path)
        for f in files:
            path=os.path.relpath(os.path.join(root, f), d)
            paths.append(path)
    paths=sorted(paths)
    return paths

from os.path import splitext,basename
def basenamenoext(p): return splitext(basename(p))[0]


## text files
from rohan.dandage.io_strs import getall_fillers    
def fill_form(df,templatep,template_insert_line,outp,splitini,splitend,field2replace,
           test=False):
    """
    Fill a table interactively.
    """
    template=open(templatep, 'r').read()
    fillers=getall_fillers(template_insert_line,
                  leftoff=-1,rightoff=1)
    if test:
        print(fillers)
    insert=''
    for index in df.index:
        for filleri,filler in enumerate(fillers):
            if test:
                print(filler,str(df.loc[index,filler.replace('{','').replace('}','')]))
            if filleri==0:
                line=template_insert_line
            line=line.replace(filler,str(df.loc[index,filler.replace('{','').replace('}','')]))
        insert=insert+line
    output=template.split(splitini)[0]+splitini+insert+splitend+template.split(splitend)[-1]
    for field in field2replace:
        output=output.replace(field,field2replace[field])
    if test:        
        print(output)
    with open(outp,'w') as f:
        f.write(output)
#     return True

def cat(ps,outp):
    """
    Concatenate text files.
    """
    makedirs(dirname(outp),exist_ok=True)
    with open(outp, 'w') as outfile:
        for p in ps:
            with open(p) as infile:
                outfile.write(infile.read())    

def get_encoding(p):
    """
    Get encoding of a file.
    """
    import chardet
    with open(p, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']                

import shutil
def zip_folder(source, destination):
    """
    Zip a folder.
    """
    #https://stackoverflow.com/a/50381250/3521099
    base = os.path.basename(destination)
    name = base.split('.')[0]
    fmt = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, fmt, archive_from, archive_to)
    shutil.move(f'{name}.{fmt}', destination)
    
def backup_to_zip(ps,destp,test=False):
    """
    Zip files for backup.
    """
    if not destp.endswith('.zip'):
        loggin.error('arg destp should have .zip extension')
        return 0
    
    from rohan.dandage.io_strs import get_common_preffix
    if '/' in destp: 
        destdp=destp.split('.')[0]
        makedirs(destdp,exist_ok=True)
    else:
        destdp='./'
    for p in ps:
        if '*' in p:
            ps_=list(iglob(p))
        else:
            ps_=[p]
        print(ps_)
        for p_ in ps_:
            if exists(p_):
                p_dest=f"{destdp}/{p_.replace(get_common_preffix(ps_),'')}"
                makedirs(dirname(p_dest),exist_ok=True)
                print(f"cp {p_} {p_dest}")
                copyfile(p_,p_dest)
    zip_folder(destdp, destp)    
    
def read_url(url):
    """
    Read text from a url.
    """
    from urllib.request import urlopen
    f = urlopen(url)
    myfile = f.read()
    return str(myfile)

def download(url,path=None,outd='data/database',
             force=False):
    if path is None:
        path=replacemany(url,
               {'https://':'',
                'http://':'',
               })
        path=f"{outd}/{path}"
    if not exists(path) or force:
        import urllib.request
        makedirs(dirname(path),exist_ok=True)
        urllib.request.urlretrieve(url, path)
    else:
        logging.info(f"exists: {path}")
    return path

from rohan.dandage.io_sys import p2time

## dfs
from rohan.global_imports import rd

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

from rohan.dandage.io_text import get_header
def read_table(p,
               params={},
#                params_read_csv={}, # deprecate
               ext=None,
               test=False,
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
    ## deprecate params_read_csv
#     if len(params_read_csv.keys())!=0:
#         params=params_read_csv.copy()
    if isinstance(p,list) or '*' in p:
        if '*' in p:
            ps=glob(p)
            if exists(p.replace('/*','')):
                logging.warning(f"exists: {p.replace('/*','')}")
        if isinstance(p,list):
            ps=p
        return read_manytables(ps,params=params,
                               **kws_manytables)
    if len(params.keys())!=0 and not 'columns' in params:
        return pd.read_csv(p,**params).rd.clean()
    else:
        if len(params.keys())==0:
            params={}
        if ext is None:
            ext=basename(p).split('.',1)[1]
            
        if any([s==ext for s in ['pqt','parquet']]):#p.endswith('.pqt') or p.endswith('.parquet'):
            return pd.read_parquet(p,engine='fastparquet',**params).rd.clean()
        
        params['compression']='gzip' if ext.endswith('.gz') else 'zip' if ext.endswith('.zip') else None
        
        if not params['compression'] is None:
            ext=ext.split('.',1)[0]
            
        if any([s==ext for s in ['tsv','tab','txt']]):
            params['sep']='\t'
        elif any([s==ext for s in ['csv']]):
            params['sep']=','            
        elif ext=='vcf':
            from rohan.dandage.io_strs import replacemany
            params.update(dict(sep='\t',
                               comment='#',
                               header=None,
                               names=replacemany(get_header(path=p,comment='#',lineno=-1),['#','\n'],'').split('\t'),
                              ))
        elif ext=='gpad':
            params.update(dict(
                  sep='\t',
                  names=['DB','DB Object ID','Qualifier','GO ID','DB:Reference(s) (|DB:Reference)','Evidence Code','With (or) From',
                         'Interacting taxon ID','Date','Assigned by','Annotation Extension','Annotation Properties'],
                 comment='!',
                 ))
        else: 
            logging.error(f'unknown extension {ext} in {p}')
        if test: print(params)
        return pd.read_table(p,
                  **params,
                   ).rd.clean()
             
def read_ps(ps):
    if isinstance(ps,str) and '*' in ps:
        ps=glob(ps)
    return ps
def apply_on_paths(ps,func,
                   replaces_outp=None,
                   replaces_index=None,
                   drop_index=True,
                   colindex='path',
                   filter_rows=None,
                   fast=False, 
                   progress_bar=True,
                   params={},
                   dbug=False,
                   **kws,
                  ):
    """
    :params func:
    def apply_(p,outd='data/data_analysed',force=False):
        outp=f"{outd}/{basenamenoext(p)}.pqt'
        if exists(outp) and not force:
            return
        df01=read_table(p)
    apply_on_paths(
    ps=glob("data/data_analysed/*"),
    func=apply_,
    outd="data/data_analysed/",
    force=True,
    fast=False,
    read_path=True,
    )
    :params replaces_outp: to genrate outp
    :params replaces_index: to replace (e.g. dirname) in p
    :params colindex: column containing path 
    """
    def read_table_(df,read_path=False,
                   filter_rows=None,
                   func=None,
                    replaces_outp=None,
                    dbug=False,
                   ):
        p=df.iloc[0,:]['path']
        if read_path:
            if (not replaces_outp is None) and ('outp' in inspect.getfullargspec(func).args):
                outp=replacemany(p, replaces=replaces_outp, replacewith='', ignore=False)
                if dbug: ic(outp)
#                 if exists(outp):
#                     if 'force' in kws:
#                         if kws['force']:
#                             return None,None
#                 else:
                return p,outp
            else:
                return p,
        else:
            df=read_table(p,params=params,)
            if not filter_rows is None:
                df=df.rd.filter_rows(filter_rows)            
            return df,
    import inspect
    read_path=inspect.getfullargspec(func).args[0]=='p'
    if dbug: ic(read_path)
    if not replaces_index is None: drop_index=False
    ps=read_ps(ps)
    if len(ps)==0:
        logging.error('no paths found')
        return
    df1=pd.DataFrame({'path':ps})
    if fast and not progress_bar: progress_bar=True
    df2=getattr(df1.groupby('path',as_index=True),
                            f"{'parallel' if fast else 'progress'}_apply" if progress_bar else "apply"
               )(lambda df: func(*(read_table_(df,read_path=read_path,
                                                 func=func,
                                                 replaces_outp=replaces_outp,
                                                 filter_rows=filter_rows,
                                                 dbug=dbug)),
                                 **kws))
    df2=df2.rd.clean().reset_index(drop=drop_index).rd.clean()
    if not replaces_index is None:
        if isinstance(replaces_index,str):
            if replaces_index=='basenamenoext':
                replaces_index=basenamenoext
        if not isinstance(replaces_index,dict):
            ## function
            rename={p:replaces_index(p) for p in df2[colindex].unique()}
            df2[colindex]=df2[colindex].map(rename)
        else:
            ## dict
            df2[colindex]=df2[colindex].apply(lambda x: replacemany(x, replaces=replaces_index, replacewith='', ignore=False))
    return df2

def read_manytables(ps,
                    fast=False,
                    drop_index=True,
                    to_dict=False,
                    params={},
                    **kws,
                   ):
    """
    :params ps: list
    
    :TODO: info: creation dates of the newest and the oldest files.
    """       
    if not to_dict:
        df2=apply_on_paths(ps,func=lambda df: df,
                           fast=fast,drop_index=drop_index,
                           params=params,
                           **kws)
        return df2
    else:
        return {p:read_table(p,
                             params=params) for p in read_ps(ps)}
    
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
    if len(basename(p))>100:
        p=f"{dirname(p)}/{basename(p)[:95]}_{basename(p)[-4:]}"
        logging.warning(f"p shortened to {p}")
    if not df.index.name is None:
        df=df.reset_index()
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(dirname(p),exist_ok=True)
    if not colgroupby is None:
        to_manytables(df,p,colgroupby,**kws)
        return
    if p.endswith('.tsv') or p.endswith('.tab'):
        df.to_csv(p,sep='\t')
        if test:
            logging.info(p)
    elif p.endswith('.pqt'):
        to_table_pqt(df,p,**kws)
        if test:
            logging.info(p)
    else: 
        logging.error(f'unknown extension {p}')
        
def to_manytables(df,p,colgroupby,**kws_get_chunks):
    """
    :colvalue: if colgroupby=='chunk'
    """
    if colgroupby=='chunk':
        df[colgroupby]=get_chunks(df1=df,value='right',
                                  **kws_get_chunks)
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