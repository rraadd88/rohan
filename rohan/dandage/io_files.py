import pandas as pd

# paths
from glob import glob,iglob
import os
from os import makedirs
from os.path import exists,basename,dirname,abspath,realpath


from shutil import copyfile
# def copy(src, dst):copyfile(src, dst)
# def cp(src, dst):copy(src, dst)

from os.path import splitext,basename
def basenamenoext(p): return splitext(basename(p))[0]


## text files
from rohan.dandage.io_strs import getall_fillers    
def fill_form(df,templatep,template_insert_line,outp,splitini,splitend,field2replace,
           test=False):
    """
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
    makedirs(dirname(outp),exist_ok=True)
    with open(outp, 'w') as outfile:
        for p in ps:
            with open(p) as infile:
                outfile.write(infile.read())    

def get_encoding(p):
    import chardet
    with open(p, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']                

import shutil
def zip_folder(source, destination):
    #https://stackoverflow.com/a/50381250/3521099
    base = os.path.basename(destination)
    name = base.split('.')[0]
    fmt = base.split('.')[1]
    archive_from = os.path.dirname(source)
    archive_to = os.path.basename(source.strip(os.sep))
    shutil.make_archive(name, fmt, archive_from, archive_to)
    shutil.move(f'{name}.{fmt}', destination)
    
def backup_to_zip(ps,destp):
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
        for p_ in ps_:
            if exists(p_):
                p_dest=f"{destdp}/{p_.replace(get_common_preffix(ps_),'')}"
                makedirs(dirname(p_dest),exist_ok=True)
                copyfile(p_,p_dest)
    zip_folder(destdp, destp)    