import subprocess
from os.path import exists
import os
import getpass
print('user=',getpass.getuser())
if not getpass.getuser()=='rohan':
    prjd='/content'
    if not exists(f'{prjd}/rohan'):
        subprocess.call('git clone https://github.com/rraadd88/rohan.git;cd rohan;pip install -e .',shell=True)
    else:
        os.system('cd {prjd}/rohan;git pull')
    import sys
    sys.path.append('rohan/rohan/dandage')
    sys.path.append('rohan/rohan')
    sys.path.append('rohan')    
else:
    prjd='content'
###
import pandas as pd
#dict
from collections import OrderedDict
import numpy as np
# plots
import matplotlib.pyplot as plt
# plt.style.use('default')
import seaborn as sns
# sns.set('notebook',font='Monaco')
# stats
import scipy as sc
# paths
from glob import glob
from os import makedirs
from os.path import exists,basename,dirname,abspath
# strings
import re
# recepies
from rohan.dandage.io_dfs import *
from rohan.dandage.io_sets import *
from rohan.dandage.io_strs import make_pathable_string
from rohan.dandage.plot.annot import *
## plotting
from rohan.dandage.plot.colors import get_cmap_subset
from rohan.dandage.plot.ax_ import grid
# debug
import logging
###
import subprocess
def dlfromgdrive(link,p,):
    makedirs(dirname(p),exist_ok=True)
    if not 'spreadsheets' in link:
        id=link.split('id=')[1]
        com=f"wget --no-check-certificate 'https://docs.google.com/uc?export=download&id={id}' -O {p}"
    else: 
        id=link.split('/d/')[1].split('/')[0]
        com=f"wget --no-check-certificate 'https://docs.google.com/spreadsheets/d/{id}/export?gid=0&format=tsv' -O {p}"
#   https://docs.google.com/spreadsheets/d/0At2sqNEgxTf3dEt5SXBTemZZM1gzQy1vLVFNRnludHc/export?gid=0&format=csv'
    print(com)
    subprocess.call(com,shell=True)
def vname2df(vname,force=False):
    fp=f"data/{vname}.{dinfo.loc[vname,'file format']}"
    if not exists(fp) or force:
        dlfromgdrive(dinfo.loc[vname,'link to table'],fp)
    if fp.endswith('.pqt'):
        return read_table_pqt(fp)
    elif fp.endswith('.tsv'):
        return read_table(fp)     
if not getpass.getuser()=='rohan':
    import warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    ###    
    try:
        import fastparquet
    except:
        os.system('pip install fastparquet',)
    os.system('apt install tree')
    # %matplotlib inline
    from google.colab import drive
    drive.mount('{prjd}/gdrive',force_remount=True)
    # os.system('sed -i \'s/ggplot/fast/\' /content/rohan/rohan/global_imports.py')
    # os.system('sed -i \'s/Monaco/Arial/\' /content/rohan/rohan/global_imports.py')  
    # from rohan.global_imports import *
    dlfromgdrive('https://docs.google.com/spreadsheets/d/id/edit?usp=sharing',
                 f'{prjd}/data/dinfo.tsv')
    dinfo=read_table(f'{prjd}/data/dinfo.tsv').dropna(subset=['variable name']).set_index('variable name')
    # print(dinfo.index.tolist())
    dinfo.loc[:,['link to table','file format']]    
