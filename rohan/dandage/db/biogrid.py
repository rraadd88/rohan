import pandas as pd
import numpy as np
from os.path import basename,dirname,exists
from rohan.dandage.io_dfs import *

from rohan.dandage.db.intact import get_degrees
import logging

def dintmap2lin(dint_db):
#     dint_db=read_table('database/biogrid/559292/physical/all/dintmap_symm.pqt')
    dint_db=dmap2lin(dint_db,idxn='gene id gene name interactor1',coln='gene id gene name interactor2',colvalue_name='interaction score db').replace(0,np.nan).dropna()
    dint_db['interaction id']=dint_db.apply(lambda x : '--'.join(sorted([x['gene id gene name interactor1'],x['gene id gene name interactor2']])),axis=1)
    dint_db=dint_db.drop_duplicates(subset=['interaction id'])
    dint_db['interaction bool db']=True
    return dint_db

def get_dint(taxid,dint_rawp,
                        outd=None,
                        dintmap_symmp=None,dintlinp=None,dintmapp=None,
                        logf=None,
                        experimental_system_type='physical',
                        genefmt=None,
                        force=False,test=False,
                        keep_exp_syss=[],del_exp_syss=[],
                        filldiagonal_withna=False):
    """
    taxid=559292
    del_exp_syss=["Co-purification", "Co-fractionation", "Proximity Label-MS", 
                                  "Affinity Capture-RNA", "Protein-peptide"]
    """    
    if not genefmt is None:
        logging.warning('genefmt is deprecated, both gene ids and gene names will be used')
    if not outd is None:
        dintp=f'{outd}/dint.pqt'
        logf=f'{outd}/log.tsv'
    if dint_rawp.endswith('tab2.txt') and not exists(dint_rawp+".pqt"):
        print(f"converting txt to parquet")
        dint=pd.read_csv(dint_rawp,
           sep='\t',low_memory=False)
        dint_rawp=dint_rawp+".pqt"
        print(f"use {dint_rawp}")
    if (not exists(dintp)) or force:
        dint=read_table(dint_rawp)
        # filter biogrid
        # taxonomic id of Scer
        print(dint.shape,end='')
        dint=dint.loc[((dint['Experimental System Type']==experimental_system_type) \
                   & (dint['Organism Interactor A']==taxid) \
                   & (dint['Organism Interactor B']==taxid)),:]
        print(dint.shape)
        
        # filter by expt sys
        if test:
            dlog=pd.DataFrame(dint['Experimental System'].value_counts())
            print(dlog)
            if not logf is None:
                to_table(dlog,f'{logf}.all_int.tsv')                    
        if len(keep_exp_syss)!=0:
            print(dint.shape,end='')    
            dint=dint.loc[((dint['Experimental System'].isin(keep_exp_syss))),:]
            print(dint.shape)            
        elif len(del_exp_syss)!=0:
            print(dint.shape,end='')                
            dint=dint.loc[((~dint['Experimental System'].isin(del_exp_syss))),:]
            print(dint.shape)            
        elif len(del_exp_syss)!=0 and len(keep_exp_syss)!=0:
            logging.error('Error: either specify keep_exp_syss or del_exp_syss')
            return None
        if test:
            print('Experimental Systems used.')
            dlog=pd.DataFrame(dint['Experimental System'].value_counts())
            if not logf is None:
                to_table(dlog,f'{logf}.kept_int.tsv')
            print(dlog)
        to_table(dint,dintlinp)
        for inti in ['A','B']:
            dint[f'gene id gene name interactor {inti}']=dint.apply(lambda x : ' '.join(list([x[f'Systematic Name Interactor {inti}'],x[f'Official Symbol Interactor {inti}']])),axis=1)
    else:
        dint=read_table_pqt(dintlinp)