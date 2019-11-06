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

def get_dintmap(taxid,dbiogridp,
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
        dintlinp=f'{outd}/dintlin.pqt'
        dintmap_symmp=f'{outd}/dintmap_symm.pqt'
        dintmapp=f'{outd}/dintmap.pqt'
        logf=f'{outd}/log.tsv'
    if (not exists(dintmap_symmp)) or force:
        if not exists(dintmapp) or force:
            if not exists(dintlinp) or force:
                if dbiogridp.endswith('tab2.txt'):
                    print(f"converting to parquet")
                    dbiogrid=pd.read_csv(dbiogridp,
                       sep='\t',low_memory=False)
                    dbiogridp=dbiogridp+".pqt"
                    to_table_pqt(dbiogrid,dbiogridp)
                    print(f"use {dbiogridp}")
                elif dbiogridp.endswith('tab2.txt.pqt'):
                    dbiogrid=read_table_pqt(dbiogridp)
                # filter biogrid
                # taxonomic id of Scer
                dbiogrid=dbiogrid.loc[((dbiogrid['Experimental System Type']==experimental_system_type) \
                           & (dbiogrid['Organism Interactor A']==taxid) \
                           & (dbiogrid['Organism Interactor B']==taxid)),:]

                print('All physical Experimental Systems.')
                if test:
                    dlog=pd.DataFrame(dbiogrid['Experimental System'].value_counts())
                    print(dlog)
                    if not logf is None:
                        to_table(dlog,f'{logf}.all_int.tsv')                    
                if len(keep_exp_syss)!=0:
                    dbiogrid=dbiogrid.loc[((dbiogrid['Experimental System'].isin(keep_exp_syss))),:]
                elif len(del_exp_syss)!=0:
                    dbiogrid=dbiogrid.loc[((~dbiogrid['Experimental System'].isin(del_exp_syss))),:]
                elif len(del_exp_syss)!=0 and len(keep_exp_syss)!=0:
                    print('Error: either specify keep_exp_syss or del_exp_syss')
                    return False
                if test:
                    print('Experimental Systems used.')
                    dlog=pd.DataFrame(dbiogrid['Experimental System'].value_counts())
                    if not logf is None:
                        to_table(dlog,f'{logf}.kept_int.tsv')
                    print(dlog)
                to_table_pqt(dbiogrid,dintlinp)
            else:
                dbiogrid=read_table_pqt(dintlinp)
            # this cell makes a symmetric interaction map
            dbiogrid['count']=1
            if test:
                print(dbiogrid['count'].sum())
            # mean of counts of intearations
            ## higher is more confident interaction captured in multiple assays
            for inti in ['A','B']:
                dbiogrid[f'gene id gene name Interactor {inti}']=dbiogrid.apply(lambda x : ' '.join(list([x[f'Systematic Name Interactor {inti}'],x[f'Official Symbol Interactor {inti}']])),axis=1)
            dbiogrid_grid=dbiogrid.pivot_table(values='count',
                                             index="gene id gene name Interactor A",
                                            columns="gene id gene name Interactor B",
                                            aggfunc='sum',)

            # make it symmetric
            if test:
                print('shape of non-symm intmap: ',dbiogrid_grid.shape)
            to_table_pqt(dbiogrid_grid,dintmapp)
        else:         
#             dbiogrid_grid=pd.read_table(dintmapp)
            dbiogrid_grid=read_table_pqt(dintmapp+'.pqt')
        dbiogrid_grid=set_index(dbiogrid_grid,"gene id gene name Interactor A")
        geneids=set(dbiogrid_grid.index).union(set(dbiogrid_grid.columns))
        if test:
            print('total number of genes',len(geneids))
        # missing rows to nan
        dintmap_symm=pd.DataFrame(columns=geneids,index=geneids)

        dintmap_symm.loc[dbiogrid_grid.index,:]=dbiogrid_grid.loc[dbiogrid_grid.index,:]
        dintmap_symm.loc[:,dbiogrid_grid.columns]=dbiogrid_grid.loc[:,dbiogrid_grid.columns]
        if test:
            print(dintmap_symm.shape)
        dintmap_symm=dintmap_symm.fillna(0)
        dintmap_symm=(dintmap_symm+dintmap_symm.T)/2
        dintmap_symm.index.name='Interactor A'
        dintmap_symm.columns.name='Interactor B'
#         if test:
#             dintmap_symm=dintmap_symm.iloc[:5,:5]
        if filldiagonal_withna:
            dintmap_symm=filldiagonal(dintmap_symm)
        to_table_pqt(dintmap_symm,dintmap_symmp)
        print('file saved at: ',dintmap_symmp)
        ddegnonself,ddegself=get_degrees(dintmap_symm)
        print(ddegself.head())
        dintmap_symm_lin=ddegself.copy()
        dintmap_symm_lin.index.name=f'gene {genefmt}'
        to_table(dintmap_symm_lin,f'{dintmap_symmp}.degrees.tsv')
        dintlin=dintmap2lin(dintmap_symm)
        to_table(dintlin,f'{dirname(dintmap_symmp)}/dintlin.pqt')
    else:
        dintmap_symm=read_table_pqt(dintmap_symmp)
    return dintmap_symm