import pandas as pd
import numpy as np
from os.path import basename,dirname,exists
from rohan.dandage.io_dfs import *

# def get_degrees(dintmap):
#     dintmap=filldiagonal(dintmap,0)
#     dintmap_binary=dintmap!=0
#     dints=dintmap_binary.sum()
#     dints.name='# of interactions'
#     dints=pd.DataFrame(dints)
#     return dints
from rohan.dandage.db.intact import get_degrees
import logging

def get_dbiogrid_intmap(taxid,dbiogridp,dbiogrid_intmap_symmp,dbiogrid_intlinp,dbiogrid_intmapp,
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
    if (not exists(dbiogrid_intmap_symmp)) or force:
        if not exists(dbiogrid_intmapp) or force:
            if not exists(dbiogrid_intlinp) or force:
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
                to_table_pqt(dbiogrid,dbiogrid_intlinp)
            else:
                dbiogrid=read_table_pqt(dbiogrid_intlinp)
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
            to_table_pqt(dbiogrid_grid,dbiogrid_intmapp)
        else:         
#             dbiogrid_grid=pd.read_table(dbiogrid_intmapp)
            dbiogrid_grid=read_table_pqt(dbiogrid_intmapp+'.pqt')
        dbiogrid_grid=set_index(dbiogrid_grid,"gene id gene name Interactor A")
        geneids=set(dbiogrid_grid.index).union(set(dbiogrid_grid.columns))
        if test:
            print('total number of genes',len(geneids))
        # missing rows to nan
        dbiogrid_intmap_symm=pd.DataFrame(columns=geneids,index=geneids)

        dbiogrid_intmap_symm.loc[dbiogrid_grid.index,:]=dbiogrid_grid.loc[dbiogrid_grid.index,:]
        dbiogrid_intmap_symm.loc[:,dbiogrid_grid.columns]=dbiogrid_grid.loc[:,dbiogrid_grid.columns]
        if test:
            print(dbiogrid_intmap_symm.shape)
        dbiogrid_intmap_symm=dbiogrid_intmap_symm.fillna(0)
        dbiogrid_intmap_symm=(dbiogrid_intmap_symm+dbiogrid_intmap_symm.T)/2
        dbiogrid_intmap_symm.index.name='Interactor A'
        dbiogrid_intmap_symm.columns.name='Interactor B'
#         if test:
#             dbiogrid_intmap_symm=dbiogrid_intmap_symm.iloc[:5,:5]
        if filldiagonal_withna:
            dbiogrid_intmap_symm=filldiagonal(dbiogrid_intmap_symm)
        to_table_pqt(dbiogrid_intmap_symm,dbiogrid_intmap_symmp)
        print('file saved at: ',dbiogrid_intmap_symmp)
        ddegnonself,ddegself=get_degrees(dbiogrid_intmap_symm)
        print(ddegself.head())
        dbiogrid_intmap_symm_lin=ddegself.copy()
        dbiogrid_intmap_symm_lin.index.name=f'gene {genefmt}'
        to_table(dbiogrid_intmap_symm_lin,f'{dbiogrid_intmap_symmp}.degrees.tsv')
    else:
        dbiogrid_intmap_symm=read_table_pqt(dbiogrid_intmap_symmp)
    return dbiogrid_intmap_symm