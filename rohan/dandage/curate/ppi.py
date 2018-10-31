import pandas as pd
import numpy as np
from os.path import basename,dirname,exists
from rohan.dandage.io_dfs import *

def get_dbiogrid_intmap(taxid,dbiogridp,dbiogrid_intmapp,dbiogrid_intlin_ppip,dbiogrid_intmap_ppip,
                        genefmt='name',force=False,test=False,del_exp_syss=[],
                        filldiagonal_withna=False):
    """
    taxid=559292
    del_exp_syss=["Co-purification", "Co-fractionation", "Proximity Label-MS", 
                                  "Affinity Capture-RNA", "Protein-peptide"]
    """    
    gene_fmt2colns={'biogrid':{'id':'Systematic Name',
                 'name':'Official Symbol'},}
    if (not exists(dbiogrid_intmapp)) or force:
        if not exists(dbiogrid_intmap_ppip) or force:
            if not exists(dbiogrid_intlin_ppip) or force:
                dbiogrid=pd.read_csv(dbiogridp,
                       sep='\t',low_memory=False)
                # filter biogrid
                # taxonomic id of Scer
                dbiogrid=dbiogrid.loc[((dbiogrid['Experimental System Type']=='physical') \
                           & (dbiogrid['Organism Interactor A']==taxid) \
                           & (dbiogrid['Organism Interactor B']==taxid)),:]

                print('All physical Experimental Systems.')
                if test:
                    print(pd.DataFrame(dbiogrid['Experimental System'].value_counts()))
                if len(del_exp_syss)!=0:
                    dbiogrid=dbiogrid.loc[((~dbiogrid['Experimental System'].isin(del_exp_syss))),:]
                    if test:
                        print('After removing',del_exp_syss)
                        print('Physical Experimental Systems used.')
                        print(pd.DataFrame(dbiogrid['Experimental System'].value_counts()))
                dbiogrid.to_csv(dbiogrid_intlin_ppip,sep='\t')
            else:
                dbiogrid=pd.read_table(dbiogrid_intlin_ppip)
            # this cell makes a symmetric interaction map
            dbiogrid['count']=1
            if test:
                print(dbiogrid['count'].sum())
            # mean of counts of intearations
            ## higher is more confident interaction captured in multiple assays
            dbiogrid_grid=dbiogrid.pivot_table(values='count',
                                             index=f"{gene_fmt2colns['biogrid'][genefmt]} Interactor A",
                                            columns=f"{gene_fmt2colns['biogrid'][genefmt]} Interactor B",
                                            aggfunc='sum',)

            # make it symmetric
            if test:
                print('shape of non-symm intmap: ',dbiogrid_grid.shape)
            to_table(dbiogrid_grid,dbiogrid_intmap_ppip)
            to_table_pqt(dbiogrid_grid,dbiogrid_intmap_ppip+'.pqt')
        else:         
#             dbiogrid_grid=pd.read_table(dbiogrid_intmap_ppip)
            dbiogrid_grid=read_table_pqt(dbiogrid_intmap_ppip+'.pqt')
        dbiogrid_grid=set_index(dbiogrid_grid,'Official Symbol Interactor A')
        geneids=set(dbiogrid_grid.index).union(set(dbiogrid_grid.columns))
        if test:
            print('total number of genes',len(geneids))
        # missing rows to nan
        dbiogrid_grid_symm=pd.DataFrame(columns=geneids,index=geneids)

        dbiogrid_grid_symm.loc[dbiogrid_grid.index,:]=dbiogrid_grid.loc[dbiogrid_grid.index,:]
        dbiogrid_grid_symm.loc[:,dbiogrid_grid.columns]=dbiogrid_grid.loc[:,dbiogrid_grid.columns]
        if test:
            print(dbiogrid_grid_symm.shape)
        dbiogrid_grid_symm=dbiogrid_grid_symm.fillna(0)
        dbiogrid_grid_symm=(dbiogrid_grid_symm+dbiogrid_grid_symm.T)/2
        dbiogrid_grid_symm.index.name='Interactor A'
        dbiogrid_grid_symm.columns.name='Interactor B'
#         if test:
#             dbiogrid_grid_symm=dbiogrid_grid_symm.iloc[:5,:5]
        if filldiagonal_withna:
            dbiogrid_grid_symm=filldiagonal(dbiogrid_grid_symm)
        to_table(dbiogrid_grid_symm,dbiogrid_intmapp)
        to_table_pqt(dbiogrid_grid_symm,dbiogrid_intmapp+'.pqt')
        print('file saved at: ',dbiogrid_intmapp)
    else:
        dbiogrid_grid_symm=read_table_pqt(dbiogrid_intmapp+'.pqt')
    return dbiogrid_grid_symm