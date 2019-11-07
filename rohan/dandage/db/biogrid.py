from rohan.global_imports import *
# from rohan.dandage.db.intact import get_degrees

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
        outd=f"{dirname(dint_rawp)}/{taxid}/"
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
        dn2params={'direct interaction':{'include':['Two-hybrid','Biochemical Activity','Protein-peptide', 'PCA','Far Western'],},
     'association':{'exclude':['Protein-RNA', "Affinity Capture-RNA",]},
     'physical association':{'include':['Affinity Capture-MS',
                                'Affinity Capture-Western',
                                'Affinity Capture-Luminescence',
                                'Co-fractionation',
                                'Co-purification',
                                'Co-localization',
                                'Proximity Label-MS',
                                'Reconstituted Complex',
                                 ],},}
        if test:
            dlog=pd.DataFrame(dint['Experimental System'].value_counts())
            print(dlog)
            if not logf is None:
                to_table(dlog,f'{logf}.all_int.tsv')
        cols_index=[c for c in dint if c.startswith('Systematic Name Interactor') or c.startswith('Official Symbol Interactor')]
        print(dint.shape,end='')
        dint=dint.dropna(subset=cols_index)
        print(dint.shape)
        dint['interaction id']=dint.apply(lambda x: '--'.join(sorted([' '.join([x[f'Systematic Name Interactor {i}'],x[f'Official Symbol Interactor {i}']]) for i in ['A','B']])),axis=1)

        dint=dint.rename(columns={'Pubmed ID':'interaction score biogrid'})
        dn2df={}
        for dn in dn2params:
            if 'include' in dn2params[dn]:
                print(dint.shape,end='')    
                dn2df[dn]=dint.loc[(dint['Experimental System'].isin(dn2params[dn]['include'])),['interaction id','interaction score biogrid','Experimental System']]
                print(dint.shape)            
            elif 'exclude' in dn2params[dn]:
                print(dint.shape,end='')                
                dn2df[dn]=dint.loc[(~dint['Experimental System'].isin(dn2params[dn]['exclude'])),['interaction id','interaction score biogrid','Experimental System']]
                print(dint.shape)            
            elif len(del_exp_syss)!=0 and len(keep_exp_syss)!=0:
                logging.error('Error: either specify include or exclude')
            if test:
                print('Experimental Systems used.')
                dlog=pd.DataFrame(dn2df[dn]['Experimental System'].value_counts())
                if not logf is None:
                    to_table(dlog,f'{logf}_{dn}.kept_int.tsv')
    #             print(dlog)

        dint_groupbyexptsys=pd.concat(dn2df,axis=0,names=['experimental system type','Unnamed']).reset_index()
        dint_aggbyintid=dint_groupbyexptsys.groupby(['experimental system type','interaction id']).agg({'interaction score biogrid':lambda x: len(unique_dropna(x))}).reset_index()
        dint_aggbyintidmap=dint_aggbyintid.pivot_table(index='interaction id',columns='experimental system type',values='interaction score biogrid')
        dint_aggbyintidmap=dint_aggbyintidmap.rename(columns={k:f"interaction score biogrid {k}" for k in dint_aggbyintidmap})
        for c in dint_aggbyintidmap:
            dint_aggbyintidmap[c.replace('score','bool')]=~dint_aggbyintidmap[c].isnull()
        if test:
            print(dint_aggbyintidmap.filter(like='interaction bool',axis=1).sum())
        dint_aggbyintidmap=dint_aggbyintidmap.reset_index()
        to_table(dint_aggbyintidmap,dintp)
    else:
        dint_aggbyintidmap=read_table(dintp)
    return dint_aggbyintidmap


