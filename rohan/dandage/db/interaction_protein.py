from rohan.global_imports import *

def miid2descr(miid):
    import requests
    page=requests.get(f"https://www.ebi.ac.uk/ols/api/ontologies/mi/terms?iri=http://purl.obolibrary.org/obo/{miid.replace(':','_')}")
    if page.ok:    
        try:
            return page.json()['_embedded']['terms'][0]['annotation']['definition'][0]
        except:
            return np.nan

def get_degrees(dintmapbinary,coln='# of interactions'):
    ddegself=getdiagonalvals(dintmapbinary).rename(columns={'diagonal value':'# of interactions'})
    ddegself[coln]=ddegself[coln].astype(int)
    
    dintmapbinary=filldiagonal(dintmapbinary,False)
    ddegnonself=dintmapbinary.sum()
    ddegnonself.name='# of interactions'
    ddegnonself=pd.DataFrame(ddegnonself)
    return ddegnonself,ddegself  

def dintmap2lin(dint_db):
    dint_db=dmap2lin(dint_db,idxn='gene id gene name interactor1',coln='gene id gene name interactor2',colvalue_name='interaction score db').replace(0,np.nan).dropna()
    dint_db['interaction id']=dint_db.apply(lambda x : '--'.join(sorted([x['gene id gene name interactor1'],x['gene id gene name interactor2']])),axis=1)
    dint_db=dint_db.drop_duplicates(subset=['interaction id'])
    dint_db['interaction bool db']=True
    return dint_db
                      
class biogrid():
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


                      
class hitpredict():
    def get_int(dint_rawp,dgene_annotp,
                force=False):

        dint_intidp=f"{dirname(dint_rawp)}/{basenamenoext(dint_rawp)}/dint_intid.pqt"
        dint_scorep=f"{dirname(dint_rawp)}/{basenamenoext(dint_rawp)}/dint_score.pqt"

        if not exists(dint_scorep) or force:
            if not exists(dint_intidp) or force:
                df0=pd.read_csv(dint_rawp, compression='gzip', header=4, sep='\t', quotechar='"')

                for i,a in zip([1,2],['A','B']):
                    df0[f'gene{i} name']=df0[f'Interactor {a} Name'].apply(lambda x: x.split('_')[0] if not pd.isnull(x) else x)

                genename2id=read_table(dgene_annotp).drop_duplicates(subset=['gene name','gene id']).dropna(subset=['gene name','gene id']).set_index('gene name')['gene id'].to_dict()
                for i in [1,2]:
                    df0[f'gene{i} id']=df0[f'gene{i} name'].map(genename2id)
                print(df0.shape,end='')
                df0=df0.dropna(subset=df0.filter(regex='^gene',axis=1).columns.tolist())
                print(df0.shape)
                df0['interaction id']=df0.apply(lambda x: '--'.join(sorted([' '.join([x[f'gene{i} id'],x[f'gene{i} name']]) for i in [1,2]])),axis=1)
                df0=df0.rename(columns={'Confidence score':'interaction score hitpredict'})
                to_table(df0,dint_intidp)
            else:
                df0=read_table(dint_intidp)

            df1=df0.drop_duplicates(subset=['interaction id','interaction score hitpredict']).loc[:,['interaction id','interaction score hitpredict']]
            print(len(df1)==len(df0))
            df1.loc[:,'interaction bool hitpredict']=~df1['interaction score hitpredict'].isnull()
            for q in list(np.arange(0.25,1,0.25)):
                df1.loc[:,f'interaction bool hitpredict (score>q{q:.2f})']=df1['interaction score hitpredict']>=(df1['interaction score hitpredict'].quantile(q) if q!=0 else 0)
            print(df1.filter(like='interaction bool',axis=1).sum())
            to_table(df1,dint_scorep)
        else:
            df1=read_table(dint_scorep)    
        return df1
    
class string():
    def get_dint(dint_rawp,dgene_annotp,force=False):
        dint_aggscorep=f'{dirname(dint_rawp)}/dint_aggscore.pqt'
        dint_intidp=f'{dirname(dint_rawp)}/dint_intid.pqt'
        if not exists(dint_intidp) or force:
            dint=pd.read_csv(dint_rawp,compression='gzip', sep=' ')
            ## daction has extra info abt directionality and mode of interaction which is unnecessary
            # # dactions=pd.read_csv('database/string/4932/4932.protein.actions.v11.0.txt.gz', compression='gzip', sep='\t')
            # dactions['mode'].value_counts()
            # print(dactions.shape,end='')
            # dactions=dactions.loc[((dactions['mode']=='binding') & (dactions['is_directional']=='f') & (dactions['a_is_acting']=='f')),:]
            # print(dactions.shape)
            for i in [1,2]:
                dint[f'gene{i} id']=dint[f'protein{i}'].apply(lambda x:x.split('.')[1])
            geneid2name=read_table(dgene_annotp).drop_duplicates(subset=['gene name','gene id']).dropna(subset=['gene name','gene id']).set_index('gene id')['gene name'].to_dict()
            for i in [1,2]:
                dint[f'gene{i} name']=dint[f'gene{i} id'].map(geneid2name)
            print(dint.shape,end='')
            dint=dint.dropna(subset=dint.filter(like='gene',axis=1).columns.tolist())
            print(dint.shape)

            dint['interaction id']=dint.apply(lambda x: '--'.join(sorted([' '.join([x[f'gene{i} id'],x[f'gene{i} name']]) for i in [1,2]])),axis=1)
            dint=dint.rename(columns={'combined_score':'interaction score string'})
            print(dint.shape,end='')
            dint=dint.drop_duplicates(subset=['interaction id','interaction score string'])
            print(dint.shape)

            print(dint['interaction id'].unique().shape[0]==dint.shape[0])
            dint=dint.reset_index()
            to_table(dint,dint_intidp)
        else:
            dint=read_table(dint_intidp)
        if not exists(dint_aggscorep) or force:    
            dint.loc[:,f'interaction bool string']=~dint['interaction score string'].isnull()
            for q in list(np.arange(0.2,0.8,0.2))+[0.8,0.9,0.95,0.975,0.99]:
                dint.loc[:,f'interaction bool string (score>q{q:.2f})']=dint['interaction score string']>=(dint['interaction score string'].quantile(q) if q!=0 else 0)
            print(dint.filter(like='interaction bool',axis=1).sum())
            to_table(dint,dint_aggscorep)
        else:
            dint=read_table(dint_aggscorep)   
        return dint   
      
                      
# from rohan.dandage.db.intact import *
class intact():
    def get_dint(speciesn,dgene_annotp,dintact_rawp=None,force=False):
        """
        contains huri 

           dintact.loc[(dintact['Publication 1st author(s)']=='Huri, et al. (2017)'),'Update date'].unique()

           array(['2018/04/29', '2018/04/30', '2018/09/18', '2019/01/23',
           '2018/10/12', '2018/10/11', '2018/07/23', '2018/09/21',
           '2019/01/24', '2018/07/05', '2019/02/20', '2019/02/21',
           '2018/05/01', '2018/06/02', '2018/05/03'], dtype=object)

        intact ftp: ftp://ftp.ebi.ac.uk/pub/databases/intact/current/psimitab/

        file format https://github.com/HUPO-PSI/miTab/blob/master/PSI-MITAB27Format.md
        """
        from rohan.dandage.io_strs import replacemany,get_bracket
        if dintact_rawp is None:
            dintact_rawp='database/intact/pub/databases/intact/current/psimitab/intact.pqt'

        dintact_speciesidsp=f"{dirname(dintact_rawp)}/intact_speciesids.tsv"
        if not exists(dintact_speciesidsp):
            dintact=read_table(dintact_rawp)
            speciesids=unique_dropna(dintact['Taxid interactor A'].unique().tolist()+dintact['Taxid interactor B'].unique().tolist())
            to_table(pd.DataFrame({'species id':speciesids}),
                    dintact_speciesidsp)
        else:
            speciesids=read_table(dintact_speciesidsp)['species id'].tolist()   
        speciesids_=[s for s in speciesids if speciesn in s]
        if len(speciesids_)==0:
            logging.error('species not found')
            logging.erorr(f"please choose between {','.join(dintact['Taxid interactor A'].unique().tolist())}")
            return None
        elif len(speciesids_)>1:
            logging.error('multiple species not found')
            print(speciesids_)
            return None 
        speciesid=speciesids_[0]

        # get the species id for filtering by species
        dintact_aggscorep=f"{dirname(dintact_rawp)}/dintact_flt_{speciesid.replace(' ','_')}_aggscore.pqt"
        if exists(dintact_aggscorep) and not force:
            return read_table(dintact_aggscorep)
    #     print(exists(dintact_aggscorep), force)
    #     print(dintact_aggscorep)
    #     brk
        # filter the raw data file 
        dintact_fltbyspeciesp=f'{dirname(dintact_rawp)}/dintact_flt_{speciesid}.pqt'
        if not exists(dintact_fltbyspeciesp):
            if not 'dintact' in globals():
                dintact=read_table(dintact_rawp)

            ### filterby organism of proteins
            print(dintact.shape,end=' ')
            dintact=dintact.loc[(dintact['Taxid interactor A']==speciesid),:]
            print(dintact.shape,end=' ')
            dintact=dintact.loc[((dintact['Taxid interactor A']==dintact['Taxid interactor B'])),:]
            print(dintact.shape)
            to_table(dintact,dintact_fltbyspeciesp)
        else:
            dintact=read_table(dintact_fltbyspeciesp)

        ### get only proteins

        for i in ['A','B']:
            print(dintact[f'Type(s) interactor {i}'].apply(lambda x: x.count('(')).value_counts()==len(dintact))
            dintact[f'Type(s) interactor {i}: type']=dintact[f'Type(s) interactor {i}'].apply(lambda x: get_bracket(x))
            print(dintact[f'Type(s) interactor {i}: type'].value_counts())

        ### filter to only ppi
        print(dintact.shape,end=' ')
        dintact=dintact.loc[(dintact['Type(s) interactor A: type']=='protein'),:]
        print(dintact.shape,end=' ')
        dintact=dintact.loc[((dintact['Type(s) interactor A: type']==dintact['Type(s) interactor B: type'])),:]
        print(dintact.shape)

        #### only annoted genes
        def alias2geneid(x,test=False):
            l=[replacemany(s,['uniprotkb:','(locus name)'],replacewith='') for s in x.split('|') if '(locus name)' in s]
            if len(l)==1:
                return l[0]
            else:
                if test:
                    print(len(l),x)
                return None
        dintact['gene1 id']=dintact['Alias(es) interactor A'].apply(lambda x: alias2geneid(x))
        dintact['gene2 id']=dintact['Alias(es) interactor B'].apply(lambda x: alias2geneid(x))

        print(dintact.shape,end='')
        dintact=dintact.dropna(subset=['gene1 id','gene2 id'])
        print(dintact.shape)

        #### only
    #         physical association
    #         direct interaction

        print(dintact['Interaction type(s)'].apply(lambda x: x.count('(')).value_counts()==len(dintact))
        dintact['Interaction type(s): type']=dintact['Interaction type(s)'].apply(lambda x: get_bracket(x))
        print(dintact['Interaction type(s): type'].value_counts())

        print(dintact.shape,end='')
        dintact=dintact.loc[dintact['Interaction type(s): type'].isin(['physical association','direct interaction']),:]
        print(dintact.shape)

        #### make interaction id and trim the table

        geneid2name=read_table(dgene_annotp).drop_duplicates(subset=['gene name','gene id']).dropna(subset=['gene name','gene id']).set_index('gene id')['gene name'].to_dict()

        for i in [1,2]:
            dintact[f'gene{i} name']=dintact[f'gene{i} id'].map(geneid2name)

        print(dintact.shape,end='')
        dintact=dintact.dropna(subset=dintact.filter(regex='^gene *',axis=1).columns.tolist())
        print(dintact.shape)

        dintact['interaction id']=dintact.apply(lambda x: '--'.join(sorted([' '.join([x[f'gene{i} id'],x[f'gene{i} name']]) for i in [1,2]])),axis=1)

        print(sum(dintact['Confidence value(s)'].str.contains('intact-miscore:'))==len(dintact))

        dintact['Confidence value(s) intact-miscore']=dintact['Confidence value(s)'].apply(lambda x: [s.replace('intact-miscore:','') for s in x.split('|') if 'intact-miscore:' in s][0]).apply(float)

        #### aggreagate scores

        def agg_scores(df):
            print(df['Interaction type(s): type'].unique())
            print(df.groupby('interaction id').agg({'Confidence value(s) intact-miscore':lambda x: len(unique(x))})['Confidence value(s) intact-miscore'].sum()==len(df['interaction id'].unique()))
            dfagg=df.groupby('interaction id').agg({'Confidence value(s) intact-miscore':np.mean})
            dfagg.columns=coltuples2str(dfagg.columns)
            return dfagg 
        dintact_agg=dintact.groupby(['Interaction type(s): type']).apply(lambda df : agg_scores(df)).reset_index()

        dintact_aggmap=dintact_agg.pivot_table(columns='Interaction type(s): type',index='interaction id',values='Confidence value(s) intact-miscore')

        cols=dintact_aggmap.columns.tolist()
        for col in cols:
            dintact_aggmap.loc[:,f'interaction bool intact {col}']=~dintact_aggmap[col].isnull()
            for q in list(np.arange(0.25,1,0.25)):
                dintact_aggmap.loc[:,f'interaction bool intact {col} (score>q{q:.2f})']=dintact_aggmap[col]>=(dintact_aggmap[col].quantile(q) if q!=0 else q)
        print(dintact_aggmap.sum())
        dintact_aggmap=dintact_aggmap.reset_index()
        to_table(dintact_aggmap,dintact_aggscorep)
        return dintact_aggmap                     
                          

def get_dint_combo(force=False):
    dintp='database/interactions_protein/559292/dint.pqt'
    if not exists(dintp) or force:
        dn2df={}
        dn2df['biogrid']=biogrid.get_dint(taxid=559292, dint_rawp='database/biogrid/BIOGRID-ORGANISM-3.5.167.tab2/BIOGRID-ORGANISM-Saccharomyces_cerevisiae_S288c-3.5.167.tab2.txt.pqt',
                    outd=None, experimental_system_type='physical', 
                    force=force, test=False).filter(regex='^interaction ',axis=1).set_index('interaction id')

        dn2df['intact']=intact.get_dint(speciesn='559292',
                    dintact_rawp='database/intact/pub/databases/intact/current/psimitab/intact.pqt',
                    dgene_annotp='data_hybridator/data_annot/dgene_annot.tsv',
                    force=force).filter(regex='^interaction ',axis=1).set_index('interaction id')

        dn2df['string']=string.get_dint(dint_rawp='database/string/4932/4932.protein.links.v11.0.txt.gz',
                    dgene_annotp='data_hybridator/data_annot/dgene_annot.tsv',
                    force=force).filter(regex='^interaction ',axis=1).set_index('interaction id')
        dn2df['hitpredict']=hitpredict.get_int(dint_rawp='database/hitpredict/S_cerevisiae_interactions_MITAB-2.5.tgz',
                    dgene_annotp='data_hybridator/data_annot/dgene_annot.tsv',
                    force=force).filter(regex='^interaction ',axis=1).set_index('interaction id')

        dint=pd.concat(dn2df,join='outer',axis=1,sort=True).filter(like='interaction ',axis=1)
        print(intersections({k:unique(dn2df[k].index.tolist()) for k in dn2df}))
        dint.columns=dint.columns.droplevel(0)
        dint.index.name='interaction id'
        ## combine bool
        for col in dint.filter(like='interaction bool',axis=1):
            dint[col]=dint[col].fillna(False).apply(bool)
        dint.loc[:,'interaction bool db direct interaction']=dint.loc[:,['interaction bool biogrid direct interaction',
                                                                         'interaction bool intact direct interaction',
                                                                        ]].T.any()
        dint.loc[:,'interaction bool db physical association']=dint.loc[:,['interaction bool biogrid physical association',
                                                                           'interaction bool intact physical association',
                                                                          ]].T.any()
        print(dint.filter(like='interaction bool',axis=1).sum())
        ## combine scores
        print(dint.filter(like='interaction score',axis=1).columns.tolist())
        for c in ["interaction score hitpredict",'interaction score string']:
            dint[f"{c} rescaled"]=(dint[c]-dint[c].min())/(dint[c].max()-dint[c].min())
        dint['interaction score db']=dint.filter(regex=r'^interaction score .*rescaled$',axis=1).T.mean()
        ## save 
        to_table(dint,dintp)
    else:
        dint=read_table(dintp)
    return dint