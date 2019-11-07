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
                      
# from rohan.dandage.db.intact import *

def get_dintact_flt(speciesn,dgene_annotp,dintact_rawp=None,force=False):
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
    dintact_aggscorep=f'{dirname(dintact_rawp)}/dintact_flt_{speciesid}_aggscore.pqt'
    if exists(dintact_aggscorep) and not force:
        return read_table(dintact_aggscorep)
    dintact_fltbyspeciesp=f'{dirname(dintact_rawp)}/dintact_flt_{speciesid}.pqt'
    if not exists(dintact_fltbyspeciesp):
        dintact=read_table(dintact_rawp)

        df_=dintact.loc[(dintact['Taxid interactor A'].isin([s for s in dintact['Taxid interactor A'].unique().tolist() if speciesn in s])),'Taxid interactor A'].value_counts()

        if len(df_)==0:
            logging.error('species not found')
            logging.erorr(f"please choose between {','.join(dintact['Taxid interactor A'].unique().tolist())}")
            return None
        elif len(df_)>1:
            logging.error('multiple species not found')
            print(df_)
            return None 

        speciesid=df_.index.tolist()[0]
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
        physical association
        direct interaction

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

    for col in dintact_aggmap:
        for q in list(np.arange(0,1,0.25)):
            dintact_aggmap.loc[:,f'interaction bool intact {col} (score>q{q:.2f})']=dintact_aggmap[col]>(dintact_aggmap[col].quantile(q) if q!=0 else q)
    print(dintact_aggmap.sum())
    to_table(dintact_aggmap,dintact_aggscorep)
    return dintact_aggmap                      