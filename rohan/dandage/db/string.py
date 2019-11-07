from rohan.global_imports import *
def get_dint(dint_rawp,dgene_annotp):
    dint_aggscorep=f'{dirname(dint_rawp)}/dint_aggscore.pqt'
    if not exists(dint_aggscorep) or force:
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
        print(dint.shape,end='')
        dint=dint.drop_duplicates(subset=['interaction id','combined_score'])
        print(dint.shape)

        dint['interaction id'].unique().shape[0]==dint.shape[0]

        for q in list(np.arange(0.25,1,0.25)):
            dint.loc[:,f'interaction bool string (score>q{q:.2f})']=dint['combined_score']>dint['combined_score'].quantile(q)
        print(dint.filter(like='interaction bool',axis=1).sum())

        to_table(dint,dint_aggscorep)
    else:
        dint=read_table(dint_aggscorep)   
    return dint_aggscorep