import requests
import pandas as pd
from Bio import SeqIO,SeqRecord
from rohan.lib.io_dfs import *

# map ids

def map_ids_request(queries,frm='ACC',to='ENSEMBL_PRO_ID',
        organism_taxid=9606,
        test=False):
    url = 'https://www.uniprot.org/uploadlists/'
    params = {
    'from':frm,
    'to':to,
    'format':'tab',
    'organism':organism_taxid,    
    'query':' '.join(queries),
    }
    response = requests.get(url, params=params)
    if test:
        print(response.url)
    if response.ok:
        df=pd.read_table(response.url)
        if len(df.columns)==2:
            df.columns=[frm,to]
        else:
            renamecols=[c for c in df if c.startswith('yourlist:')]+[c for c in df if c.startswith('isomap:')]
            df=df.rename(columns={c:c[:6] for c in renamecols})
        return df
    else:
        print('Something went wrong ', response.status_code) 

def map_ids_request_in_batches(queries,frm='ACC',to='ENSEMBL_PRO_ID',
        organism_taxid=9606,
        interval=1000,
        test=False,
        **kws_queries,
        ):
    range2df={}
    for ini,end in zip(range(0,len(queries)-1,interval),range(interval,len(queries)-1+interval,interval)):
        print(end,end=', ')
        dgeneids=map_ids_request(queries=queries[ini:end],frm=frm,to=to,
        organism_taxid=organism_taxid,
        test=test)
        range2df[ini]=dgeneids
    if len(range2df.keys())!=0:
#         print(range2df)
        return pd.concat(range2df,axis=0).drop_duplicates()
    else:
        return pd.DataFrame(columns=[frm,to])        

def map_ids(queries,frm='ACC',to='ENSEMBL_PRO_ID',
            organism_taxid=9606,
            intermediate=None,
            interval=1000,
            test=False,
            **kws_map_ids,
            ):
    """
    
        https://www.uniprot.org/help/api_idmapping
        Category: Genome annotation databases
        Ensembl	ENSEMBL_ID	both
        Ensembl Protein	ENSEMBL_PRO_ID	both
        Ensembl Transcript	ENSEMBL_TRS_ID	both
        Ensembl Genomes	ENSEMBLGENOME_ID	both
        Ensembl Genomes Protein	ENSEMBLGENOME_PRO_ID	both
        Ensembl Genomes Transcript	ENSEMBLGENOME_TRS_ID	both
        Entrez Gene (GeneID)	P_ENTREZGENEID
        Category: 3D structure databases
        PDB	PDB_ID	both
        Category: Protein-protein interaction databases
        BioGrid	BIOGRID_ID	both
        ComplexPortal	COMPLEXPORTAL_ID	both
        DIP	DIP_ID	both
        STRING	STRING_ID	both
        def map_ids_batch(queries,interval=1000,params_map_ids={'frm':'ACC','to':'ENSEMBL_PRO_ID'},
                     intermediate=None):
        # def map_ids_chained(queries,frm='GENENAME',
        #                     to='ENSEMBL_ID',intermediate='ACC',
        #                     **kws_map_ids):
        
    """
    if intermediate is None:
        return map_ids_request_in_batches(queries=queries,
                                         frm=frm,to=to,
                                        organism_taxid=organism_taxid,
                                        interval=interval,
                                        test=test)
    else:
        df1=map_ids_request_in_batches(queries=queries,
                                         frm=frm,to=intermediate,
                                        organism_taxid=organism_taxid,
                                        interval=interval,
                                        test=test)
        if len(df1)==0:
            logging.error(f"conversion from {frm} to {intermediate} failed")
        df2=map_ids_request_in_batches(queries=df1[intermediate].unique().tolist(),
                                         frm=intermediate,to=to,
                                        organism_taxid=organism_taxid,
                                        interval=interval,
                                        test=test)
        if len(df2)==0:
            logging.error(f"conversion from {intermediate} to {to} failed")
        return df1.merge(df2,on=intermediate,how='left').drop([intermediate],axis=1).dropna()        
        
def geneid2proteinid(df,frm='ENSEMBLGENOME_ID',to='ENSEMBLGENOME_PRO_ID',interval=400):
    rename={frm:'gene id',to:'protein id'}
    organism_id=str(df['organism id'].unique()[0])
    df_1=map_ids_batch(
        queries=df['gene id'].tolist(),
        interval=interval,
        params_map_ids={'frm': frm, 'to': 'ACC',
                        'organism_taxid':organism_id,'test':True,},
    )
    if len(df_1)==0:
        print(organism_id)
        return #df_1.rename(columns=rename)
    df_2=map_ids_batch(
        queries=df_1['ACC'].unique().tolist(),
        interval=interval,
        params_map_ids={'frm': 'ACC', 'to': to,
                        'organism_taxid':organism_id,'test':True},
    )
    return df_1.merge(df_2,on='ACC',how='left').drop_duplicates(subset=[frm,to]).rename(columns=rename)    

## sequence 
def get_sequence(queries,fap=None,fmt='fasta',
            organism_taxid=9606,
                 test=False):
    """
    Examples:
        http://www.ebi.ac.uk/Tools/dbfetch/dbfetch?db=uniprotkb&id=P14060+P26439&format=fasta&style=raw&Retrieve=Retrieve
        https://www.uniprot.org/uniprot/?format=fasta&organism=9606&query=O75116+O75116+P35548+O14944+O14944
        
    """
    url = 'http://www.ebi.ac.uk/Tools/dbfetch/dbfetch'
    params = {
    'id':' '.join(queries),
    'db':'uniprotkb',
    'organism':organism_taxid,    
    'format':fmt,
    'style':'raw',
    'Retrieve':'Retrieve',
    }
    response = requests.get(url, params=params)
    if test:
        print(response.url)
    if response.ok:
        if not fap is None:
            with open(fap,'w') as f:
                f.write(response.text)
            return fap
        else:
            return response.text            
    else:
        print('Something went wrong ', response.status_code) 

def get_sequence_batch(queries,fap,interval=1000,params_get_sequence={'organism_taxid':9606,}):
    text=''
    for ini,end in zip(range(0,len(queries)-1,interval),range(interval,len(queries)-1+interval,interval)):
        print(f"{ini}-{end}",end=' ')
        text_=get_sequence(queries=queries[ini:end],**params_get_sequence
                          )
        text=f"{text}\n{text_}"
    with open(fap,'w') as f:
        f.write(text)
    return fap

from rohan.lib.io_sys import runbashcmd
def uniproitid2seq(id,fap='tmp.fasta'):
    runbashcmd(f"wget https://www.uniprot.org/uniprot/{id}.fasta -O {fap}")
    from Bio import SeqIO
    for record in SeqIO.parse(fap, "fasta"):
        return str(record.seq)
        break
        
        
## consanguinity with ensembl
def get_ensembl_ids(df,coluid):
    dgeneids_ensp=map_ids(queries=df[coluid].unique().tolist(),
                     frm='ACC',to='ENSEMBL_PRO_ID',test=True)
    dgeneids_enst=map_ids(queries=df[coluid].unique().tolist(),
                     frm='ACC',to='ENSEMBL_TRS_ID',test=True)
    dgeneids=dgeneids_ensp.merge(dgeneids_enst,on='ACC',how='outer')
    df_ens=df.merge(dgeneids,
                      left_on=coluid,right_on='ACC',how='outer')
    return df_ens

def compare_uniprot2ensembl(uid,ensp,ensembl,test=False):
    if test:
        print(f"uid='{uid}',enpt='{enpt}'")
    useq=uniproitid2seq(uid)    
    eseq=ensembl.protein_sequence(ensp)
#     print(useq)
#     print(eseq)
    if useq==eseq:
        return True
    else:
        return False
#         logging.info('lengths not equal')     

# from rohan.lib.db.ensembl import enst2prtseq
# def filter_compare_uniprot2ensembl(df,coluid,colensp,test=False): 
# #     id2seq={}
# #     for record in SeqIO.parse(fap, "fasta"):
# #         id2seq[record.id]=str(record.seq)
#     import pyensembl
#     ensembl = pyensembl.EnsemblRelease(species=pyensembl.species.Species.register(
#     latin_name='homo_sapiens',
#     synonyms=['homo_sapiens'],
#     reference_assemblies={
#         'GRCh38': (95, 95),
#     }),release=95)
#     return=df.apply(lambda x: compare_uniprot2ensembl(x[coluid],x[colensp],ensembl=ensembl,test=test),axis=1)
    
from rohan.lib.io_seqs import ids2seqs2fasta
from Bio import SeqIO,SeqRecord
def normalise_uniprot_fasta(fap,test=True):
    faoutp=f"{fap}.normalised.fasta"
    id2seq={}
    for record in SeqIO.parse(fap, "fasta"):
        if record.description.startswith('sp|'):
            record_id=record.description.split('|')[1]            
        else:    
            record_id=record.description.split(' ')[1]
        if test:
            print(record_id)
        id2seq[record_id]=str(record.seq)
    ids2seqs2fasta(id2seq,faoutp)
    return faoutp

## sequence features

# from rohan.global_imports import *
# from rohan.lib.io_dict import to_dict,read_dict

def uniprotid2features(uniprotid,databasep='data/database',force=False,out_dict=True):
    from rohan.lib.io_dfs import read_table,to_table
    from rohan.lib.io_dict import read_dict,to_dict
    dp=f'{databasep}/uniprot/{uniprotid}.json'
    dfp=f'{databasep}/uniprot/{uniprotid}.tsv'
    if all([exists(dp),exists(dfp)]) and not force:
        if out_dict:
            return read_dict(dp),read_table(dfp)
        else:
            return read_table(dfp)
    import requests, sys
    urls=[ f'https://www.ebi.ac.uk/proteins/api/features/{uniprotid}?types=ACT_SITE,DOMAIN,HELIX,TURN,STRAND,REGION,MOTIF,VARIANT,INIT_MET,SIGNAL,PROPEP,TRANSIT,CHAIN,PEPTIDE,TOPO_DOM,TRANSMEM,REPEAT,CA_BIND,ZN_FING,DNA_BIND',
            'https://www.ebi.ac.uk/proteins/api/features/P12931?types=NP_BIND,COILED,COMPBIAS,METAL,BINDING,SITE,NON_STD,MOD_RES,LIPID,CARBOHYD,DISULFID,CROSSLNK,VAR_SEQ,MUTAGEN,UNSURE,CONFLICT,NON_CONS,NON_TER,INTRAMEM',
         ]
    uniprotid2features={}
    for requestURL in urls:
        r = requests.get(requestURL, headers={ "Accept" : "application/json"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        responseBody = r.text
        import ast
        uniprotid2features.update(ast.literal_eval(responseBody))

    dfs=[]
    for requestURL in urls: 
        r = requests.get(requestURL, headers={ "Accept" : "text/x-gff"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        responseBody = r.text
        from io import StringIO
        df1=delunnamedcol(pd.read_table(StringIO(responseBody),comment='#',
                      names=[
                             'end','Unnamed1','Unnamed2','Unnamed3','feature description'],
                                       header=None,))
        df1.index.names=['uniprot id','db','feature type','start']
        df1=df1.reset_index()
        dfs.append(df1)
    df1=pd.concat(dfs,axis=0)
    def get_feature_name(s):
        if len(s['feature description'])>1:
            from rohan.lib.io_dict import str2dict
            d=str2dict(s['feature description'])
            return d['Note'] if 'Note' in d else d['ID'] if 'ID' in d else s['feature type']
    df1['feature name']=df1.apply(lambda x: get_feature_name(x),axis=1)
    df1.loc[df1['feature type'].isin(['HELIX','STRAND','TURN']),'feature type']='secondary structure'
    d,df=uniprotid2features,df1.drop(['feature description'],axis=1)
    to_dict(d,dp)
    to_table(df,dfp)
    if out_dict:
        return d,df
    else:
        return df
#     to_dict(uniprotid2features,uniprotid2featuresp)
#     to_table(df1.drop(['feature description'],axis=1),dseqfeaturesp)
def uniprotids2features(queries,databasep='data/database',fast=False,force=False):
    df1=pd.DataFrame({'uniprot id':queries,}).dropna()
    df2=getattr(df1.groupby('uniprot id',as_index=False),f"{'parallel' if fast else 'progress'}_apply")(lambda x: uniprotid2features(x.iloc[0,:]['uniprot id'],
                                                                                                                                     databasep='data/database',
                                                                                                                                     force=force,out_dict=False))
    return df2

