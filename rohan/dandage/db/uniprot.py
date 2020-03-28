import requests
import pandas as pd
from Bio import SeqIO,SeqRecord
from rohan.dandage.io_dfs import *

def get_sequence(queries,fap=None,fmt='fasta',
            organism_taxid=9606,
                 test=False):
    """
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
        print(ini,end)
        text_=get_sequence(queries=queries[ini:end],**params_get_sequence
                          )
        text=f"{text}\n{text_}"
    with open(fap,'w') as f:
        f.write(text)
    return fap


def map_ids(queries,frm='ACC',to='ENSEMBL_PRO_ID',
            organism_taxid=9606,test=False):
    """
    https://www.uniprot.org/help/api_idmapping
    Category: Genome annotation databases
    Ensembl	ENSEMBL_ID	both
    Ensembl Protein	ENSEMBL_PRO_ID	both
    Ensembl Transcript	ENSEMBL_TRS_ID	both
    Ensembl Genomes	ENSEMBLGENOME_ID	both
    Ensembl Genomes Protein	ENSEMBLGENOME_PRO_ID	both
    Ensembl Genomes Transcript	ENSEMBLGENOME_TRS_ID	both
    Category: 3D structure databases
    PDB	PDB_ID	both
    Category: Protein-protein interaction databases
    BioGrid	BIOGRID_ID	both
    ComplexPortal	COMPLEXPORTAL_ID	both
    DIP	DIP_ID	both
    STRING	STRING_ID	both
    """
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
        
def map_ids_batch(queries,interval=1000,params_map_ids={'frm':'ACC','to':'ENSEMBL_PRO_ID'}):
    range2df={}
    for ini,end in zip(range(0,len(queries)-1,interval),range(interval,len(queries)-1+interval,interval)):
        print(ini,end)
        dgeneids=map_ids(queries=queries[ini:end],**params_map_ids)
        range2df[ini]=dgeneids
    if len(range2df.keys())!=0:
        return pd.concat(range2df,axis=0).drop_duplicates()
    else:
        return pd.DataFrame(columns=[params_map_ids['frm'],params_map_ids['to']])
from rohan.dandage.io_sys import runbashcmd
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

# from rohan.dandage.db.ensembl import enst2prtseq
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
    
from rohan.dandage.io_seqs import ids2seqs2fasta
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

# from rohan.global_imports import *
# from rohan.dandage.io_dict import to_dict,read_dict

def uniprotid2features(uniprotid):
    import requests, sys
    urls=[f'https://www.ebi.ac.uk/proteins/api/features/{uniprotid}?types=ACT_SITE,DOMAIN,HELIX,TURN,STRAND,REGION,MOTIF,VARIANT,INIT_MET,SIGNAL,PROPEP,TRANSIT,CHAIN,PEPTIDE,TOPO_DOM,TRANSMEM,REPEAT,CA_BIND,ZN_FING,DNA_BIND',
'https://www.ebi.ac.uk/proteins/api/features/P12931?types=NP_BIND,COILED,COMPBIAS,METAL,BINDING,SITE,NON_STD,MOD_RES,LIPID,CARBOHYD,DISULFID,CROSSLNK,VAR_SEQ,MUTAGEN,UNSURE,CONFLICT,NON_CONS,NON_TER,INTRAMEM']
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
                             'end','Unnamed1','Unnamed2','Unnamed3','feature description']))
        df1.index.names=['uniprot id','db','feature type','start']
        df1=df1.reset_index()
        dfs.append(df1)
    df1=pd.concat(dfs,axis=0)
    def get_feature_name(s):
        from rohan.dandage.io_dict import str2dict
        d=str2dict(s['feature description'])
        return d['Note'] if 'Note' in d else d['ID'] if 'ID' in d else s['feature type']
    df1['feature name']=df1.apply(lambda x: get_feature_name(x),axis=1)
    df1.loc[df1['feature type'].isin(['HELIX','STRAND','TURN']),'feature type']='secondary structure'
    return uniprotid2features,df1.drop(['feature description'],axis=1)
#     to_dict(uniprotid2features,uniprotid2featuresp)
#     to_table(df1.drop(['feature description'],axis=1),dseqfeaturesp)
    