import requests
import pandas as pd
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
        df.columns=[frm,to]
        return df
    else:
        print('Something went wrong ', response.status_code)  
        
def map_ids_batch(queries,interval=1000,params_map_ids={'frm':'ACC','to':'ENSEMBL_PRO_ID'}):
    range2df={}
    for ini,end in zip(range(0,len(queries)-1,interval),range(interval,len(queries)-1+interval,interval)):
        print(ini,end)
        dgeneids=map_ids(queries=queries[ini:end],**params_map_ids)
        range2df[ini]=dgeneids
    return pd.concat(range2df,axis=0).drop_duplicates()

from rohan.dandage.io_sys import runbashcmd
def uniproitid2seq(id,fap):
    runbashcmd(f"wget https://www.uniprot.org/uniprot/{id}.fasta -O tmp.fasta")
    from Bio import SeqIO
    for record in SeqIO.parse(fap, "fasta"):
        return str(record.seq)
        break