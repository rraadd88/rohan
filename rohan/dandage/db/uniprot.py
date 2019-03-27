import requests
import pandas as pd
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