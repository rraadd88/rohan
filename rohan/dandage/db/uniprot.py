import requests
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