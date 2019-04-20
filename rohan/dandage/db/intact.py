import numpy as np
def miid2descr(miid):
    import requests
    page=requests.get(f"https://www.ebi.ac.uk/ols/api/ontologies/mi/terms?iri=http://purl.obolibrary.org/obo/{miid.replace(':','_')}")
    if page.ok:    
        try:
            return page.json()['_embedded']['terms'][0]['annotation']['definition'][0]
        except:
            return np.nan