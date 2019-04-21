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