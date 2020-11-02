from rohan.global_imports import *
import requests

def get_goid_info(queries,result=None,interval=500):
    """
    quickgo
    result: 'name','definition','synonyms'
    
    {'numberOfHits': 1,
 'results': [{'id': 'GO:0000006',
   'isObsolete': False,
   'name': 'high-affinity zinc transmembrane transporter activity',
   'definition': {'text': 'Enables the transfer of zinc ions (Zn2+) from one side of a membrane to the other, probably powered by proton motive force. In high-affinity transport the transporter is able to bind the solute even if it is only present at very low concentrations.',
    'xrefs': [{'dbCode': 'TC', 'dbId': '2.A.5.1.1'}]},
   'synonyms': [{'name': 'high-affinity zinc uptake transmembrane transporter activity',
     'type': 'related'},
    {'name': 'high affinity zinc uptake transmembrane transporter activity',
     'type': 'exact'}],
   'aspect': 'molecular_function',
   'usage': 'Unrestricted'}],
 'pageInfo': None}
    """
    if isinstance(queries,str):
        queries=[queries]
    import requests, sys
    ds=[]
    for ini,end in zip(range(0,len(queries)-1,interval),range(interval,len(queries)-1+interval,interval)):
        print(end,end=', ')
        queries_str='%2C'.join(queries[ini:end]).replace(':','%3A')
        requestURL = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{queries_str}"
        r = requests.get(requestURL, headers={ "Accept" : "application/json"})
        if not r.ok:
            r.raise_for_status()
            logging.error(f"check the list: {', '.join(queries[ini:end])}")
        responseBody = r.json()
        for di,d in enumerate(responseBody['results']):
            if not result is None:
                ds.append(pd.Series({k:d[k] for k in d if k in ['id',result]}))            
            else:
                ds.append(pd.Series({k:d[k] if (isinstance(d[k],(str,bool))) else d[k]['text'] for k in d if isinstance(d[k],(str,bool)) or (isinstance(d[k],(dict)) and k=='definition')}))
                
    return pd.concat(ds,axis=1).T
#     from rohan.dandage.io_dict import merge_dict_values
#     return merge_dict_values(ds)

def get_genes_bygoids(
    goids=['GO:0004713','GO:0004725'],
    taxid=559292):
    """
    geneProductSubset=Swiss-Prot&proteome=gcrpCan,complete&
    reference=PMID,DOI&
    """
    requestURL=f"https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?taxonId={taxid}&taxonUsage=exact&geneProductType=protein&goId={','.join(goids)}&goUsage=descendants&downloadLimit=50000"
    import requests
    # requestURL=
    r = requests.get(requestURL, headers={"Accept" : "text/gpad"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    responseBody = r.text
    from rohan.dandage.db.go import read_gpad
    from io import StringIO
    return read_gpad(StringIO(responseBody))                          
                          
def slim_goterms(queries,interval=500,subset='generic'):
    """
    "results": [
    {
      "slimsFromId": "GO:1905103",
      "slimsToIds": [
        "GO:0005764"
      ]
    },
    """
    subset2terms={   'generic':'GO:0140014,GO:0071941,GO:0071554,GO:0065003,GO:0061024,GO:0055085,GO:0051604,GO:0051301,GO:0051276,GO:0051186,GO:0050877,GO:0048870,GO:0048856,GO:0048646,GO:0044403,GO:0044281,GO:0043473,GO:0042592,GO:0042254,GO:0040011,GO:0040007,GO:0034655,GO:0034641,GO:0034330,GO:0032196,GO:0030705,GO:0030198,GO:0030154,GO:0022618,GO:0022607,GO:0021700,GO:0019748,GO:0016192,GO:0015979,GO:0015031,GO:0009790,GO:0009058,GO:0009056,GO:0008283,GO:0008219,GO:0007568,GO:0007267,GO:0007165,GO:0007155,GO:0007059,GO:0007049,GO:0007034,GO:0007010,GO:0007009,GO:0007005,GO:0006950,GO:0006914,GO:0006913,GO:0006810,GO:0006790,GO:0006629,GO:0006605,GO:0006520,GO:0006464,GO:0006457,GO:0006412,GO:0006399,GO:0006397,GO:0006259,GO:0006091,GO:0005975,GO:0003013,GO:0002376,GO:0000902,GO:0000278,GO:0000003,GO:0043226,GO:0032991,GO:0031410,GO:0031012,GO:0030312,GO:0009579,GO:0009536,GO:0005929,GO:0005886,GO:0005856,GO:0005840,GO:0005829,GO:0005815,GO:0005811,GO:0005794,GO:0005783,GO:0005777,GO:0005773,GO:0005768,GO:0005764,GO:0005739,GO:0005737,GO:0005730,GO:0005694,GO:0005654,GO:0005635,GO:0005634,GO:0005623,GO:0005622,GO:0005618,GO:0005615,GO:0005576,GO:0000229,GO:0000228,GO:0051082,GO:0043167,GO:0042393,GO:0032182,GO:0030674,GO:0030555,GO:0030533,GO:0030234,GO:0022857,GO:0019899,GO:0019843,GO:0016887,GO:0016874,GO:0016853,GO:0016829,GO:0016810,GO:0016798,GO:0016791,GO:0016779,GO:0016765,GO:0016757,GO:0016746,GO:0016491,GO:0016301,GO:0008289,GO:0008233,GO:0008168,GO:0008135,GO:0008134,GO:0008092,GO:0005198,GO:0004518,GO:0004386,GO:0003924,GO:0003735,GO:0003729,GO:0003723,GO:0003700,GO:0003677'
             }
    slimedterms= subset2terms[subset] if subset in  subset2terms else subset
    from rohan.dandage.io_strs import str2urlformat
    slimedterms_str=str2urlformat(slimedterms)
    import requests, sys
    ds=[]
    for ini,end in zip(range(0,len(queries)-1,interval),range(interval,len(queries)-1+interval,interval)):
        print(end,end=', ')
        queries_str=str2urlformat(','.join(queries[ini:end]))
        requestURL = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/slim?slimsToIds={slimedterms_str}&slimsFromIds={queries_str}&relations=is_a%2Cpart_of%2Coccurs_in%2Cregulates"
        r = requests.get(requestURL, headers={ "Accept" : "application/json"})
        if not r.ok:
            r.raise_for_status()
            logging.error(f"check the list: {', '.join(queries[ini:end])}")
        responseBody = r.json()
        ds.append({d["slimsFromId"]:d["slimsToIds"] for d in responseBody['results']})
    from rohan.dandage.io_dict import merge_dict_list
    from rohan.dandage.io_dict import dict2df
    df=dict2df(merge_dict_list(ds))
    return df.rename(columns={'key':'go id','value':'go id slimmed'})

def read_gpad(outp,params_read_table={}):
    cols_gpad=['DB','DB Object ID','Qualifier','GO ID','DB:Reference(s) (|DB:Reference)','Evidence Code','With (or) From','Interacting taxon ID','Date','Assigned by','Annotation Extension','Annotation Properties']
    df1=pd.read_table(outp,
                  names=cols_gpad,
                 comment='!',
                 **params_read_table
                 )
    return df1

def get_goterms_physical_interactions(taxid,outd):
    import requests, sys
    dn2df={}
    for aspect in ['molecular_function','biological_process','cellular_component']:
        requestURL = f"https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?aspect={aspect}&taxonId={taxid}&taxonUsage=exact&geneProductSubset=Swiss-Prot&proteome=gcrpCan,complete&geneProductType=protein&reference=PMID,DOI&goId=GO:0000228,GO:0000229,GO:0005576,GO:0005615,GO:0005618,GO:0005622,GO:0005623,GO:0005634,GO:0005635,GO:0005654,GO:0005694,GO:0005730,GO:0005737,GO:0005739,GO:0005764,GO:0005768,GO:0005773,GO:0005777,GO:0005783,GO:0005794,GO:0005811,GO:0005815,GO:0005829,GO:0005840,GO:0005856,GO:0005886,GO:0005929,GO:0009536,GO:0009579,GO:0030312,GO:0031012,GO:0031410,GO:0032991,GO:0043226,GO:0003677,GO:0003700,GO:0003723,GO:0003729,GO:0003735,GO:0003924,GO:0004386,GO:0004518,GO:0005198,GO:0008092,GO:0008134,GO:0008135,GO:0008168,GO:0008233,GO:0008289,GO:0016301,GO:0016491,GO:0016746,GO:0016757,GO:0016765,GO:0016779,GO:0016791,GO:0016798,GO:0016810,GO:0016829,GO:0016853,GO:0016874,GO:0016887,GO:0019843,GO:0019899,GO:0022857,GO:0030234,GO:0030533,GO:0030555,GO:0030674,GO:0032182,GO:0042393,GO:0043167,GO:0051082,GO:0000003,GO:0000278,GO:0000902,GO:0002376,GO:0003013,GO:0005975,GO:0006091,GO:0006259,GO:0006397,GO:0006399,GO:0006412,GO:0006457,GO:0006464,GO:0006520,GO:0006605,GO:0006629,GO:0006790,GO:0006810,GO:0006913,GO:0006914,GO:0006950,GO:0007005,GO:0007009,GO:0007010,GO:0007034,GO:0007049,GO:0007059,GO:0007155,GO:0007165,GO:0007267,GO:0007568,GO:0008219,GO:0008283,GO:0009056,GO:0009058,GO:0009790,GO:0015031,GO:0015979,GO:0016192,GO:0019748,GO:0021700,GO:0022607,GO:0022618,GO:0030154,GO:0030198,GO:0030705,GO:0032196,GO:0034330,GO:0034641,GO:0034655,GO:0040007,GO:0040011,GO:0042254,GO:0042592,GO:0043473,GO:0044281,GO:0044403,GO:0048646,GO:0048856,GO:0048870,GO:0050877,GO:0051186,GO:0051276,GO:0051301,GO:0051604,GO:0055085,GO:0061024,GO:0065003,GO:0071554,GO:0071941,GO:0140014&goUsageRelationships=is_a,part_of,occurs_in&goUsage=descendants&evidenceCode=ECO:0007005,ECO:0000353,ECO:0000314&evidenceCodeUsage=descendants&qualifier=part_of,involved_in,enables&downloadLimit=50000"
        r = requests.get(requestURL, headers={"Accept" : "text/gpad"})
        if not r.ok:
            r.raise_for_status()
            sys.exit()
        responseBody = r.text
        outp=f"{outd}/{taxid}/{aspect}.gpad"
        makedirs(dirname(outp),exist_ok=True)
        with open(outp,'w') as f:
            f.write(responseBody)
        df=read_gpad(outp)
        if len(df)==50000:
            logging.warning('some terms are cut off. keep it within the limit of 50k lines')
        if len(df)==0:
            logging.warning('no results. check the taxid')            
        # remove multi-org interactions
        dn2df[aspect]=df.loc[df['Interacting taxon ID'].isnull(),:]
    return pd.concat(dn2df,names=['aspect'],axis=0).reset_index()

def get_curated_goterms_physical_interactions(taxid,outd,force=False):
    dgenesets_smallp=f'{outd}/{taxid}/dgenesets_small.pqt'
    if exists(dgenesets_smallp) and not force:
        return read_table(dgenesets_smallp)
    df1=get_goterms_physical_interactions(taxid=taxid,outd=outd)
    df=slim_goterms(df1['GO ID'].unique().tolist(),interval=300)
    print(df['go id'].unique().shape,df['go id slimmed'].unique().shape)
    df2=df1.merge(df,left_on='GO ID',right_on='go id',how='left')
    print(df1.shape,df2.shape)
    from rohan.dandage.db.go import get_go_info
    genesetid2namep=f"{outd}/{taxid}/genesetid2name.json"
    if not exists(genesetid2namep):
        genesetid2name=goid2name(queries=unique(df2['GO ID'].tolist()+df2['go id slimmed'].tolist()),
                                 result='name',interval=500)
        to_dict(genesetid2name,genesetid2namep)
    else:
        genesetid2name=read_dict(genesetid2namep)
    df2['gene set name']=df2['GO ID'].map(genesetid2name)
    df2['gene set name slimmed']=df2['go id slimmed'].map(genesetid2name)
    from rohan.dandage.db.uniprot import map_ids_batch
    df=map_ids_batch(queries=df1['DB Object ID'].unique().tolist(),
                 params_map_ids={'frm': 'ACC', 'to': 'ENSEMBL_PRO_ID'},)
    df3=df2.merge(df,left_on='DB Object ID',right_on='ACC',how='left')
    to_table(df3,f'{outd}/{taxid}/dgenesets_raw.pqt')
    rename={
     'aspect':'gene set aspect', 
     'go id':'gene set id',
     'gene set name': 'gene set name',
     'go id slimmed':'gene set id slimmed',
     'gene set name slimmed': 'gene set name slimmed',
     'ENSEMBL_PRO_ID':'protein id',
     'ACC':'uniprot id',
    }
    print(df3.shape,end='')
    df4=df3.loc[:,rename.keys()].drop_duplicates().rename(columns=rename)
    print(df4.shape)
    to_table(df4,dgenesets_smallp)
    return df4
                          
                          
                          