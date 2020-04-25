import requests
import logging

def goid2name(queries,result='name',interval=500):
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
        ds.append({d['id']:d[result] for d in responseBody['results']})
    from rohan.dandage.io_dict import merge_dict_list
    return merge_dict_list(ds)
    
get_go_info=goid2name

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
        queries_str=str2urlformat(','.join(queries))
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

def get_goterms_physical_interactions(tax_id,outp):
    import requests, sys
    requestURL = f"https://www.ebi.ac.uk/QuickGO/services/annotation/downloadSearch?taxonId={tax_id}&taxonUsage=exact&geneProductSubset=Swiss-Prot&proteome=gcrpCan,gcrpIso,complete&evidenceCode=ECO:0000314,ECO:0000353,ECO:0007005&evidenceCodeUsage=descendants&qualifier=part_of,involved_in&downloadLimit=50000&geneProductType=protein"
    r = requests.get(requestURL, headers={"Accept" : "text/gpad"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    responseBody = r.text
    if not outp is None:
        makedirs(dirname(outp),exist_ok=True)
        with open(outp,'w') as f:
            f.write(responseBody)
        return outp
    else:
        return responseBody
