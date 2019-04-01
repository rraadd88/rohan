import requests
def get_go_info(goterm,result='name'):
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
    response=requests.get(f'https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{goterm}')
    try:       
        return response.json()['results'][0][result]
    except:
        print(response.json())