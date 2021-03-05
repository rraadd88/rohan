from rohan.dandage.io_dict import read_dict
def ebiid2uniprotids(k):
    d1=read_dict(f'https://www.ebi.ac.uk/ebisearch/ws/rest/intact-interactors/entry/{k}/xref/uniprot?format=json')
    assert(len(d1['entries'])==1)
    return [d['acc'] for d in d1['entries'][0]['references']]
