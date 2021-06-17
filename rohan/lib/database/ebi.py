from rohan.lib.io_dict import read_dict
def ebiid2uniprotids(k):
    """
    
    """
    d1=read_dict(f'https://www.ebi.ac.uk/ebisearch/ws/rest/intact-interactors/entry/{k}/xref/uniprot?format=json')
    assert(len(d1['entries'])==1)
    return [d['acc'] for d in d1['entries'][0]['references']]

## protein structures
def get_pos(d2,pdbid,uniprotid):
    """
    Sifts a dataframe.
    """
    a=np.array([[d2[pdbid]['UniProt'][uniprotid]['mappings'][0]['start'][k] for k in d2[pdbid]['UniProt'][uniprotid]['mappings'][0]['start'] if 'number' in k],
    [d2[pdbid]['UniProt'][uniprotid]['mappings'][0]['end'][k] for k in d2[pdbid]['UniProt'][uniprotid]['mappings'][0]['start'] if 'number' in k]])

    ## lengths
    l1=flatten(list(np.diff(a.T)+1)+[len([r for r in model[d1['chain_id']].get_residues()])])
    assert(nunique(l1)==1)
    # d2[pdbid]['UniProt'][uniprotid]['mappings'][0]['start']#['author_residue_number'],
    # d2[pdbid]['UniProt'][uniprotid]['mappings'][0]['end']['author_residue_number'],
    df1=pd.DataFrame({"position uniprot":range(d1['unp_start'],d1['unp_end']+1,1),
                    # "position index":range(d1['start'],d1['end']+1,1),
                    "position author":range(d2[pdbid]['UniProt'][uniprotid]['mappings'][0]['start']['author_residue_number'],
                           d2[pdbid]['UniProt'][uniprotid]['mappings'][0]['end']['author_residue_number']+1,1)                 
                     })
    d3=model[d1['chain_id']].child_dict
    df1['aa author']=df1['position author'].map({k[1]:d3[k].resname for k in d3})
    return df1

def get_gene_description(pdbid):
    d5=read_dict(f"https://www.ebi.ac.uk/pdbe/api/pdb/entry/molecules/{pdbid}")
    d6={}
    for d in d5[pdbid]:
        if not d['molecule_type'] in ['polyribonucleotide','polypeptide(L)']:
            continue
        if not ('in_chains' in d and 'molecule_name' in d):
            continue        
        if not (len(d['in_chains'])==1 and len(d['molecule_name'])==1):
            continue        
        if d['in_chains'][0] in d6:
            print(d['in_chains'][0])
        d6[d['in_chains'][0]]=d['molecule_name'][0] 
    #         break
    return d6