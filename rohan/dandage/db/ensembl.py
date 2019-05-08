# ensembl = pyensembl.EnsemblRelease(species=pyensembl.species.Species.register(
# latin_name='saccharomyces_cerevisiae',
# synonyms=['saccharomyces_cerevisiae'],
# reference_assemblies={
#     'R64-1-1': (92, 92),
# }),release=92)
import numpy as np

def ensg2genename(id,ensembl):
    try:
        return ensembl.gene_name_of_gene_id(id)
    except:
        return np.nan

def genename2ensg(name,ensembl):
    try:
        return gene_ids_of_gene_name('KRT8P11')
    except:
        return np.nan
def enst2ensp(id,ensembl):
    try:
        t=ensembl.transcript_by_id(id)
        return t.protein_id
    except:
        return np.nan    
    
def ensg2dnaseq(id,ensembl):
    try:
        g=ensembl.gene_by_id(id)
        ts=g.transcripts
        lens=[len(t.protein_sequence) if not t.protein_sequence is None else 0 for t in ts]
        return ts[lens.index(max(lens))].id, ts[lens.index(max(lens))].protein_sequence
    except:
        return np.nan,np.nan     
def enst2prtseq(id,ensembl):
    try:
        t=ensembl.transcript_by_id(id)
        return t.protein_sequence
    except:
        return np.nan        
def enst2rnaseq(id,ensembl):
    try:
        t=ensembl.transcript_by_id(id)
        return t.coding_sequence
    except:
        return np.nan 
    
def gene_id2homology(gene_id,headers={'target_taxon':'9606',
                                          "type":"paralogues",
                                          "format":"full",
                                          "Content-Type" : "application/json",},test=False):
    import requests, sys
    server = "https://rest.ensembl.org"
    ext = f"/homology/id/{gene_id}?"
    r = requests.get(server+ext, headers=headers)
    if test:
        print(f"https://rest.ensembl.org/homology/id/{gene_id}?format=full;target_taxon=9606;type=paralogues;content-type=application/json")
    if r.ok:
        return r.json()