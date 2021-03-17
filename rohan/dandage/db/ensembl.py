### Ensembl
# for human genome
# release 77 uses human reference genome GRCh38
# from pyensembl import EnsemblRelease
# EnsemblRelease(release=100)
# for many other species
# ensembl = pyensembl.EnsemblRelease(species=pyensembl.species.Species.register(
# latin_name='saccharomyces_cerevisiae',
# synonyms=['saccharomyces_cerevisiae'],
# reference_assemblies={
#     'R64-1-1': (92, 92),
# }),release=92)
import numpy as np
import pandas as pd
import logging
import requests, sys

#pyensembl faster
def ensg2genename(id,ensembl):
    try:
        return ensembl.gene_name_of_gene_id(id)
    except:
        return np.nan

def genename2ensg(name,ensembl):
    try:
        names=ensembl.gene_ids_of_gene_name(name)
        if len(names)>1:
            logging.warning('more than one ids')
            return '; '.join(names)
        else:
            return names[0]
    except:
        return np.nan
    
def enst2ensp(id,ensembl):
    try:
        t=ensembl.transcript_by_id(id)
        return t.protein_id
    except:
        return np.nan    
    
def enst2ensg(id,ensembl):
    try:
        t=ensembl.transcript_by_id(id)
        return t.gene_id
    except:
        return np.nan 
    
def ensp2enst(id,ensembl):
    try:
        return ensembl.transcript_id_of_protein_id(id)
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
def enst2cdsseq(id,ensembl):
    try:
        t=ensembl.transcript_by_id(id)
        return t.coding_sequence
    except:
        return np.nan 
    
def get_utr_sequence(ensembl,x,loc='five'):
    try:
        t=ensembl.transcript_by_protein_id(x)
        return getattr(t,f'{loc}_prime_utr_sequence')
    except: 
        logging.warning(f"{x}: no sequence found")
        return     
    
def protein_id2transcript_id(protein_id,ensembl):    
    if (protein_id in ensembl.protein_ids() and (not pd.isnull(protein_id))):
        return ensembl.transcript_by_protein_id(protein_id).transcript_id
    else:
        return np.nan    

def is_protein_coding(x,ensembl,geneid=True):
    try:
        if geneid:
            g=ensembl.gene_by_id(x)
        else:
            g=ensembl.transcript_by_id(x)
    except:
        return 'gene id not found'
    return g.is_protein_coding

def get_gene_name(x,ensembl):
    try:
        return ensembl.gene_name_of_gene_id(x)
    except:
        return np.nan
#restful api    
import requests, sys    
def ensembl_rest(id_,function,headers={'target_taxon':'9606',
                                               'release':95,
                                          "format":"full",
                                          "Content-Type" : "application/json",},test=False):
    """
    param fmt: id sequence homology
    
    https://rest.ensembl.org/sequence/id/ENSP00000288602?content-type=application/json    
    """
    server = "https://rest.ensembl.org"
    ext = f"/{function}/id/{id_}?"
    r = requests.get(server+ext, headers=headers)
    if test:
        print(f"{server}/{ext}?format=full;content-type=application/json")
    if r.ok:
        return r.json()            

    
def gene_id2homology(gene_id,headers={'target_taxon':'9606',
                                          "type":"paralogues",
                                          "format":"full",
                                          "Content-Type" : "application/json",},test=False):
    server = "https://rest.ensembl.org"
    ext = f"/homology/id/{gene_id}?"
    r = requests.get(server+ext, headers=headers)
    if test:
        print(r.url)
    if r.ok:
        return r.json()
    
def ensembl_lookup(id_,headers={'target_taxon':'9606',
                                               'release':95,
                                          "format":"full",
                                          "Content-Type" : "application/json",},test=False):
    """
    prefer ensembl rest
    to be deprecated
    https://rest.ensembl.org/lookup/id/ENSP00000351933?target_taxon=9606;release=95;content-type=application/json    
    """
    return ensembl_rest(id_,function='lookup',headers=headers,test=test)

def taxid2name(k):
    server = "https://rest.ensembl.org"
    ext = f"/taxonomy/id/{k}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    decoded = r.json()
    return decoded['scientific_name']

def taxname2id(k):
    server = "https://rest.ensembl.org"
    ext = f"/taxonomy/name/{k}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok or r.status_code==400:
        logging.warning(f'no tax id found for {k}')
        return 
    decoded = r.json()
    if len(decoded)!=0:
        return decoded[0]['id']
    else:
        logging.warning(f'no tax id found for {k}')
        return

def convert_coords_human_assemblies(chrom,start,end,frm=38,to=37):
    import requests, sys 
    server = "http://rest.ensembl.org"
    ext = f"/map/human/GRCh{frm}/{chrom}:{start}..{end}:1/GRCh{to}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    decoded = r.json()
    d=eval(repr(decoded))
    if 'mappings' in d:
        for d_ in d['mappings']:
            if 'mapped' in d_:
#                 return d_['mapped']['seq_region_name'],d_['mapped']['start'],d_['mapped']['end']
                return pd.Series(d_['mapped'])#['seq_region_name'],d_['mapped']['start'],d_['mapped']['end']
    
    