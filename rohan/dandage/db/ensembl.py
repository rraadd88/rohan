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
from rohan.global_imports import *
import numpy as np
import pandas as pd
import logging
import requests, sys

#pyensembl faster
def gid2gname(id,ensembl):
    try:
        return ensembl.gene_name_of_gene_id(id)
    except:
        return np.nan

def gname2gid(name,ensembl):
    try:
        names=ensembl.gene_ids_of_gene_name(name)
        if len(names)>1:
            logging.warning('more than one ids')
            return '; '.join(names)
        else:
            return names[0]
    except:
        return np.nan
    
def tid2pid(id,ensembl):
    try:
        t=ensembl.transcript_by_id(id)
        return t.protein_id
    except:
        return np.nan    
    
def tid2gid(id,ensembl):
    try:
        t=ensembl.transcript_by_id(id)
        return t.gene_id
    except:
        return np.nan 
    
def pid2tid(id,ensembl):
    try:
        return ensembl.transcript_id_of_protein_id(id)
    except:
        return np.nan    
    
def gid2dnaseq(id,ensembl):
    try:
        g=ensembl.gene_by_id(id)
        ts=g.transcripts
        lens=[len(t.protein_sequence) if not t.protein_sequence is None else 0 for t in ts]
        return ts[lens.index(max(lens))].id, ts[lens.index(max(lens))].protein_sequence
    except:
        return np.nan,np.nan     
def tid2prtseq(id,ensembl):
    try:
        t=ensembl.transcript_by_id(id)
        return t.protein_sequence
    except:
        return np.nan
def pid2prtseq(id,ensembl,
               length=False):
    try:
        t=ensembl.protein_sequence(id)
        if not length:
            return t
        else:
            return len(t)            
    except:
        return np.nan    
    
def tid2cdsseq(id,ensembl):
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
    
def pid2tid(protein_id,ensembl):    
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

#restful api    
def rest(ids,function='lookup',
                 target_taxon='9606',
                 release='100',
                 format_='full',
                 test=False,
                 **kws):
    import requests, sys

    server = f"https://e{release}.rest.ensembl.org"
    ext = f"/{function}/id"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    
    headers['target_taxon']=target_taxon
    headers['release']=release
    headers['format']=format_
    headers.update(kws)
    if test: print(headers)
        
    if isinstance(ids,str):
        r = requests.get(server+ext+f'{ids}?', headers=headers)
    elif isinstance(ids,list):
        r = requests.post(server+ext, headers=headers, 
                          data='{ "ids" : ['+', '.join([f'"{s}"' for s in ids])+' ] }')
    else:
        raise ValueError(f"ids should be str or list")
    if not r.ok:
        r.raise_for_status()
    else:
        decoded = r.json()
    #     print(repr(decoded))
        return decoded

def geneid2homology(x='ENSG00000148584',
                    release=100,
                    homologytype='orthologues',
                   outd='data/database',
                   force=False):
    """
    # outp='data/database/'+replacemany(p.split(';content-type')[0],{'https://':'','?':'/',';':'/'})+'.json'
    Ref: f"https://e{release}.rest.ensembl.org/documentation/info/homology_ensemblgene
    """
    p=f"https://e{release}.rest.ensembl.org/homology/id/{x}?type={homologytype};compara=vertebrates;sequence=none;cigar_line=0;content-type=application/json;format=full"
    outp=outp=f"{outd}/{p.replace('https://','')}.json"
    if exists(outp) and not force:
        return read_dict(outp)
    else:
        d1=read_dict(p)
        to_dict(d1,outp)
    return d1

def proteinid2domains(x,
                    release,
                     outd='data/database',
                     force=False):
    """
    """
    p=f'https://e{release}.rest.ensembl.org/overlap/translation/{x}?content-type=application/json;species=homo_sapiens;feature=protein_feature;type=pfam'
    outp=outp=f"{outd}/{p.replace('https://','')}.json"
    if exists(outp) and not force:
        d1=read_dict(outp)
    else:
        d1=read_dict(p)
        to_dict(d1,outp)
    if d1 is None: return
    if len(d1)==0:
        logging.error(x)
        return
    #d1 is a list
    return pd.concat([pd.DataFrame(pd.Series(d)).T for d in d1],
                     axis=0)



## species
def taxid2name(k):
    server = f"https://e{release}.rest.ensembl.org"
    ext = f"/taxonomy/id/{k}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    decoded = r.json()
    return decoded['scientific_name']

def taxname2id(k):
    server = f"https://e{release}.rest.ensembl.org"
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
    
## convert between assemblies    
def convert_coords_human_assemblies(release,chrom,start,end,
                                    frm=38,to=37):
    import requests, sys 
    server = f"https://e{release}.rest.ensembl.org"
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
    
## convert coords 
def coords2geneid(x,
                 biotype='protein_coding'):
    # x=df02.iloc[0,:]
    from pyensembl import EnsemblRelease
    ensembl=EnsemblRelease(release=100)
    contig,pos=x['Genome Location'].split(':')

    start,end=[int(s) for s in pos.split('-')]

    # l1=ensembl.gene_ids_at_locus
    l1=ensembl.genes_at_locus(contig=contig,
                              position=start, 
                              end=end, 
                              strand=None)

    # def range_overlap(l1,l2):
    #     return set.intersection(set(range(l1[0],l1[1]+1,1)),
    #                             set(range(l2[0],l2[1]+1,1)))
#     ds1=pd.Series({ 
    d1={}
    for g in l1:
        if g.biotype==biotype:
            d1[g.gene_id]=len(range_overlap([g.start,g.end],[start,end]))
    ds1=pd.Series(d1).sort_values(ascending=False)
    print(ds1)
    return ds1.index[0]

## idmapper
def read_idmapper_(s,ini,end,sep=', '):
    d1=dict(data=[x.split(sep) for x in s.split('\n')[1:]],
            columns=s.split('\n')[0].split(sep),)
    if np.shape(d1['data'])[1]!=np.shape(d1['columns'])[0]:
        if np.shape(d1['data'])[0]==1:
            k='New stable ID'
            return {f'{k} release={ini}':d1['data'][0][0],
                    f'{k} release={end}':d1['data'][0][1],
                   }
        else:
            print(s)
#         d1['columns']=d1['columns'][:np.shape(d1['data'])[1]]
    df1=pd.DataFrame(**d1)
    df1['Release']=df1['Release'].astype(float)
    df1=df1.sort_values('Release')
    def get_dict(df1,i):
        df_=df1.loc[(df1['Release']<=i),:].tail(1)
        if len(df_)!=0:
            x=df_.iloc[0,:]
        else:
            x=df1.loc[(df1['Release']>=i),:].head(1).iloc[0,:]
#             print(x)
            x=x.drop(['New stable ID','Release','Mapping score'],axis=0)
            x=x.rename(index={'Old stable ID':'New stable ID'},axis=0)
        if 'Old stable ID' in x.index:
            x=x.drop(['Old stable ID'],axis=0)
        x=x.add_suffix(f', release={i}')
        return x.to_dict()
#         {f'{k} release={i}':df1.loc[(df1['Release']>=i),:].head(1).iloc[0,:][k] for i in [ini,end] for k in ['Old stable ID','Release','Mapping score']}
    d2=get_dict(df1,ini)
    d3=get_dict(df1,end)
#     d2={f"{k} release" for k in d2}
    d2.update(d3)
    return d2

def read_idmapper(outd=None,ini=75,
                    end=100,
                 ids=None,force=False,
                 idtype='gene'):
    """
    :params ps: glob(f'{outd}/*.idmapper.txt')
    """
    ps=glob(f'{outd}/*.idmapper.txt')
    ic(ps)
    outp=f"{outd}/{ini}_{end}.tsv"
    if exists(outp) and not force:
        return read_table(outp)
    l1=[]
    for p in ps:
        lines=open(p,'r').read().split('\n\n')
        df2=pd.DataFrame([read_idmapper_(s,ini,end) for s in tqdm(lines) if '\n' in s])
        l1.append(df2)
    df3=pd.concat(l1,axis=0)
    for i in [ini,end]:
        df3[f'{idtype} id, release={i}']=df3[f'New stable ID, release={i}'].str.split('.',expand=True)[0]
    info(f"retired ids={sum(~((df3[f'{idtype} id, release={ini}']!='<retired>') & (df3[f'{idtype} id, release={end}']!='<retired>')))}")
    df3=df3.loc[((df3[f'{idtype} id, release={ini}']!='<retired>') & (df3[f'{idtype} id, release={end}']!='<retired>')),:]
    if not ids is None:
        assert(len(set(df3[f'{idtype} id, release={ini}'].tolist()) - set(ids))==0)
    df3=df3.loc[:,[f'{idtype} id, release={ini}',
                   f'{idtype} id, release={end}']].rd.get_mappings(keep='m:m')
    info(df3['mapping'].value_counts())
    to_table(df3,outp,colgroupby='mapping')
    ic(outp)
    return df3

def check_release(ids,release,p,
                 ):
    """
    :params ids:
    :params p: database  
    """
    idtype=basename(dirname(dirname(p)))
    l2=read_table(p)[f'{idtype} id, release={release}'].tolist()
    ic(jaccard_index(ids,l2))
    return len(set(l2) - set(ids))==0

# to be deprecated
def read_idmapper_results(ps):
    """
    :params ps:glob(f'{outd}/Results-Homo_sapiens_Tools_IDMapper_*.csv')
    Note: deprecated becase, it maps to the latest release.
    """
    return pd.concat([read_table(p).loc[:,['Requested ID','Matched ID(s)']] for p in ps],
             axis=0)#.log.drop_duplicates()

## idmapper results
def map_ids_by_release(ids,outd,idtype='gene', force=False):
    """
    uses idmapper
    TODO: use `convert_coords_human_assemblies` + `pyensembl`
    Note: deprecated becase, it maps to the latest release.
    """
    chunks=int(np.ceil(len(ids)/10000))
    if not exists(outd+'.zip') or force:
        makedirs(outd,exist_ok=True)
        for i,l in enumerate(np.array_split(ids, chunks)):
            with open(f'{outd}/{i}.txt','w') as f:
                f.write('\n'.join(l))
        zip_folder(outd, outd+'.zip')
        return
    ps=glob(f'{outd}/Results-Homo_sapiens_Tools_IDMapper_*.csv')
    assert(chunks==len(ps))
    df=read_idmapper_results(ps)
    df=df.rd.get_mappings(cols=None,keep='1:1')
    info(df.rd.check_mappings(list(df)))
    info(f"identical ids={(sum(df['Requested ID']==df['Matched ID(s)'])/len(df))*100}%")
    df=df.rename(columns={'Requested ID':f'{idtype} id (GRCh37)','Matched ID(s)':f'{idtype} id (GRCh38)'})
    to_table(df,f"{outd}.tsv")
    l=list(set(ids) - set(df[f'{idtype} id (GRCh37)']))
    info(f"ids not mapped = {len(l)}")
    to_table(pd.DataFrame({f'{idtype} id (GRCh37)':l}),f"{outd}/unmapped.tsv")
    return df    