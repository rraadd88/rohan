from rohan.global_imports import *

## relative 
def get_chr_loc_type(l,ignore=['MT']):
    if nintersection([l,ignore])!=0:
        return '|'.join(ignore)
    elif nunique(l)==2:
        return 'different'   
    else:
        return 'same'

## coords (ranges)
from rohan.lib.io_sets import range_overlap

def get_direction():
    """
    TODO get transcription direction
    """
    if x['gene1']['strand']==x['gene2']['strand']:
        d1['direction']="3'-5'"
#             if abs(x['gene2']['start']-x['gene1']['end']) > abs(x['gene1']['start']-x['gene2']['end']):
        if x['gene1']['end']>x['gene2']['start']:
            # gene2 -- gene1
            d1['distance']= x['gene1']['end']-x['gene2']['start']
            d1['genes inbetween list']=ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                          position=x['gene2']['start'], end=x['gene1']['end'], strand=None)
        elif x['gene1']['end']<x['gene2']['start']:
            # gene1 -- gene2 
            d1['distance']=x['gene2']['end']-x['gene1']['start']
            d1['genes inbetween list']=ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                          position=x['gene1']['start'], end=x['gene2']['end'], strand=None)
        else:
            d1['direction']="not 3'-5'?"
            d1['distance']=0
            d1['genes inbetween list']=[]
    else:
        if abs(x['gene1']['start'] - x['gene2']['start']) < abs(x['gene1']['end'] - x['gene2']['end']):
            d1['direction']="5'-5'"
            d1['distance']=abs(x['gene1']['start'] - x['gene2']['start'])
            d1['genes inbetween list']=ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                                                    position=min([x['gene1']['start'] , x['gene2']['start']]), 
                                                                    end=max([x['gene1']['end'] , x['gene2']['end']]), 
                                                                    strand=None)

        elif abs(x['gene1']['start'] - x['gene2']['start']) > abs(x['gene1']['end'] - x['gene2']['end']):
            d1['direction']="3'-3'"        
            d1['distance']=abs(x['gene1']['end'] - x['gene2']['end'])
            d1['genes inbetween list']=ensembl.gene_ids_at_locus(contig=min([x['gene1']['end'] , x['gene2']['end']]),
                                                                    position=max([x['gene1']['start'] , x['gene2']['start']]), 
                                                                    end=x['gene1']['start'], strand=None)
        else:
            d1['direction']="not 5'-5' or 3'-3'?"
            d1['distance']=0
            d1['genes inbetween list']=[]

def get_location_relative(k1,k2,ensembl,test=False):
    """
    chromosome: same| different
    if same
        distance : bp, genes
    """
    d1={}
    try:
        x={f'gene{i+1}': ensembl.locus_of_gene_id(k).to_dict() for i,k in enumerate([k1,k2])}
    except:
        logging.warning(f"ids not found: {k1}|{k2}")
        return d1
    if test: print(x)
    d1['chromosomes']=get_chr_loc_type([x['gene1']['contig'],x['gene2']['contig']])
    if d1['chromosomes']=='same':
        overlap=range_overlap([x['gene1']['start'],x['gene1']['end']],
                              [x['gene2']['start'],x['gene2']['end']])
        d1['overlap']=len(overlap)!=0
        if d1['overlap']:
            d1['distance bp']=-1*len(overlap)
            d1['distance genes list']=ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                                                  position=min([overlap[0],overlap[1]]), 
                                                                  end=max([overlap[0],overlap[1]]),
                                                                  strand=None)
        else:
            ## average coords
            x1,x2=sorted([int(np.mean([x['gene1']['start'],x['gene1']['end']])),
                          int(np.mean([x['gene2']['start'],x['gene2']['end']]))])
            d1['distance bp']=x2-x1
            d1['distance genes list']=ensembl.gene_ids_at_locus(contig=x['gene1']['contig'],
                                                                  position=x1, 
                                                                  end=x2,
                                                                  strand=None)            
        d1['distance genes list']=[s for s in d1['distance genes list'] if not s in [k1,k2]]
        d1['distance genes']=nunique(d1['distance genes list'])            
        if d1['distance genes']==0:
            d1['chromosomes']='tandem' if not d1['overlap'] else 'overlapping'
    return d1


# #!usr/bin/python

# import sys
# from os.path import exists,splitext,dirname,splitext,basename,abspath
# from os import makedirs
# from glob import glob
# import pandas as pd
# import subprocess
# import logging
# from rohan.lib.io_sys import runbashcmd    

# #use good old bash programs for speed
# # bed_colns = ['chromosome','start','end','id','NM','strand']
# gff_colns = ['chromosome', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
# bed_colns = ['chromosome','start','end','id','NM','strand']


# def validate_cfg(cfg):
#     import pyensembl
#     cfg['host']=pyensembl.species.normalize_species_name(cfg['host'])        
#     cfg['scriptp']=abspath(__file__)
#     return cfg
    
# # GET INPTS    
# def get_genomes(cfg):
#     """
#     Installs genomes
    
#     :param cfg: configuration dict
##     """
#     if cfg['host']=='homo_sapiens':
#         contigs=['1',
#                  '2',
#                  '3',
#                  '4',
#                  '5',
#                  '6',
#                  '7',
#                  '8',
#                  '9',
#                  '10',
#                  '11',
#                  '12',
#                  '13',
#                  '14',
#                  '15',
#                  '16',
#                  '17',
#                  '18',
#                  '19',
#                  '20',
#                  '21',
#                  '22',
#                  'X','Y']
#     else:
#         runbashcmd(f"pyensembl install --reference-name {cfg['genomeassembly']} --release {cfg['genomerelease']} --species {cfg['host']}")

#         import pyensembl
#         ensembl = pyensembl.EnsemblRelease(species=pyensembl.species.Species.register(
#                                             latin_name=cfg['host'],
#                                             synonyms=[cfg['host']],
#                                             reference_assemblies={
#                                                 cfg['genomeassembly']: (cfg['genomerelease'], cfg['genomerelease']),
#                                             }),release=cfg['genomerelease'])
#         contig_mito=['MTDNA','MITO','MT']
#         contigs=[c for c in ensembl.contigs() if ((not '.' in c) and (len(c)<5) and (c not in contig_mito))]    
#     if len(contigs)==0:
#         contigs=[c for c in ensembl.contigs()]
#         # logging.error([c for c in ensembl.contigs()])
#         # logging.error('no contigs identified by pyensembl; aborting')
#         # sys.exit(0)
#     logging.info(f"{len(contigs)} contigs/chromosomes in the genome")
#     logging.info(contigs)
#     # raw genome next
#     if 'human' in cfg['host'].lower():
#         cfg['host']='homo_sapiens'
#     if 'yeast' in cfg['host'].lower():
#         cfg['host']='saccharomyces_cerevisiae'
#     host_="_".join(s for s in cfg['host'].split('_')).capitalize()
#     if 'GRCh37' in cfg['genomeassembly']:    
#         ensembl_fastad=f"pub/grch37/update/fasta/{cfg['host']}/dna/"
#     else:
#         ensembl_fastad=f"pub/release-{cfg['genomerelease']}/fasta/{cfg['host']}/dna/"        
#     genome_fastad=f"{cfg['genomed']}/{ensembl_fastad}"
#     cfg['genomep']=f'{genome_fastad}/genome.fa'.format()
#     if not exists(cfg['genomep']):
#         logging.error(f"not found: {cfg['genomep']}")
#         ifdlref = input(f"Genome file are not there at {genome_fastad}.\n Download?[Y/n]: ")
#         if ifdlref=='Y':
#         # #FIXME download contigs and cat and get index, sizes
#             for contig in contigs:
#                 fn=f"{cfg['host'].capitalize()}.{cfg['genomeassembly']}.dna_sm.chromosome.{contig}.fa.gz"
#                 fp=f'{ensembl_fastad}/{fn}'
#                 if not exists(f"{cfg['genomed']}{fp.replace('.gz','')}"):
#                     if not exists(f"{cfg['genomed']}{fp}"):
#                         logging.info(f'downloading {fp}')
#                         cmd=f"wget -q -x -nH ftp://ftp.ensembl.org/{fp} -P {cfg['genomed']}"
#                         try:
#                             runbashcmd(cmd,test=cfg['test'])
#                         except:
#                             fn=f"{cfg['host'].capitalize()}.{cfg['genomeassembly']}.dna_sm.toplevel.fa.gz"
#                             fp='{}/{}'.format(ensembl_fastad,fn)                        
#                             if not exists(fp):
#                                 cmd=f"wget -q -x -nH ftp://ftp.ensembl.org/{fp} -P {cfg['genomed']}"
#                                 # print(cmd)
#                                 runbashcmd(cmd,test=cfg['test'])
#                                 break
#                 #                 break
#             # make the fa ready
#             if not exists(cfg['genomep']):
#                 cmd=f"gunzip {genome_fastad}*.fa.gz;cat {genome_fastad}/*.fa > {genome_fastad}/genome.fa;"
#                 runbashcmd(cmd,test=cfg['test'])
#         else:
#             logging.error('abort')
#             sys.exit(1)
#     if not exists(cfg['genomep']+'.bwt'):
#         cmd=f"{cfg['bwa']} index {cfg['genomep']}"
#         runbashcmd(cmd,test=cfg['test'])
#     else:        
#         logging.info('bwa index is present')
#     if not exists(cfg['genomep']+'.fai'):
#         cmd=f"{cfg['samtools']} faidx {cfg['genomep']}"
#         runbashcmd(cmd,test=cfg['test'])
#     else:
#         logging.info('samtools index is present')
#     if not exists(cfg['genomep']+'.sizes'):
#         cmd=f"cut -f1,2 {cfg['genomep']}.fai > {cfg['genomep']}.sizes"            
#         runbashcmd(cmd,test=cfg['test'])
#     else:
#         logging.info('sizes of contigs are present')

#     if 'GRCh37' in cfg['genomeassembly']:    
# #     ftp://ftp.ensembl.org/pub/grch37/update/gff3/homo_sapiens/Homo_sapiens.GRCh37.87.gff3.gz
#         ensembl_gff3d=f"pub/grch37/update/gff3/{cfg['host']}/"    
#     else:
#         ensembl_gff3d=f"pub/release-{cfg['genomerelease']}/gff3/{cfg['host']}/"    
    
#     genome_gff3d=f"{cfg['genomed']}/{ensembl_gff3d}"
#     cfg['genomegffp']=f'{genome_gff3d}/genome.gff3'
#     if not exists(cfg['genomegffp']):
#         logging.error('not found: {}'.format(cfg['genomegffp']))
#         ifdlref = input("Download genome annotations at {}?[Y/n]: ".format(genome_gff3d))
#         if ifdlref=='Y':
#         # #FIXME download contigs and cat and get index, sizes
#             fn=f"{cfg['host'].capitalize()}.{cfg['genomeassembly']}.{cfg['genomerelease']}.gff3.gz"
#             fp=f"{ensembl_gff3d}/{fn}"
#             if not exists(fp):
#                 cmd=f"wget -x -nH ftp://ftp.ensembl.org/{fp} -P {cfg['genomed']}"
#                 runbashcmd(cmd,test=cfg['test'])
#             # move to genome.gff3
#                 cmd=f"cp {genome_gff3d}/{fn} {cfg['genomegffp']}"
#                 runbashcmd(cmd,test=cfg['test'])

#         else:
#             logging.error('abort')
#             sys.exit(1)
#     logging.info('genomes are installed!')
#     return cfg

