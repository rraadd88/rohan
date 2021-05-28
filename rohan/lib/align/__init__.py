#!usr/bin/python

import sys
from os.path import exists,splitext,dirname,splitext,basename,abspath
from os import makedirs
from glob import glob
import pandas as pd
import subprocess
import logging
from rohan.lib.io_sys import runbashcmd    

#use good old bash programs for speed
# bed_colns = ['chromosome','start','end','id','NM','strand']
gff_colns = ['chromosome', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes']
bed_colns = ['chromosome','start','end','id','NM','strand']


def validate_cfg(cfg):
    import pyensembl
    cfg['host']=pyensembl.species.normalize_species_name(cfg['host'])        
    cfg['scriptp']=abspath(__file__)
    return cfg
    
# GET INPTS    
def get_genomes(cfg):
    """
    Installs genomes
    
    :param cfg: configuration dict
    """
    if cfg['host']=='homo_sapiens':
        contigs=['1',
                 '2',
                 '3',
                 '4',
                 '5',
                 '6',
                 '7',
                 '8',
                 '9',
                 '10',
                 '11',
                 '12',
                 '13',
                 '14',
                 '15',
                 '16',
                 '17',
                 '18',
                 '19',
                 '20',
                 '21',
                 '22',
                 'X','Y']
    else:
        runbashcmd(f"pyensembl install --reference-name {cfg['genomeassembly']} --release {cfg['genomerelease']} --species {cfg['host']}")

        import pyensembl
        ensembl = pyensembl.EnsemblRelease(species=pyensembl.species.Species.register(
                                            latin_name=cfg['host'],
                                            synonyms=[cfg['host']],
                                            reference_assemblies={
                                                cfg['genomeassembly']: (cfg['genomerelease'], cfg['genomerelease']),
                                            }),release=cfg['genomerelease'])
        contig_mito=['MTDNA','MITO','MT']
        contigs=[c for c in ensembl.contigs() if ((not '.' in c) and (len(c)<5) and (c not in contig_mito))]    
    if len(contigs)==0:
        contigs=[c for c in ensembl.contigs()]
        # logging.error([c for c in ensembl.contigs()])
        # logging.error('no contigs identified by pyensembl; aborting')
        # sys.exit(0)
    logging.info(f"{len(contigs)} contigs/chromosomes in the genome")
    logging.info(contigs)
    # raw genome next
    if 'human' in cfg['host'].lower():
        cfg['host']='homo_sapiens'
    if 'yeast' in cfg['host'].lower():
        cfg['host']='saccharomyces_cerevisiae'
    host_="_".join(s for s in cfg['host'].split('_')).capitalize()
    if 'GRCh37' in cfg['genomeassembly']:    
        ensembl_fastad=f"pub/grch37/update/fasta/{cfg['host']}/dna/"
    else:
        ensembl_fastad=f"pub/release-{cfg['genomerelease']}/fasta/{cfg['host']}/dna/"        
    genome_fastad=f"{cfg['genomed']}/{ensembl_fastad}"
    cfg['genomep']=f'{genome_fastad}/genome.fa'.format()
    if not exists(cfg['genomep']):
        logging.error(f"not found: {cfg['genomep']}")
        ifdlref = input(f"Genome file are not there at {genome_fastad}.\n Download?[Y/n]: ")
        if ifdlref=='Y':
        # #FIXME download contigs and cat and get index, sizes
            for contig in contigs:
                fn=f"{cfg['host'].capitalize()}.{cfg['genomeassembly']}.dna_sm.chromosome.{contig}.fa.gz"
                fp=f'{ensembl_fastad}/{fn}'
                if not exists(f"{cfg['genomed']}{fp.replace('.gz','')}"):
                    if not exists(f"{cfg['genomed']}{fp}"):
                        logging.info(f'downloading {fp}')
                        cmd=f"wget -q -x -nH ftp://ftp.ensembl.org/{fp} -P {cfg['genomed']}"
                        try:
                            runbashcmd(cmd,test=cfg['test'])
                        except:
                            fn=f"{cfg['host'].capitalize()}.{cfg['genomeassembly']}.dna_sm.toplevel.fa.gz"
                            fp='{}/{}'.format(ensembl_fastad,fn)                        
                            if not exists(fp):
                                cmd=f"wget -q -x -nH ftp://ftp.ensembl.org/{fp} -P {cfg['genomed']}"
                                # print(cmd)
                                runbashcmd(cmd,test=cfg['test'])
                                break
                #                 break
            # make the fa ready
            if not exists(cfg['genomep']):
                cmd=f"gunzip {genome_fastad}*.fa.gz;cat {genome_fastad}/*.fa > {genome_fastad}/genome.fa;"
                runbashcmd(cmd,test=cfg['test'])
        else:
            logging.error('abort')
            sys.exit(1)
    if not exists(cfg['genomep']+'.bwt'):
        cmd=f"{cfg['bwa']} index {cfg['genomep']}"
        runbashcmd(cmd,test=cfg['test'])
    else:        
        logging.info('bwa index is present')
    if not exists(cfg['genomep']+'.fai'):
        cmd=f"{cfg['samtools']} faidx {cfg['genomep']}"
        runbashcmd(cmd,test=cfg['test'])
    else:
        logging.info('samtools index is present')
    if not exists(cfg['genomep']+'.sizes'):
        cmd=f"cut -f1,2 {cfg['genomep']}.fai > {cfg['genomep']}.sizes"            
        runbashcmd(cmd,test=cfg['test'])
    else:
        logging.info('sizes of contigs are present')

    if 'GRCh37' in cfg['genomeassembly']:    
#     ftp://ftp.ensembl.org/pub/grch37/update/gff3/homo_sapiens/Homo_sapiens.GRCh37.87.gff3.gz
        ensembl_gff3d=f"pub/grch37/update/gff3/{cfg['host']}/"    
    else:
        ensembl_gff3d=f"pub/release-{cfg['genomerelease']}/gff3/{cfg['host']}/"    
    
    genome_gff3d=f"{cfg['genomed']}/{ensembl_gff3d}"
    cfg['genomegffp']=f'{genome_gff3d}/genome.gff3'
    if not exists(cfg['genomegffp']):
        logging.error('not found: {}'.format(cfg['genomegffp']))
        ifdlref = input("Download genome annotations at {}?[Y/n]: ".format(genome_gff3d))
        if ifdlref=='Y':
        # #FIXME download contigs and cat and get index, sizes
            fn=f"{cfg['host'].capitalize()}.{cfg['genomeassembly']}.{cfg['genomerelease']}.gff3.gz"
            fp=f"{ensembl_gff3d}/{fn}"
            if not exists(fp):
                cmd=f"wget -x -nH ftp://ftp.ensembl.org/{fp} -P {cfg['genomed']}"
                runbashcmd(cmd,test=cfg['test'])
            # move to genome.gff3
                cmd=f"cp {genome_gff3d}/{fn} {cfg['genomegffp']}"
                runbashcmd(cmd,test=cfg['test'])

        else:
            logging.error('abort')
            sys.exit(1)
    logging.info('genomes are installed!')
    return cfg

## protein
diamond_blastp_cols=['query',
'subject',
'% identity',
'alignment length',
'# of mistmatches',
'gap openings',
'query start',
'query end',
'subject start',
'subject end',
'E-value',
'bit score']