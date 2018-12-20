#!usr/bin/python

import logging
import subprocess
import re
import sys
import logging 
logging.basicConfig(level=logging.INFO)
from os.path import join, basename, dirname, abspath, exists
from os import makedirs,stat

import pandas as pd
# import modin.pandas as pd

import pysam
import numpy as np
from glob import glob

from rohan.dandage.align import bed_colns,gff_colns    
from rohan.dandage.io_sys import runbashcmd
from rohan.dandage.io_seqs import fa2df,gffatributes2ids,hamming_distance,align 
from rohan.dandage.io_dfs import * 
from rohan.dandage.io_nums import str2num

def dqueries2queriessam(cfg,dqueries):    
    """
    Aligns queries to genome and gets SAM file
    step#1

    :param cfg: configuration dict
    :param dqueries: dataframe of queries
    """
    datatmpd=cfg['datatmpd']
    dqueries=set_index(dqueries,'query id')
    queryls=dqueries.loc[:,'query sequence'].apply(len).unique()
    for queryl in queryls:
        logging.debug(f"now aligning queries of length {queryl}")
        queriesfap = f'{datatmpd}/01_queries_queryl{queryl:02}.fa'
        logging.info(basename(queriesfap))
        if not exists(queriesfap) or cfg['force']:
            with open(queriesfap,'w') as f:
                for gi in dqueries.index:
                    f.write('>{}\n{}\n'.format(gi.replace(' ','_'),dqueries.loc[gi,'query sequence']))
        ## BWA alignment command is adapted from cripror 
        ## https://github.com/rraadd88/crisporWebsite/blob/master/crispor.py
        # BWA allow up to X mismatches
        # maximum number of occurences in the genome to get flagged as repeats. 
        # This is used in bwa samse, when converting the sam file
        # and for warnings in the table output.
        MAXOCC = 60000

        # the BWA queue size is 2M by default. We derive the queue size from MAXOCC
        MFAC = 2000000/MAXOCC

        genomep=cfg['genomep']
        genomed = dirname(genomep) # make var local, see below
        genomegffp=cfg['genomegffp']

        # increase MAXOCC if there is only a single query, but only in CGI mode
        bwaM = MFAC*MAXOCC # -m is queue size in bwa
        queriessap = f'{datatmpd}/01_queries_queryl{queryl:02}.sa'
        logging.info(basename(queriessap))
        if not exists(queriessap) or cfg['force']:
            cmd=f"{cfg['bwa']} aln -t 1 -o 0 -m {bwaM} -n {cfg['mismatches_max']} -k {cfg['mismatches_max']} -N -l {queryl} {genomep} {queriesfap} > {queriessap} 2> {queriessap}.log"
            runbashcmd(cmd)

        queriessamp = f'{datatmpd}/01_queries_queryl{queryl:02}.sam'
        logging.info(basename(queriessamp))        
        if not exists(queriessamp) or cfg['force']:
            cmd=f"{cfg['bwa']} samse -n {MAXOCC} {genomep} {queriessap} {queriesfap} > {queriessamp} 2> {queriessamp}.log"
            runbashcmd(cmd)
    return cfg

def queriessam2dalignbed(cfg):
    """
    Processes SAM file to get the genomic coordinates in BED format
    step#2

    :param cfg: configuration dict
    """
    datatmpd=cfg['datatmpd']
    alignmentbedp=cfg['alignmentbedp']
    dalignbedp=cfg['dalignbedp']
    logging.info(basename(dalignbedp))
    if not exists(alignmentbedp) or cfg['force']:
        #input/s
        queriessamps=glob(f'{datatmpd}/01_queries_queryl*.sam')
        for queriessamp in queriessamps:
            if stat(queriessamp).st_size != 0:
                samfile=pysam.AlignmentFile(queriessamp, "rb")
                dalignbed=pd.DataFrame(columns=bed_colns)
                for read in samfile.fetch():
                    algnids=[]
                    
#                     read_position=read.positions[0]
#                     tag_NM=read.get_tag('NM')                    
                    if len(read.positions)!=0:
                        read_position=read.positions[0]
                    else:
                        logging.error('no alignments found')
                        print(read)
                        continue
#                         read_position=[None]
                    try: 
                        tag_NM=read.get_tag('NM')
                    except: 
                        logging.error('no NM tag found')
                        print(read)
                        continue
#                         tag_NM=None
                    algnids.append(f"{read.reference_name}|{'-' if read.is_reverse else '+'}{read_position}|{read.cigarstring}|{tag_NM}")
                    if read.has_tag('XA'):
                        algnids+=['|'.join(s.split(',')) for s in read.get_tag('XA').split(';') if len(s.split(','))>1]
#                     print(len(algnids))
                    chroms=[]
                    starts=[]
                    ends=[]
                    algnids=algnids[:20]
                    NMs=[]
                    strands=[]
                    for a in algnids:
                        strand=a.split('|')[1][0]
                        chroms.append(a.split('|')[0])
                        if strand=='+':
                            offset=0
                        elif strand=='-':
                            offset=0                    
                        starts.append(int(a.split('|')[1][1:])+offset)
                        ends.append(int(a.split('|')[1][1:])+str2num(a.split('|')[2])+offset)
                        NMs.append(a.split('|')[3])
                        strands.append(strand)
                        del strand,offset
                    col2dalignbed={'chromosome':chroms,
                                   'start':starts,
                                   'end':ends,
                                   'id':algnids,
                                   'NM':NMs,
                                   'strand':strands}
                #     col2dalignbed=dict(zip(cols,[a.split('|')[0],a.split('|')[1],a.split('|')[2],a,a.split('|')[3],a.split('|')[4] for a in algnids]))
                    dalignbed_=pd.DataFrame(col2dalignbed)
                    dalignbed_['query id']=read.qname.replace('_',' ')
                    dalignbed = dalignbed.append(dalignbed_,ignore_index=True,sort=True)
                #     break
                samfile.close()
            else:
                logging.warning(f"file is empty: {queriessamp}")
        dalignbed.to_csv(dalignbedp,sep='\t')
        from rohan.dandage.io_nums import str2numorstr
        dalignbed['chromosome']=dalignbed.apply(lambda x : str2numorstr(x['chromosome']),axis=1)
        dalignbed=dalignbed.sort_values(['chromosome','start','end'], ascending=[True, True, True])
        dalignbed.loc[:,bed_colns].to_csv(alignmentbedp,sep='\t',
                        header=False,index=False,
                        chunksize=5000)
    return cfg

def dalignbed2annotationsbed(cfg):
    """
    Get annotations from the aligned BED file
    step#3

    :param cfg: configuration dict
    """
    datatmpd=cfg['datatmpd']
    alignmentbedp=cfg['alignmentbedp']    
    alignmentbedsortedp=alignmentbedp+'.sorted.bed'
    logging.info(basename(alignmentbedsortedp))
    if not exists(alignmentbedsortedp) or cfg['force']:
        cmd=f"{cfg['bedtools']} sort -i {alignmentbedp} > {alignmentbedsortedp}"
        runbashcmd(cmd)
    
    genomegffsortedp=cfg['genomegffp']+'.sorted.gff3.gz'
    logging.info(basename(genomegffsortedp))
    if not exists(genomegffsortedp):    
        cmd=f"{cfg['bedtools']} sort -i {cfg['genomegffp']} > {genomegffsortedp}"
        runbashcmd(cmd)

    annotationsbedp=f'{datatmpd}/03_annotations.bed'
    cfg['annotationsbedp']=annotationsbedp
    logging.info(basename(annotationsbedp))
    if not exists(annotationsbedp) or cfg['force']:    
        cmd=f"{cfg['bedtools']} intersect -wa -wb -loj -a {alignmentbedsortedp} -b {genomegffsortedp} > {annotationsbedp}"
        runbashcmd(cmd)
    return cfg

def dalignbed2dalignbedqueries(cfg):
    """
    Get query seqeunces from the BED file
    step#4

    :param cfg: configuration dict
    """
    datatmpd=cfg['datatmpd']
    dalignbed=del_Unnamed(pd.read_csv(cfg['dalignbedp'],sep='\t'))
    dqueries=set_index(del_Unnamed(pd.read_csv(cfg['dqueriesp'],sep='\t')),'query id')
    
#     if the error in human, use: `cut -f 1 data/alignment.bed.sorted.bed | sort| uniq -c | grep -v CHR | grep -v GL | grep -v KI`
    dalignbedqueriesp=cfg['dalignbedqueriesp']
    logging.info(basename(dalignbedqueriesp))
    if not exists(dalignbedqueriesp) or cfg['force']:
        dalignbed=pd.merge(dalignbed,dqueries,on='query id',suffixes=('', '.1'))
        dalignbed.to_csv(dalignbedqueriesp,'\t')
    return cfg

def alignmentbed2dalignedfasta(cfg):
    """
    Get sequences in FASTA format from BED file
    step#5

    :param cfg: configuration dict
    """    
    datatmpd=cfg['datatmpd']
    alignmentbedp=cfg['alignmentbedp']    
    dalignedfastap=cfg['dalignedfastap']
    logging.info(basename(dalignedfastap))
    if not exists(dalignedfastap) or cfg['force']:
        alignedfastap='{}/05_alignment.fa'.format(datatmpd)
        if not exists(alignedfastap) or cfg['force']:
            cmd=f"{cfg['bedtools']} getfasta -s -name -fi {cfg['genomep']} -bed {alignmentbedp} -fo {alignedfastap}"
            runbashcmd(cmd)

        dalignedfasta=fa2df(alignedfastap)
        dalignedfasta.columns=['aligned sequence']
        dalignedfasta=dalignedfasta.loc[(dalignedfasta.apply(lambda x: not 'N' in x['aligned sequence'],axis=1)),:] #FIXME bwa aligns to NNNNNs
        dalignedfasta.index=[i.split('(')[0] for i in dalignedfasta.index] # for bedtools 2.27, the fasta header now has hanging (+) or (-)
        dalignedfasta.index.name='id'
        dalignedfasta.to_csv(dalignedfastap,sep='\t')
    return cfg

def dalignbed2dalignbedqueriesseq(cfg):
    """
    Get sequences from BED file
    step#6

    :param cfg: configuration dict
    """
    datatmpd=cfg['datatmpd']
    dalignbedqueries=del_Unnamed(pd.read_csv(cfg['dalignbedqueriesp'],sep='\t'))
    dalignedfasta=del_Unnamed(pd.read_csv(cfg['dalignedfastap'],sep='\t'))
    dalignbedqueriesseqp=cfg['dalignbedqueriesseqp']
    logging.info(basename(dalignbedqueriesseqp))
    if not exists(dalignbedqueriesseqp) or cfg['force']:        
        dalignbedqueriesseq=pd.merge(dalignbedqueries,dalignedfasta,on='id',suffixes=('', '.2'))
        dalignbedqueriesseq=dalignbedqueriesseq.dropna(subset=['aligned sequence'],axis=0)

        # dalignbed.index.name='id'
        dalignbedqueriesseq=dalignbedqueriesseq.drop_duplicates()
        dalignbedqueriesseq.to_csv(dalignbedqueriesseqp,sep='\t')
    return cfg

def dalignbedqueriesseq2dalignbedstats(cfg):
    """
    Gets scores for queries
    step#7

    :param cfg: configuration dict
    """
    datatmpd=cfg['datatmpd']
    dalignbedqueriesseq=del_Unnamed(pd.read_csv(cfg['dalignbedqueriesseqp'],sep='\t'))
    
    dalignbedstatsp=cfg['dalignbedstatsp']  
    logging.info(basename(dalignbedstatsp))
    if not exists(dalignbedstatsp) or cfg['force']:
        df=dalignbedqueriesseq.apply(lambda x: align(x['query sequence'],x['aligned sequence'],
                                                    psm=2,pmm=0.5,pgo=-3,pge=-1,),
                           axis=1).apply(pd.Series)
        print(df.head())
        df.columns=['alignment','alignment: score']
        dalignbedstats=dalignbedqueriesseq.join(df)
        del df
        dalignbedstats.to_csv(dalignbedstatsp,sep='\t')
    return cfg
def dannots2dalignbed2dannotsagg(cfg):
    """
    Aggregate annotations per query
    step#8

    :param cfg: configuration dict
    """
    datatmpd=cfg['datatmpd']
    
    daannotp=f'{datatmpd}/08_dannot.tsv'
    cfg['daannotp']=daannotp
    dannotsaggp=cfg['dannotsaggp']
    logging.info(basename(daannotp))
    if ((not exists(daannotp)) and (not exists(dannotsaggp))) or cfg['force']:
        gff_renamed_cols=[c+' annotation' if c in set(bed_colns).intersection(gff_colns) else c for c in gff_colns]
        dannots=pd.read_csv(cfg['annotationsbedp'],sep='\t',
                   names=bed_colns+gff_renamed_cols,
                           low_memory=False)
        dannots=del_Unnamed(dannots)

        dannots=dannots.set_index('id')
        dannots['annotations count']=1
        # separate ids from attribute columns
        dannots=lambda2cols(dannots,lambdaf=gffatributes2ids,
                    in_coln='attributes',
                to_colns=['gene name','gene id','transcript id','protein id','exon id'])
        dannots=dannots.drop(['attributes']+[c for c in gff_renamed_cols if 'annotation' in c],axis=1)
        logging.debug('or this step takes more time?')
        to_table(dannots,daannotp)
#         to_table_pqt(dannots,daannotp)
    else:
#         dannots=read_table_pqt(daannotp)
        dannots=read_table(daannotp)
        dannots=del_Unnamed(dannots)
        
    logging.info(basename(dannotsaggp))
    if not exists(dannotsaggp) or cfg['force']:
        if not 'dannots' in locals():
#             dannots=read_table_pqt(daannotp)
            dannots=pd.read_table(daannotp,low_memory=False)
        dannots=del_Unnamed(dannots)
        dannots=dannots.reset_index()
        logging.debug('aggregating the annotations')
        from rohan.dandage.io_sets import unique 
        cols2aggf={'annotations count':np.sum,
                  'type': unique,
                  'gene name': unique,
                  'gene id': unique,
                  'transcript id': unique,
                  'protein id': unique,
                  'exon id': unique}
        dannotsagg=dannots.groupby('id').agg(cols2aggf)
        dannotsagg['annotations count']=dannotsagg['annotations count']-1
        dannotsagg.loc[dannotsagg['annotations count']==0,'region']='intergenic'
        dannotsagg.loc[dannotsagg['annotations count']!=0,'region']='genic'
        logging.debug('end of the slowest step')            
        del dannots    
        dannotsagg=dannotsagg.reset_index()
#         to_table_pqt(dannotsagg,dannotsaggp)
        dannotsagg.to_csv(dannotsaggp,sep='\t')        
    return cfg

def dannotsagg2dannots2dalignbedannot(cfg):
    """
    Map aggregated annotations to queries
    step#9

    :param cfg: configuration dict
    """
    datatmpd=cfg['datatmpd']
    
    dannotsagg=del_Unnamed(pd.read_csv(cfg['dannotsaggp'],sep='\t'))
    dalignbedstats=del_Unnamed(pd.read_csv(cfg['dalignbedstatsp'],sep='\t'))
    dalignbedannotp=cfg['dalignbedannotp']
    logging.info(basename(dalignbedannotp))
    if not exists(dalignbedannotp) or cfg['force']:
        # df2info(dalignbed)
        # df2info(dannotsagg)
        dalignbedannot=dalignbedstats.set_index('id').join(set_index(dannotsagg,'id'),
                                              rsuffix=' annotation')
        dalignbedannot['NM']=dalignbedannot['NM'].apply(int)
#         from rohan.dandage.get_scores import get_beditorscore_per_alignment,get_cfdscore
#         dalignbedannot['beditor score']=dalignbedannot.apply(lambda x : get_beditorscore_per_alignment(NM=x['NM'],
#                                genic=True if x['region']=='genic' else False,
#                                alignment=x['alignment'],
#                                pam_length=len(x['PAM']),
#                                pam_position=x['original position'],
#                                # test=cfg['test'],
#                                 ),axis=1) 
#         dalignbedannot['CFD score']=dalignbedannot.apply(lambda x : get_cfdscore(x['query sequence'].upper(), x['aligned sequence'].upper()), axis=1)            
        dalignbedannot.to_csv(dalignbedannotp,sep='\t')
    return cfg

def dalignbedannot2daggbyquery(cfg):
    """
    Aggregate annotations per alignment to annotations per query.
    step#10

    :param cfg: configuration dict
    """
    datatmpd=cfg['datatmpd']

    dalignbedannot=del_Unnamed(pd.read_csv(cfg['dalignbedannotp'],sep='\t',low_memory=False))
    
    daggbyqueryp=f'{datatmpd}/10_daggbyquery.tsv'      
    logging.info(basename(daggbyqueryp))
    if not exists(daggbyqueryp) or cfg['force']:
        dalignbedannot=dfliststr2dflist(dalignbedannot,
                                ['type', 'gene name', 'gene id', 'transcript id', 'protein id', 'exon id'],
                                colfmt='tuple')    
        dalignbedannot['alternate alignments count']=1
        import itertools
        from rohan.dandage.io_sets import unique
        def unique_dropna(l): return unique(l,drop='nan')
        def merge_unique_dropna(l): return unique(list(itertools.chain(*l)),drop='nan')
        cols2aggf={'id':unique_dropna,
         'type':merge_unique_dropna,
         'gene name':merge_unique_dropna,
         'gene id':merge_unique_dropna,
         'transcript id':merge_unique_dropna,
         'protein id':merge_unique_dropna,
         'exon id':merge_unique_dropna,
         'region':unique_dropna,
         'alternate alignments count':sum,
        }
        daggbyquery=dalignbedannot.groupby('query id').agg(cols2aggf)
        daggbyquery.to_csv(daggbyqueryp,sep='\t')
        daggbyquery.to_csv(cfg['dalignannotedp'],sep='\t')
    return cfg

def queries2alignments(cfg):
    """
    All the processes in alignannoted detection are here.
    
    :param cfg: Configuration settings provided in .yml file
    """
    from rohan.dandage.align import get_genomes
    get_genomes(cfg)
    
    cfg['datad']=cfg['prjd']
    cfg['plotd']=cfg['datad']
    dalignannotedp=f"{cfg['datad']}/dalignannoted.tsv"  
    
#     stepn='04_alignannoteds'
#     logging.info(stepn)
    
    cfg['datatmpd']=f"{cfg['datad']}/tmp"
    for dp in [cfg['datatmpd']]:
        if not exists(dp):
            makedirs(dp)
            
    step2doutp={
    1:'01_queries_queryl*.fa',
    2:'02_dalignbed.tsv',
    3:'03_annotations.bed',
    4:'04_dalignbedqueries.tsv',
    5:'05_dalignedfasta.tsv',
    6:'06_dalignbedqueriesseq.tsv',
    7:'07_dalignbedstats.tsv',
    8:'08_dannotsagg.tsv',
    9:'09_dalignbedannot.tsv',
    10:'10_daggbyquery.tsv',
    }
    cfg['dqueriesp']=cfg['dinp']
    cfg['alignmentbedp']=f"{cfg['datatmpd']}/02_alignment.bed"
    cfg['dalignbedp']=f"{cfg['datatmpd']}/02_dalignbed.tsv"
    cfg['dalignbedqueriesp']=f"{cfg['datatmpd']}/04_dalignbedqueries.tsv"
    cfg['dalignedfastap']=f"{cfg['datatmpd']}/05_dalignedfasta.tsv"
    cfg['dalignbedqueriesseqp']=f"{cfg['datatmpd']}/06_dalignbedqueriesseq.tsv"
    cfg['dalignbedstatsp']=f"{cfg['datatmpd']}/07_dalignbedstats.tsv"
    cfg['dannotsaggp']=f"{cfg['datatmpd']}/08_dannotsagg.tsv"
    cfg['dalignbedannotp']=f"{cfg['datatmpd']}/09_dalignbedannot.tsv"
    cfg['daggbyqueryp']=f"{cfg['datatmpd']}/10_daggbyquery.tsv"

    annotationsbedp=f"{cfg['datatmpd']}/03_annotations.bed"
    cfg['annotationsbedp']=annotationsbedp
    
    dqueries=read_table(cfg['dqueriesp'])
    print(dqueries.head())
    #check which step to process
    for step in range(2,10+1,1):
        if not exists(f"{cfg['datatmpd']}/{step2doutp[step]}"):
            if step==2:
                step=-1
            break
    logging.info(f'process from step:{step}')
    cfg['dalignannotedp']='{}/dalignannoted.tsv'.format(cfg['datad'])
    if not exists(cfg['dalignannotedp']) or cfg['force']:
        if step<=1:
            cfg=dqueries2queriessam(cfg,dqueries)
        if step<=2:
            cfg=queriessam2dalignbed(cfg)
        if step<=3:
            cfg=dalignbed2annotationsbed(cfg)
        if step<=4:
            cfg=dalignbed2dalignbedqueries(cfg)
        if step<=5:
            cfg=alignmentbed2dalignedfasta(cfg)
        if step<=6:
            cfg=dalignbed2dalignbedqueriesseq(cfg)
        if step<=7:
            cfg=dalignbedqueriesseq2dalignbedstats(cfg)
        if step<=8:
            cfg=dannots2dalignbed2dannotsagg(cfg)
        if step<=9:
            cfg=dannotsagg2dannots2dalignbedannot(cfg)
        if step<=10:
            cfg=dalignbedannot2daggbyquery(cfg)

        if cfg is None:        
            logging.warning(f"no alignment found")
            cfg['step']=4
            return saveemptytable(cfg,cfg['dalignannotedp'])
        import gc
        gc.collect()