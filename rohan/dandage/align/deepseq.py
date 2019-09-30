from rohan.global_imports import * 
from rohan.dandage.io_sys import runbashcmd
from rohan.dandage.io_seqs import *
from os.path import isdir
from glob import iglob
import logging

def get_coverage(aligned,reference2seq,target_ini,target_end,target=False,ref_name=None):
    cov=pd.DataFrame()
    cov.index.name='refi'
    if target:
        for pileupcol in aligned.pileup('amplicon', 0,len(reference2seq['amplicon']),max_depth=10000000):
            cov.loc[pileupcol.pos, 'amplicon']=pileupcol.n
        for pileupcol in aligned.pileup('amplicon', target_ini,target_end,max_depth=10000000):
            cov.loc[pileupcol.pos, 'target']=pileupcol.n    
    else:
        for pileupcol in aligned.pileup(ref_name, 0,len(reference2seq[ref_name]),max_depth=10000000):
            cov.loc[pileupcol.pos, ref_name]=pileupcol.n        
    return cov

def qual_chars2nums(qual_chars,qual_format):
    """
    This converts the phred qualities from the ASCII format to numbers.
    
    :param qual_chars: phred score in ASCII format (str).
    :param qual_format: phred score in numbers (list).
    """
    # format : 33 or 64 
    qual_nums = []
    for char in qual_chars.rstrip('\n'):
        qual_nums.append(ord(char) - qual_format)
        #print('quality is ' + str(qual_nums))
    return qual_nums 

def get_location_first_indel(read):
    location=0
    for cigari,(cigar_type,cigar_length) in enumerate(read.cigar):
        if cigar_type == 1 or cigar_type == 2:
            break        
        location+=cigar_length        
    if cigari==len(read.cigar)-1:
        return np.nan
    else:
        return location
    
def get_aligned(dirp,method=None,test=False):
    from rohan.dandage.io_sys import runbashcmd
    logging.info(dirp)
    coms=[]
    if method is None:
        # detect if paired end local or single global
        if exists(f"{dirp}/R1.fastq") and exists(f"{dirp}/R2.fastq"):
            method='local'
        elif exists(f"{dirp}/R.fastq"):
            method='global'
    elif method=='global' and exists(f"{dirp}/R1.fastq") and exists(f"{dirp}/R2.fastq"):
        logging.info(f"merging the reads: {dirp}/R1/2.fastq")
        id2seq=read_fasta(f"{dirp}/reference.fasta")
        reflen=int(len(id2seq[list(id2seq.keys())[0]]))
#         -n {reflen} -m {reflen}
        coms+=[f"pear -f {dirp}/R1.fastq -r {dirp}/R2.fastq  -o {dirp}/R.fastq > {dirp}/log_pear.log;mv {dirp}/R.fastq.assembled.fastq {dirp}/R.fastq"]
    coms+=[f'fastp -i {dirp}/R1.fastq -I {dirp}/R2.fastq -o {dirp}/R1_flt.fastq -O {dirp}/R2_flt.fastq -q 20 -e 20 -j {dirp}/R2_flt.fastq.report.json -h {dirp}/R2_flt.fastq.report.html 2> {dirp}/log_fastp.log' if method=='local' else f'fastp -i {dirp}/R.fastq -o {dirp}/R_flt.fastq -q 20 -e 20 -j {dirp}/R_flt.fastq.report.json -h {dirp}/R_flt.fastq.report.html 2> {dirp}/log_fastp.log',
    f'bowtie2-build --quiet {dirp}/reference.fasta {dirp}/reference',
    f'bowtie2 -p 6 --very-sensitive-local --no-discordant --no-mixed -x {dirp}/reference -1 {dirp}/R1_flt.fastq -2 {dirp}/R2_flt.fastq -S {dirp}/aligned.sam 2> {dirp}/log_bowtie2.log' if method=='local' else f'bowtie2 -p 6 --end-to-end --very-sensitive --no-discordant --no-mixed -x {dirp}/reference -U {dirp}/R_flt.fastq -S {dirp}/aligned.sam 2> {dirp}/log_bowtie2.log',
    f'samtools view -bS {dirp}/aligned.sam | samtools sort - -o {dirp}/aligned.bam',
    f'samtools index {dirp}/aligned.bam',
    f'samtools flagstat {dirp}/aligned.bam > {dirp}/log_samtools_flagstat.log',]
    for com in coms:
        if test:
            print(com)
        else:
            logging.info(com)
            runbashcmd(com) 
            
def get_daligned_target(dirp):
    import pysam
    from rohan.dandage.io_seqs import read_fasta
    from rohan.dandage.io_strs import findall
    logging.info(dirp)
    aligned=pysam.AlignmentFile(f'{dirp}/aligned.bam', 'rb')
    reference2seq=read_fasta(f'{dirp}/reference.fasta')
    # target_region
    cfg={}
    cfg['target_ini']=findall(reference2seq['amplicon'],reference2seq['target'])[0]
    cfg['target_end']=cfg['target_ini']+len(reference2seq['target'])
    dcov=get_coverage(aligned,reference2seq,cfg['target_ini'],cfg['target_end'])
    aligned_reads = aligned.fetch(contig='amplicon', 
                                  start=cfg['target_ini'],stop=cfg['target_end'])
    readid2seq={}
    for read in aligned_reads:
        if read.is_paired and not read.is_unmapped:
            # no indels in target region
            location_indel=get_location_first_indel(read)
            if pd.isnull(location_indel) or location_indel>cfg['target_end']:
                # trim to region
                # take query from query_alignment_start to stop 
                # and index it as reference_start to stop
                # print(read.qstart,read.qend)
                pos2seq=dict(zip(range(read.reference_start,read.reference_end+1),
                        list(read.query_sequence[read.query_alignment_start:read.query_alignment_end])))
                readid2seq[read.qname]=pd.Series(pos2seq)[list(range(cfg['target_ini'],cfg['target_end']))]
    df=pd.concat(readid2seq,axis=1).T
    df=df.sort_index(axis=1)
    df.columns=[f"{i} {s}" for i,s in zip(df.columns.tolist(),list(reference2seq['target']))]
    # save
    to_table(dcov,f"{dirp}/dcoverage.tsv")
    to_table(df,f"{dirp}/daligned.pqt")  
    yaml.dump(cfg,open(f"{dirp}/cfg.yml",'w'))
    return df            

def get_daligned(dirp,method):
    import pysam
    from rohan.dandage.io_seqs import read_fasta
    from rohan.dandage.io_strs import findall
    logging.info(dirp)
    aligned=pysam.AlignmentFile(f'{dirp}/aligned.bam', 'rb')
    reference2seq=read_fasta(f'{dirp}/reference.fasta')
    # target_region
    for ref in reference2seq:      
        logging.info(ref)
        cfg={}
        cfg['target_ini']=0
        cfg['target_end']=len(reference2seq[ref])
        dcov=get_coverage(aligned,reference2seq,cfg['target_ini'],cfg['target_end'],target=False,ref_name=ref)
        aligned_reads = aligned.fetch(contig=ref, 
                                      start=cfg['target_ini'],stop=cfg['target_end'])
        readid2seq={}
        for read in aligned_reads:
            if method=='local':
                if not read.is_paired:
                    continue # skip
            if not read.is_unmapped:
                # no indels in target region
                location_indel=get_location_first_indel(read)
                if pd.isnull(location_indel) or location_indel>cfg['target_end']:
                    # trim to region
                    # take query from query_alignment_start to stop 
                    # and index it as reference_start to stop
                    # print(read.qstart,read.qend)
                    pos2seq=dict(zip(range(read.reference_start,read.reference_end+1),
                            list(read.query_sequence[read.query_alignment_start:read.query_alignment_end])))
                    readid2seq[read.qname]=pd.Series(pos2seq)[list(range(cfg['target_ini'],cfg['target_end']))]
        if len(readid2seq.keys())!=0:
            df=pd.concat(readid2seq,axis=1).T
            df=df.sort_index(axis=1)
            df.columns=[f"{i} {s}" for i,s in zip(df.columns.tolist(),list(reference2seq[ref]))]
            # save
            to_table(dcov,f"{dirp}/dcoverage_{ref}.tsv")
            to_table(df,f"{dirp}/daligned_{ref}.pqt")  
            yaml.dump(cfg,open(f"{dirp}/cfg_{ref}.yml",'w'))
        else:
            logging.warning(f'no alignments for reference: {ref} found')
#     return df            
