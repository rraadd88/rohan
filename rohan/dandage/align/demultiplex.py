from rohan.global_imports import * 
from rohan.dandage.io_sys import runbashcmd
import logging

from Bio import SeqIO
from rohan.dandage.io_seqs import *
from rohan.dandage.align.deepseq import get_aligned,get_daligned

def get_alignment_score_coff(sample2bcr1r2):
    """
    :param sample2bcr1r2: sample->barcode dict
    """
    alignment_scores=[]
    si1_=0
    for si1,s1 in enumerate(sample2bcr1r2): 
        for si2,s2 in enumerate(sample2bcr1r2):
            if si1>si2:
                if si1>si1_:
                    print(si1,end=' ')
                si1_=si1
                alignment_scores.append(get_align_metrics(align_global(sample2bcr1r2[s1],sample2bcr1r2[s2]))[1])
    alignment_score_coff=np.max(alignment_scores)
    return alignment_score_coff

def get_sample2bcr1r2(dbarcodes,bc2seq,oligo2seq):
    import re
    dbarcodes.index=range(len(dbarcodes))
    sample2bcr1r2={}
    sample2primersr1r2={}
    sample2fragsr1r2={}
    sample2regsr1r2={}
    for i in dbarcodes.index:
        x=dbarcodes.iloc[i,:]
        samplen=f"{x['Locus']} {x['Position_DMS']}"
        sample2bcr1r2[samplen]=f"{bc2seq[str(x['PE1_index'])]}{bc2seq[str(x['RC_for_index'])]}{bc2seq[str(x['PE2_index'])]}{bc2seq[str(x['RC_rev_index'])]}"
        sample2primersr1r2[samplen]=[
            f"{bc2seq[str(x['PE1_index'])]}{oligo2seq['plate_fivep_sticky']}{bc2seq[str(x['RC_for_index'])]}{oligo2seq['RC_fivep_sticky']}",
            f"{bc2seq[str(x['PE2_index'])]}{oligo2seq['plate_threep_sticky']}{bc2seq[str(x['RC_rev_index'])]}{oligo2seq['RC_threep_sticky']}"
        ]
        sample2fragsr1r2[samplen]=[
            f"<{bc2seq[str(x['PE1_index'])]}><{oligo2seq['plate_fivep_sticky']}><{bc2seq[str(x['RC_for_index'])]}><{oligo2seq['RC_fivep_sticky']}",
            f"<{bc2seq[str(x['PE2_index'])]}><{oligo2seq['plate_threep_sticky']}><{bc2seq[str(x['RC_rev_index'])]}><{oligo2seq['RC_threep_sticky']}"
        ]
        reg_r1 = re.compile("^\w{5}"+f"{sample2primersr1r2[samplen][0]}.*")
        reg_r2 = re.compile("^\w{5}"+f"{sample2primersr1r2[samplen][1]}.*")
        sample2regsr1r2[samplen]=[reg_r1,reg_r2] 
    linkerr1r2=f"{oligo2seq['plate_fivep_sticky']}{oligo2seq['RC_fivep_sticky']}{oligo2seq['plate_threep_sticky']}{oligo2seq['RC_threep_sticky']}"
    return sample2bcr1r2,sample2primersr1r2,sample2fragsr1r2,sample2regsr1r2,linkerr1r2

def demultiplex_readids(fastqr1_reads,fastqr2_reads,
                    linkerr1r2,sample2bcr1r2,barcode_poss,
                    alignment_score_coff,outp=None,test=False):
    """
    trim the fastq, take only barcodes and only linkers
    global align linkers, relax align *0.6

    global align barcodes, stringent align *0.8
        get all the matches above a threshold 
        get the reads that are captured with two or more barcodes
        assign to the one which has higher score
    """
    def get_alignment_score(s1,s2):
        if s1==s2:
            return len(s1)
        elif len(s1)==len(s2):
            return len(s1)-hamming_distance(s1,s2)
        else:
            return get_align_metrics(align_global(s1,s2))[1]
        
    from rohan.dandage.io_dict import sort_dict
    sample2reads={sample:[] for sample in list(sample2bcr1r2.keys())+["undetermined barcode","undetermined linker"]}
    for ri,(r1,r2) in enumerate(zip(fastqr1_reads,fastqr2_reads)):
        if np.remainder(ri,100000)==0:
            print(ri,end=' ')
            logging.info(ri)
        if r1.id != r2.id:
            logging.error(f'{r2.id}')
        else:
            read_has_linker=False
            linker_seq=f"{str(r1.seq)[barcode_poss[0][1]:barcode_poss[1][0]]}{str(r1.seq)[barcode_poss[1][1]:barcode_poss[1][1]+20]}{str(r2.seq)[barcode_poss[0][1]:barcode_poss[1][0]]}{str(r2.seq)[barcode_poss[1][1]:barcode_poss[1][1]+20]}"        
            alignment_score=get_alignment_score(linkerr1r2,linker_seq)
            if alignment_score>=len(linkerr1r2)*0.7:
                read_has_linker=True
            if read_has_linker:
                bc_seq=f"{str(r1.seq)[barcode_poss[0][0]:barcode_poss[0][1]]}{str(r1.seq)[barcode_poss[1][0]:barcode_poss[1][1]]}{str(r2.seq)[barcode_poss[0][0]:barcode_poss[0][1]]}{str(r2.seq)[barcode_poss[1][0]:barcode_poss[1][1]]}"
                sample2alignmentscore={}
                for sample in sample2bcr1r2:
                    alignment_score=get_alignment_score(sample2bcr1r2[sample],bc_seq)
                    if alignment_score>alignment_score_coff:
                        sample2alignmentscore[sample]=[alignment_score]
                if len(sample2alignmentscore.values())!=0:
                    sample=sort_dict(sample2alignmentscore,1,out_list=True)[-1][0]
                    sample2reads[sample].append(r1.id)
                else:
                    sample2reads["undetermined barcode"].append(r1.id)
            else:
                sample2reads["undetermined linker"].append(r1.id)
        if test and ri>1000:
            break
    if outp is None:
        return sample2reads
    else:
        to_yaml(sample2reads,outp)

def align_demultiplexed(cfg,sample2readids,sample,test=False):
    dirp=f"{cfg['prjd']}/{sample.replace(' ','_')}"
    # save the readids
    if not exists(dirp):
        makedirs(dirp,exist_ok=False)
    with open(f'{dirp}/read_ids.txt','w') as f:
        f.write('\n'.join(sample2readids[sample]))
    # save reference
    refn=sample.split(' ')[0]
    to_fasta({refn:read_fasta(cfg['referencep'])[refn]},f'{dirp}/reference.fasta')
    # trim fasq and align
    coms=[]
    for ri in [1,2]:
        coms.append(f"seqtk subseq {cfg[f'input_r{ri}p']} {dirp}/read_ids.txt | seqtk trimfq -b {cfg['primer end']} -e 0 - > {dirp}/R{ri}.fastq")
    for com in coms:
        if test:
            print(com)
        else:
            logging.info(com)
            runbashcmd(com) 
    get_aligned(dirp,test=test)
    get_daligned(dirp)
                    
def check_undetermined(cfg,sample2readids,sample,test=False):
    dirp=f"{cfg['prjd']}/{sample.replace(' ','_')}"
    # save the readids
    if not exists(dirp):
        makedirs(dirp,exist_ok=False)
    with open(f'{dirp}/read_ids.txt','w') as f:
        f.write('\n'.join(sample2readids[sample]))
    # save reference
    sample2primer={}
    for s in cfg['sample2primersr1r2']:
        for seqi,seq in enumerate(cfg['sample2primersr1r2'][s]):
            sample2primer[(f"{s} R{seqi+1}").replace(' ','_')]=seq
    to_fasta(sample2primer,f'{dirp}/reference.fasta')
    # trim fasq and align
    coms=[]
    for ri in [1,2]:
        coms.append(f"seqtk subseq {cfg[f'input_r{ri}p']} {dirp}/read_ids.txt | seqtk trimfq -b {cfg['primer start']} -e {cfg['read length']-cfg['primer end']} - > {dirp}/R{ri}.fastq")
    for com in coms:
        if test:
            print(com)
        else:
            logging.info(com)
            runbashcmd(com) 
    get_aligned(dirp,test=test)
    get_daligned(dirp)
                    
def plot_qc(cfg):
    # get data
    ps=[p for p in iglob(f"{cfg['prjd']}/*/dcoverage_*.tsv") if not '/undetermined' in p]
    dcoverage=read_manytables(ps,axis=1,params_read_csv={'sep':'\t','index_col':'refi'},params_concat={'ignore_index':False},
                             labels=[basename(dirname(p)) for p in ps ])
    dcoverage.columns=[c[0] for c in dcoverage.columns]
    to_table(dcoverage,f"{cfg['prjd']}/data_demultiplex_qc/dcoverage.tsv")                    
    # coverage by sample
    dplot=dcoverage.sort_index(1).set_index('refi')
    plt.figure(figsize=[4+int(len(dplot)/40),8])
    ax=plt.subplot()
    ax=dplot.plot(cmap='hsv',alpha=0.5,
                 ax=ax)
    ax.legend(frameon=False, bbox_to_anchor=[1,1], ncol=int(len(dplot)/40))
    ax.set_xlabel('position')
    ax.set_ylabel('read depth')
    savefig(f"{cfg['prjd']}/plot/plot qc demupliplexed coverage.png")                    
    # coverage by sample ranked
    dplot=pd.DataFrame(dcoverage.sort_index(1).set_index('refi').mean().sort_values(ascending=True)).reset_index().rename(columns={'index':'sample',0:'read depth (mean)'}).reset_index()
    plt.figure(figsize=[4,3+len(dplot)/6])
    ax=plt.subplot()
    ax=dplot.plot(x='read depth (mean)',y='index',yticks=dplot['index'],style='.-',
                 ax=ax,legend=False)
    ax.set_yticklabels(dplot['sample'])
    plt.tight_layout()
    savefig(f"{cfg['prjd']}/plot/plot qc demupliplexed coverage ranked.png")   

def make_chunks(cfg):
    cfg_chunk=cfg
    cfg_chunk['prjd']=f"{cfg['prjd']}/chunks"
    makedirs(cfg_chunk['prjd'],exist_ok=True)
    coms=[]
    coms+=[f"split -a 8 -l {cfg['chunksize']} --numeric-suffixes=1 --additional-suffix=.fastq {cfg[f'input_r{i}p']} {cfg_chunk['prjd']}/undetermined_chunk_R{i}_" for i in [1,2]]
    for com in coms:
        runbashcmd(com)
    chunk_cfgps=[]
    for chunk_input_r1p in glob(f"{cfg_chunk['prjd']}/undetermined_chunk_R1_*.fastq"):
        cfg_chunk_=cfg_chunk
        cfg_chunk_['input_r1p']=chunk_input_r1p
        cfg_chunk_['input_r2p']=chunk_input_r1p.replace('R1','R2')
        cfg_chunk_['sample2readidsp']=f"{cfg_chunk_['prjd']}/chunk{basenamenoext(chunk_input_r1p).split('_')[-1]}_sample2readids.yml"
        cfg_chunk_['cfgp']=f"{cfg_chunk_['prjd']}/chunk{basenamenoext(chunk_input_r1p).split('_')[-1]}_cfg.yml"
        chunk_cfgps.append(cfg_chunk_['cfgp'])
        to_yaml(cfg_chunk_,cfg_chunk_['cfgp'])
    return 

def run_chunk_demultiplex_readids(cfgp):
    cfg=read_yaml(cfgp)
    fastqr1_reads=SeqIO.parse(cfg['input_r1p'],'fastq')
    fastqr2_reads=SeqIO.parse(cfg['input_r2p'],"fastq")
    logging.info('read the fastq files')
    if not exists(cfg['sample2readidsp']):
        demultiplex_readids(fastqr1_reads=fastqr1_reads,fastqr2_reads=fastqr2_reads,
                        linkerr1r2=cfg['linkerr1r2'],sample2bcr1r2=cfg['sample2bcr1r2'],barcode_poss=cfg['barcode_poss'],
                        alignment_score_coff=cfg['alignment_score_coff'],
                        outp=cfg['sample2readidsp'],
                        test=False)
           
def run_demupliplex(cfg,test=False):
    from multiprocessing import Pool

    if isinstance(cfg,str):
        if cfg.endswith('.yml'):
            cfg=read_yaml(cfg)      
        else:
            logging.error(f'should be a path to yml file: {cfg}')
    cfg['test']=test
    to_yaml(cfg,f"{cfg['prjd']}/input_cfg.yaml")
    dbarcodes=read_table(cfg['dbarcodesp']).sort_values(by=['Locus','Position_DMS'])
    bc2seq=read_fasta('data/references/indexes.fa')
    oligo2seq=read_fasta(cfg['oligo2seqp'])

    for i in [1,2]:
        cfg[f'input_r{i}p']=glob(f"{cfg['prjd']}/Undetermined*_R{i}_*.fastq")[0]

    cfg['sample2bcr1r2'],cfg['sample2primersr1r2'],cfg['sample2fragsr1r2'],_,cfg['linkerr1r2']=get_sample2bcr1r2(dbarcodes,bc2seq,oligo2seq)
    cfg['sample2readidsp']=f"{cfg['prjd']}/sample2readids.yml"            
    to_yaml(cfg,f"{cfg['prjd']}/cfg.yml")

    # get the logger running
    from rohan.dandage.io_strs import get_logger,get_datetime
    if cfg['test']:
        level=logging.INFO
    else: 
        level=logging.ERROR
    logp=get_logger(program='demultiplex',
               argv=[cfg['prjd']],
               level=level,
               dp=None)
    time_ini=get_datetime()
    logging.info(f"start. log file: {logp}")
    print(f"start. log file: {logp}")
            
    #step1 get the barcode alignment score max cut off 
    if not 'alignment_score_coff' in cfg:
        cfg['alignment_score_coff']=get_alignment_score_coff(cfg['sample2bcr1r2'])

    # demultiplex
    if not exists(cfg['sample2readidsp']):
        chunk_cfgps=make_chunks(cfg)
        
        pool=Pool(processes=cfg['cores'])
        pool.map(run_chunk_demultiplex_readids, chunk_cfgps)
        pool.close(); pool.join()           
        collect_demultiplex_cheunks(cfg)
    else:
        sample2readids=read_yaml(cfg['sample2readidsp'])        
    # output
    for sample in sample2readids:
        if not sample.startswith('undetermined '):
            # save the demultiplexed to separate directories
            align_demultiplexed(cfg,sample2readids,sample,test=cfg['test'])            
        else:
            # align the undetermined to be sure
            check_undetermined(cfg,sample2readids,sample,test=cfg['test'])
    # qc output 
    plot_qc(cfg)
    # log time taken        
    logging.info(f'end. time taken={str(get_datetime()-time_ini)}')            