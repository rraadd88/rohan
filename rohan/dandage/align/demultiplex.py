import warnings
warnings.filterwarnings(action='ignore')

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
    """
    row for:   RC_for_index    
    col rev:   RC_rev_index
    plate for: PE1_index
    plate rev: PE2_index
    """
    import re
    dbarcodes.index=range(len(dbarcodes))
    sample2bcr1r2={}
    sample2primersr1r2={}
    sample2fragsr1r2={}
    sample2regsr1r2={}
    for i in dbarcodes.index:
        x=dbarcodes.iloc[i,:]
        samplen=x['sample name']
        sample2bcr1r2[samplen]=f"{bc2seq[str(x['plate for'])]}{bc2seq[str(x['row for'])]}{bc2seq[str(x['plate rev'])]}{bc2seq[str(x['col rev'])]}"
        sample2primersr1r2[samplen]=[
            f"{bc2seq[str(x['plate for'])]}{oligo2seq['plate_fivep_sticky']}{bc2seq[str(x['row for'])]}{oligo2seq['RC_fivep_sticky']}",
            f"{bc2seq[str(x['plate rev'])]}{oligo2seq['plate_threep_sticky']}{bc2seq[str(x['col rev'])]}{oligo2seq['RC_threep_sticky']}"
        ]
        sample2fragsr1r2[samplen]=[
            f"<{bc2seq[str(x['plate for'])]}><{oligo2seq['plate_fivep_sticky']}><{bc2seq[str(x['row for'])]}><{oligo2seq['RC_fivep_sticky']}",
            f"<{bc2seq[str(x['plate rev'])]}><{oligo2seq['plate_threep_sticky']}><{bc2seq[str(x['col rev'])]}><{oligo2seq['RC_threep_sticky']}"
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
        if test and np.remainder(ri,100000)==0:
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
        to_dict(sample2reads,outp)

def align_demultiplexed(cfg,sample2readids,sample,refn,test=False):
    dirp=f"{cfg['prjd']}/{sample.replace(' ','_')}"
    print(dirp)
    # save the readids
    if not exists(dirp):
        makedirs(dirp,exist_ok=False)
    with open(f'{dirp}/read_ids.txt','w') as f:
        f.write('\n'.join(sample2readids[sample]))
    # save reference
#     refn=sample.split(' ')[0]
    to_fasta({refn:read_fasta(cfg['referencep'])[refn]},f'{dirp}/reference.fasta')
    # trim fasq and align
    coms=[]
    for ri in [1,2]:
        cutlength=cfg['primer end']+(0 if not cfg['cut target spacer'] else (cfg['target spacer 5prime'] if ri==1 else cfg['target spacer 3prime']))
        coms.append(f"seqtk subseq {cfg[f'input_r{ri}p']} {dirp}/read_ids.txt | seqtk trimfq -b {cutlength} -e 0 - > {dirp}/R{ri}.fastq")
    for com in coms:
        if test:
            print(com)
        else:
            logging.info(com)
            runbashcmd(com) 
    get_aligned(dirp,method=cfg['alignment method'],test=test)
    get_daligned(dirp,method=cfg['alignment method'])
                    
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
    get_aligned(dirp,method='global',test=test)
    get_daligned(dirp,method='global',)

def get_read_counts_bystep(cfg):
    from rohan.dandage.align.deepseq import get_read_counts_from_log
    doutp=f"{cfg['prjd']}/data_demultiplex_qc/dcoverage.tsv"
    dcoverage=read_table(doutp)
    sample2readids=read_dict(cfg['sample2readidsp'])
    dplot=pd.DataFrame(dcoverage.sort_index(1).set_index('refi').mean().sort_values(ascending=True)).reset_index().rename(columns={'index':'sample',0:'read depth (mean)'}).reset_index()

    dfs=[]
    dn2df={}
    dn2df['step#0 input demultiplexing read count']=pd.Series({k:len(sample2readids[k]) for k in sample2readids}).T
    dn2df['step#4 output alignment read depth (pysam)']=pd.Series(dcoverage.sort_index(1).set_index('refi').mean().sort_values(ascending=True))
    dfs.append(pd.concat(dn2df,axis=1,sort=False))

    dlogps=pd.DataFrame([[basename(dirname(p)),basenamenoext(p).split('_')[1],p] for p in iglob(f"{cfg['prjd']}/*/log_*.log")],
                                                                                        columns=['sample name','step','path to log']).pivot_table(columns='step',index='sample name',values='path to log',aggfunc=lambda x: list(x)[0])

    df=dlogps.apply(lambda x: [get_read_counts_from_log(x[k],step=k) for k in x.index],axis=1).apply(pd.Series)
    df.columns=dlogps.columns
    rename={
    'pear':'step#0.1',
    'fastp':'step#1',
    'bowtie2':'step#2',
    'samtools':'step#3',
    }
    for c in df:
        df_=df[c].apply(pd.Series)
        df_.columns=[f"{rename[c]} {s} {c} read count" for s in ['input','output']]
        dfs.append(df_)
    #     break
    dread_counts=pd.concat(dfs,axis=1,sort=False).sort_index(axis=1)
    return dread_counts
                         
def plot_qc(cfg):
    # get data
    ## get coverage
    doutp=f"{cfg['prjd']}/data_demultiplex_qc/dcoverage.tsv"
    if not exists(doutp):
        ps=[p for p in iglob(f"{cfg['prjd']}/*/dcoverage_*.tsv") if not '/undetermined' in p]
        dcoverage=read_manytables(ps,axis=1,params_read_csv={'sep':'\t','index_col':'refi'},params_concat={'ignore_index':False},
                                 labels=[basename(dirname(p)) for p in ps ])
        dcoverage.columns=[c[0] for c in dcoverage.columns]
        to_table(dcoverage,doutp)                 
    else:
        dcoverage=read_table(doutp)
                             
    ## get read counts by step#
    doutp=f"{cfg['prjd']}/data_demultiplex_qc/ddread_counts.tsv"
    if not exists(doutp):
        dread_counts=get_read_counts_bystep(cfg)
        to_table(dread_counts,doutp)                 
    else:
        dread_counts=read_table(doutp)
    
    # plot stuff
    ## coverage by sample
    plotp=f"{cfg['prjd']}/plot/plot qc demupliplexed coverage.png"
    if not exists(plotp):
        dplot=dcoverage.sort_index(1).set_index('refi')
        plt.figure(figsize=[4+int(len(dplot)/40),8])
        ax=plt.subplot()
        ax=dplot.plot(cmap='hsv',alpha=0.5,
                     ax=ax)
        ax.legend(frameon=False, bbox_to_anchor=[1,1], ncol=int(len(dplot)/40))
        ax.set_xlabel('position')
        ax.set_ylabel('read depth')
        savefig(plotp)                    
   
    ## coverage by sample ranked
    plotp=f"{cfg['prjd']}/plot/plot qc demupliplexed coverage ranked.png"
    if not exists(plotp):
        dplot=pd.DataFrame(dcoverage.sort_index(1).set_index('refi').mean().sort_values(ascending=True)).reset_index().rename(columns={'index':'sample',0:'read depth (mean)'}).reset_index()
        plt.figure(figsize=[4,3+len(dplot)/6])
        ax=plt.subplot()
        ax=dplot.plot(x='read depth (mean)',y='index',yticks=dplot['index'],style='.-',
                     ax=ax,legend=False)
        _=ax.set_yticklabels(dplot['sample'])
        plt.tight_layout()
        savefig(plotp)   

    ## read counts by steps
    plotp=f"{cfg['prjd']}/plot/plot qc demupliplexed read_counts by step.png"
    if not exists(plotp):                             
        def plot_read_counts_bystep(dplot):
            fig=plt.figure(figsize=[10,(len(dplot)*0.35)+2])
            ax=plt.subplot()
            ax=sns.heatmap(dplot.apply(lambda x: (x/x.max())*100,axis=1),cmap='Reds',
                           annot=True,fmt='.1f',
                           ax=ax)
        plot_read_counts_bystep(dread_counts)
        savefig(plotp)
                             
def make_chunks(cfg_chunk):
    cfg_=cfg_chunk
    cfg_chunk['prjd']=f"{cfg_['prjd']}/chunks"
    makedirs(cfg_chunk['prjd'],exist_ok=True)
    coms=[]
    coms+=[f"split -a 8 -l {cfg_['chunksize']*4} --numeric-suffixes=1 --additional-suffix=.fastq {cfg_[f'input_r{i}p']} {cfg_chunk['prjd']}/undetermined_chunk_R{i}_" for i in [1,2]]
    for com in coms:
        runbashcmd(com)
    chunk_cfgps=[]
    for chunk_input_r1p in glob(f"{cfg_chunk['prjd']}/undetermined_chunk_R1_*.fastq"):
        cfg_chunk_=cfg_chunk
        cfg_chunk_['input_r1p']=chunk_input_r1p
        cfg_chunk_['input_r2p']=chunk_input_r1p.replace('R1','R2')
        cfg_chunk_['sample2readidsp']=f"{cfg_chunk_['prjd']}/chunk{basenamenoext(chunk_input_r1p).split('_')[-1]}_sample2readids.json"
        cfg_chunk_['cfgp']=f"{cfg_chunk_['prjd']}/chunk{basenamenoext(chunk_input_r1p).split('_')[-1]}_cfg.yml"
        chunk_cfgps.append(cfg_chunk_['cfgp'])
        to_dict(cfg_chunk_,cfg_chunk_['cfgp'])
    del cfg_,cfg_chunk
    return chunk_cfgps

    logging.info(f'running {basenamenoext(cfgp)}')
    cfg=read_dict(cfgp)
    fastqr1_reads=SeqIO.parse(cfg['input_r1p'],'fastq')
    fastqr2_reads=SeqIO.parse(cfg['input_r2p'],"fastq")
    logging.info('read the fastq files')
    if not exists(cfg['sample2readidsp']):
        demultiplex_readids(fastqr1_reads=fastqr1_reads,fastqr2_reads=fastqr2_reads,
                        linkerr1r2=cfg['linkerr1r2'],sample2bcr1r2=cfg['sample2bcr1r2'],barcode_poss=cfg['barcode_poss'],
                        alignment_score_coff=cfg['alignment_score_coff'],
                        outp=cfg['sample2readidsp'],
                        test=False)
                                
def run_chunk_demultiplex_readids(cfgp):
    logging.info(f'running {basenamenoext(cfgp)}')
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
                                
def collect_chunk_demultiplex_readids(cfg):
    chunk_sample2readids=[read_dict(p) for p in glob(f"{cfg['prjd']}/chunks/chunk*sample2readids.json")]
    print(f"{cfg['prjd']}/chunks/chunk*sample2readids.json")
    return merge_dict_values(chunk_sample2readids)

def run_demupliplex(cfg,test=False):
    from multiprocessing import Pool

    if isinstance(cfg,str):
        if cfg.endswith('.yml'):
            cfg=read_dict(cfg)      
        else:
            logging.error(f'should be a path to yml file: {cfg}')
    cfg['test']=test
    to_dict(cfg,f"{cfg['prjd']}/input_cfg.yaml")
    dbarcodes=read_table(cfg['dbarcodesp']).sort_values(by=['reference','sample name'])
    sample2refn=dbarcodes.set_index('sample name')['reference'].to_dict()
    bc2seq=read_fasta(cfg['bc2seqp'])
    oligo2seq=read_fasta(cfg['oligo2seqp'])
    
    for i in [1,2]:
        cfg[f'input_r{i}p']=glob(f"{cfg['prjd']}/Undetermined*_R{i}_*.fastq")[0]

    cfg['sample2bcr1r2'],cfg['sample2primersr1r2'],cfg['sample2fragsr1r2'],_,cfg['linkerr1r2']=get_sample2bcr1r2(dbarcodes,bc2seq,oligo2seq)
    cfg['sample2readidsp']=f"{cfg['prjd']}/sample2readids.json"
    cfgp_=f"{cfg['prjd']}/cfg.yml"
    cfg['cfgp']=cfgp_
    to_dict(cfg,cfgp_)

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
    times=[get_datetime(outstr=False)]
    logging.info(f"start. log file: {logp}")
    print(f"start. log file: {logp}")
            
    #step1 get the barcode alignment score max cut off 
    if not 'alignment_score_coff' in cfg:
        cfg['alignment_score_coff']=get_alignment_score_coff(cfg['sample2bcr1r2'])

    # demultiplex
    if not exists(cfg['sample2readidsp']):
        print(cfg['prjd'])
        chunk_cfgps=make_chunks(cfg)
        cfg=read_dict(cfgp_)
        print(cfg['prjd'])
        # multi process
        pool=Pool(processes=cfg['cores'])
        pool.map(run_chunk_demultiplex_readids, chunk_cfgps)
        pool.close(); pool.join()           
        sample2readids=collect_chunk_demultiplex_readids(cfg)
        print('saving sample2readids at ',cfg['sample2readidsp'])
        to_dict(sample2readids,cfg['sample2readidsp'])
        print('done')
    else:
        sample2readids=read_dict(cfg['sample2readidsp'])        
    # output
    for sample in sorted(sample2readids.keys()):
        dirp=f"{cfg['prjd']}/{sample.replace(' ','_')}"
#         if not exists(dirp):                                 
        [f(f'processing demultiplexed sample: {sample}') for f in [logging.info,print]]
        if not sample.startswith('undetermined '):
            outp=f"{dirp}/daligned_{sample2refn[sample]}.pqt"
            if not exists(outp):
                if len(sample2readids[sample])>0:
                    # save the demultiplexed to separate directories
                    align_demultiplexed(cfg,sample2readids,
                                        sample=sample,
                                        refn=sample2refn[sample],
                                        test=cfg['test'])            
            if cfg['test']:
                print('stopping the test')
                import sys
                sys.exit()
                break
        else:
            # align the undetermined to be sure
            check_undetermined(cfg,sample2readids,sample,test=cfg['test'])
    # qc output 
    logging.info('plotting the qc')
    plot_qc(cfg)
    # log time taken        
    times.append(get_datetime(outstr=False))
    [f(f'end. time taken={str(times[-1]-times[0])}') for f in [logging.info,print]]