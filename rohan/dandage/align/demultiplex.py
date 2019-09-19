from rohan.global_imports import * 
from rohan.dandage.io_sys import runbashcmd
import logging

from Bio import SeqIO
from rohan.dandage.io_seqs import *

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
                    linker_seq,sample2bcr1r2,
                    alignment_score_coff,test=False):
    """
    trim the fastq, take only barcodes and only linkers
    global align linkers, relax align *0.6

    global align barcodes, stringent align *0.8
        get all the matches above a threshold 
        get the reads that are captured with two or more barcodes
        assign to the one which has higher score
    """
    from rohan.dandage.io_dict import sort_dict
    sample2reads={sample:[] for sample in list(sample2bcr1r2.keys())+["undetermined barcode","undetermined linker"]}
    readid2linkeralignment={}
    for ri,(r1,r2) in enumerate(zip(fastqr1_reads,fastqr2_reads)):
        if np.remainder(ri,10000)==0:
            print(ri,end=' ')
        if r1.id != r2.id:
            logging.error(f'{r2.id}')
        else:
            read_has_linker=False
            linker_seq=f"{str(r1.seq)[barcode_poss[0][1]:barcode_poss[1][0]]}{str(r1.seq)[barcode_poss[1][1]:barcode_poss[1][1]+20]}{str(r2.seq)[barcode_poss[0][1]:barcode_poss[1][0]]}{str(r2.seq)[barcode_poss[1][1]:barcode_poss[1][1]+20]}"        
            alignment=get_align_metrics(align_global(linkerr1r2,linker_seq))
            if alignment[1]>=len(linkerr1r2)*0.7:
                readid2linkeralignment[r1.id]=[str(r1.seq),str(r2.seq),alignment]
                read_has_linker=True
            if read_has_linker:
                bc_seq=f"{str(r1.seq)[barcode_poss[0][0]:barcode_poss[0][1]]}{str(r1.seq)[barcode_poss[1][0]:barcode_poss[1][1]]}{str(r2.seq)[barcode_poss[0][0]:barcode_poss[0][1]]}{str(r2.seq)[barcode_poss[1][0]:barcode_poss[1][1]]}"
                sample2alignmentscore={}
                for sample in sample2bcr1r2:
                    alignment=get_align_metrics(align_global(sample2bcr1r2[sample],bc_seq))
                    if alignment[1]>alignment_score_coff:
                        sample2alignmentscore[sample]=alignment
                if len(sample2alignmentscore.values())!=0:
                    sample=sort_dict(sample2alignmentscore,1,out_list=True)[-1][0]
                    sample2reads[sample].append(r1.id)
                else:
                    sample2reads["undetermined barcode"].append(r1.id)
            else:
                sample2reads["undetermined linker"].append(r1.id)
        if test and ri>1000:
            break
    return sample2reads