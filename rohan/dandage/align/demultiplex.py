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
    sample2regsr1r2={}
    for i in dbarcodes.index:
        x=dbarcodes.iloc[i,:]
        samplen=f"{x['Locus']} {x['Position_DMS']}"
        sample2bcr1r2[samplen]=f"{bc2seq[str(x['PE1_index'])]}{bc2seq[str(x['RC_for_index'])]}{bc2seq[str(x['PE2_index'])]}{bc2seq[str(x['RC_rev_index'])]}"
        sample2primersr1r2[samplen]=[
            f"{bc2seq[str(x['PE1_index'])]}{oligo2seq['plate_fivep_sticky']}{bc2seq[str(x['RC_for_index'])]}{oligo2seq['RC_fivep_sticky']}",
            f"{bc2seq[str(x['PE2_index'])]}{reverse_complement(oligo2seq['plate_threep_sticky'])}{bc2seq[str(x['RC_rev_index'])]}{reverse_complement(oligo2seq['RC_threep_sticky'])}"
        ]
        reg_r1 = re.compile("^\w{5}"+f"{sample2primersr1r2[samplen][0]}.*")
        reg_r2 = re.compile("^\w{5}"+f"{sample2primersr1r2[samplen][1]}.*")
        sample2regsr1r2[samplen]=[reg_r1,reg_r2] 
    linkerr1r2=f"{oligo2seq['plate_fivep_sticky']}{oligo2seq['RC_fivep_sticky']}{oligo2seq['plate_threep_sticky']}{oligo2seq['RC_threep_sticky']}"
    return sample2bcr1r2,sample2primersr1r2,sample2regsr1r2,linkerr1r2



