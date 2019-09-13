import pandas as pd
import numpy as np
from Bio import SeqIO,SeqRecord,Seq,Alphabet
import logging

##vars
# common 
mol2codes={'amino acid':["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","X","Y","*"], #for indexing
'amino acid 3letter':['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'],
'codons':["TTT",    "TTC",    "TTA",  "TTG",  "TCT",  "TCC",  "TCA",  "TCG",  "TAT",  "TAC",  "TAA",  "TAG",  "TGT",  "TGC",  "TGA",  "TGG",  "CTT",  "CTC",  "CTA",  "CTG",  "CCT",  "CCC",  "CCA",  "CCG",  "CAT",  "CAC",  "CAA",  "CAG",  "CGT",  "CGC",  "CGA",  "CGG",  "ATT",  "ATC",  "ATA",  "ATG",  "ACT",  "ACC",  "ACA",  "ACG",  "AAT",  "AAC",  "AAA",  "AAG",  "AGT",  "AGC",  "AGA",  "AGG",  "GTT",  "GTC",  "GTA",  "GTG",  "GCT",  "GCC",  "GCA",  "GCG",  "GAT",  "GAC",  "GAA",  "GAG",  "GGT",  "GGC",  "GGA",  "GGG"],}

##defs

def reverse_complement(s): return str((str2seq(s) if isinstance(s,str) else s).reverse_complement())

def reverse_complement_multintseq(seq,nt2complement):
    complement=[]
    for s in list(seq):
        for ss in nt2complement:
            if ss==s:
#                 print(nt2complement[s],s)
                complement.append(nt2complement[s])
                break
    return "".join(complement[::-1]    )
def reverse_complement_multintseqreg(seq,multint2regcomplement,nt2complement):
    complement=[]
    for s in list(seq):
        if s in multint2regcomplement.keys():
            for ss in multint2regcomplement:
                if ss==s:
    #                 print(nt2complement[s],s)
                    complement.append(multint2regcomplement[s])
                    break
        elif s in nt2complement.keys():
            for ss in nt2complement:
                if ss==s:
                    complement.append(nt2complement[s])
                    break            
        else:
            logging.error(f'odd character {s} in seq {seq}')
        
    return "".join(complement[::-1]    )


def fa2df(alignedfastap,ids2cols=False):
    dtmp=pd.read_csv(alignedfastap,names=["c"])
    dtmp=dtmp.iloc[::2].reset_index(drop=True).join(dtmp.iloc[1::2].reset_index(drop=True),rsuffix='r')
    dtmp.columns=['id','sequence']
    dtmp=dtmp.set_index('id')
    dtmp.index=[i[1:] for i in dtmp.index]
    dtmp.index.name='id'
    if ids2cols:
        for i in dtmp.index:
            seqid,contig,strand,start,end=i.split('|')
            dtmp.loc[i,'seqid']=seqid
            dtmp.loc[i,'contig']=contig
            dtmp.loc[i,'strand']=strand
            dtmp.loc[i,'start']=start
            dtmp.loc[i,'end']=end
    return dtmp

def bedids2bed(df, col_genomeocoord):
    bed_colns=['chromosome', 'start', 'end', 'id', 'NM', 'strand']
    dbed=df.apply(lambda x: x[col_genomeocoord].split('|'),axis=1).apply(pd.Series)
    dbed.columns=['gene id','chromosome', 'strand', 'start', 'end']

    dbed['id']=df[col_genomeocoord]
    dbed['NM']=np.nan
    return dbed[bed_colns]

def genomeocoords2sections(genomecoord):
    try:
        chrom=genomecoord.split(':')[0]
    except:
        raise ValueError(genomecoord)
    start=genomecoord.split(':')[1].split('-')[0]

    end=genomecoord.split(':')[1].split('-')[1].replace('+','').replace('-','')

    tail=genomecoord.split(':')[1].replace(start,'')
    if tail.endswith('+'):
        strand='+'
    elif tail.endswith('-'):
        strand='-'
    else:
        strand=''
#     print(tail,strand)
    return chrom,start,end,strand

bed_colns=['chromosome', 'start', 'end', 'id', 'NM', 'strand']
def genomeocoords2bed(df, col_genomeocoord):
    df=df.dropna(subset=[col_genomeocoord])
    dbed=df.apply(lambda x: genomeocoords2sections(x[col_genomeocoord]),axis=1).apply(pd.Series)
    if len(dbed)!=0:
        dbed.columns=['chromosome', 'start', 'end','strand']
        dbed['id']=df[col_genomeocoord]
        dbed['NM']=np.nan
        return dbed[bed_colns]
    else:
        return pd.DataFrame(columns=['chromosome', 'start', 'end','strand','id','NM'])

from Bio import Alphabet,Seq
def str2seq(s,prt=False):
    if prt:
        alpha=Alphabet.ProteinAlphabet
    else:
        alpha=Alphabet.generic_dna
    return Seq.Seq(s,alpha)

def gffatributes2ids(s):
    """
    Deconvolutes ids from `attributes` column in GFF3 to seprate columns.
    :param s: attribute string.
    :returns: tuple of ids
    """
    Name,gene_id,transcript_id,protein_id,exon_id=np.nan,np.nan,np.nan,np.nan,np.nan
    if '=' in s:
        d=dict([i.split('=') for i in s.split(';')])
        if 'Parent' in d:
            d[d['Parent'].split(':')[0]+'_id']=d['Parent'].split(':')[1]
        Name,gene_id,transcript_id,protein_id,exon_id=np.nan,np.nan,np.nan,np.nan,np.nan
        if 'Name' in d:    
            Name=d['Name']
        if 'gene_id' in d:    
            gene_id=d['gene_id']
        if 'transcript_id' in d:    
            transcript_id=d['transcript_id']
        if 'protein_id' in d:    
            protein_id=d['protein_id']
        if 'exon_id' in d:    
            exon_id=d['exon_id']
    return Name,gene_id,transcript_id,protein_id,exon_id

def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
#     print(s1,s2)
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1.upper(), s2.upper()))

def align(s1,s2,test=False,seqfmt='dna',
         psm=None,pmm=None,pgo=None,pge=None,
         matrix=None,
         outscore=False):
    """
    Creates pairwise local alignment between seqeunces.
    Get the visualization and alignment scores.
    :param s1: seqeunce 1
    :param s2: seqeunce 2    
    
    REF: http://biopython.org/DIST/docs/api/Bio.pairwise2-module.html
    The match parameters are:

    CODE  DESCRIPTION
    x     No parameters. Identical characters have score of 1, otherwise 0.
    m     A match score is the score of identical chars, otherwise mismatch
          score.
    d     A dictionary returns the score of any pair of characters.
    c     A callback function returns scores.
    The gap penalty parameters are:

    CODE  DESCRIPTION
    x     No gap penalties.
    s     Same open and extend gap penalties for both sequences.
    d     The sequences have different open and extend gap penalties.
    c     A callback function returns the gap penalties.  
    --
    DNA: 
    localms: psm=2,pmm=0.5,pgo=-3,pge=-1):
    Protein:
    http://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/650/Use_scoring_matrices.html
    """
    import operator
    from Bio import pairwise2
    if seqfmt=='dna':
        if any([p is None for p in [psm,pmm,pgo,pge]]):
            alignments = pairwise2.align.localxx(s1.upper(),s2.upper())
        else:
            alignments = pairwise2.align.localms(s1.upper(),s2.upper(),psm,pmm,pgo,pge)
    elif seqfmt=='protein':
        from Bio.pairwise2 import format_alignment
        from Bio.SubsMat import MatrixInfo
        if matrix is None:
            matrix = MatrixInfo.blosum62
        alignments =pairwise2.align.globaldx(s1, s2, matrix)
#         print(format_alignment(*a))        
    if test:
        print(alignments)
    alignsymb=np.nan
    score=np.nan
    sorted_alignments = sorted(alignments, key=operator.itemgetter(2))
    for a in alignments:
        alignstr=pairwise2.format_alignment(*a)
        alignsymb=alignstr.split('\n')[1]
        score=a[2]
        if test:
            print(alignstr)
        break
    if not outscore:
        return alignsymb.replace(' ','-'),score
    else:
        return score
    
def translate(dnaseq,fmtout=str,tax_id=None):
    """
    Translates a DNA seqeunce
    :param dnaseq: DNA sequence
    :param fmtout: format of output sequence
    """
    if isinstance(dnaseq,str): 
        dnaseq=Seq.Seq(dnaseq,Alphabet.generic_dna)
    if tax_id is None:
        tax_id=1 # stanndard codon table. ref http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc25
    prtseq=dnaseq.translate(table=tax_id)
    if fmtout is str:
        return str(prtseq)
    else:
        return prtseq
    
## io file
### multiple seq fasta
def fap2id2seq(fap):
    id2seq=SeqIO.to_dict(SeqIO.parse(fap,format='fasta'))
    id2seq={k:str(id2seq[k].seq) for k in id2seq}
    return id2seq
def ids2seqs2fasta(ids2seqs,fastap):
    seqs = (SeqRecord.SeqRecord(Seq.Seq(ids2seqs[id], Alphabet.ProteinAlphabet), id) for id in ids2seqs)
    SeqIO.write(seqs, fastap, "fasta")
def read_fasta(fap):
    return fap2id2seq(fap)
def to_fasta(ids2seqs,fastap):
    return ids2seqs2fasta(ids2seqs,fastap)
    
## generate mutations
def seq_with_substitution(record,pos,sub,test=False):
    from rohan.dandage.io_strs import replacebyposition    
    subfrom=sub[0]
    subto=sub[-1]
    seq=str(record.seq)
    if seq[pos]==subfrom:
        seq=replacebyposition(seq,pos,subto)
        return SeqRecord.SeqRecord(str2seq(seq),id=record.id)        
    else:
        logging.warning(f'indexing issue: {seq[pos-8:pos+7]} {seq[pos]}!={subfrom} {pos}')
#         return None

## lambda function
def process_fasta(infap,outfap,deff,deff_params):
    record=deff(SeqIO.read(infap,format='fasta'),**deff_params)
    record.description=outfap
    if not record is None:
        with open(outfap, "w") as handle:
            SeqIO.write(record, handle, "fasta")    
        return outfap
    
