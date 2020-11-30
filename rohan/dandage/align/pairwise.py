from rohan.global_imports import *

def get_needle_stats(p):
    lines=open(p,'r').readlines()
    d={'# Length:':'alignment length',
     '# Identity:':'% identity',
     '# Similarity:':'% similarity',
     '# Gaps:':'% gaps',
     '# Score:':'alignment score'}
    return pd.Series({d[s]:float(replacemany(l, replaces=['\n','(',')','%'], replacewith='').split(' ')[-1]) for l in lines for s in d if l.startswith(s)})

def run_needle(s1p,s2p,
               form,
               outp,
               outfmt='pair',
               test=False):
    """
    :params outfmt: fasta|pair
    """
    from rohan.dandage.io_sys import runbashcmd    
    if not exists(outp):
        if form.lower()=='dna':
            com=f"needle -auto -asequence {s1p} -bsequence {s2p} -datafile EDNAFULL -gapopen 10.0 -gapextend 0.5 -endopen 10.0 -endextend 0.5 -aformat3 {outfmt} -snucleotide1 -snucleotide2 -outfile {outp}"
        elif form.lower()=='protein':        
            com=f"needle -auto -asequence {s1p} -bsequence {s2p} -datafile EBLOSUM62 -gapopen 10.0 -gapextend 0.5 -endopen 10.0 -endextend 0.5 -aformat3 {outfmt} -sprotein1 -sprotein2 -outfile {outp}"
        else:
            logging.error(f"{outp}!=(DNA|protein)")
        if test:print(com)
        runbashcmd(com)
    return get_needle_stats(outp)
