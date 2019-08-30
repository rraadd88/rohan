from rohan.global_imports import * 
from rohan.dandage.io_sys import runbashcmd
from rohan.dandage.io_seqs import *
from os.path import isdir
from glob import iglob

def get_coverage(aligned,reference2seq):
    cov=pd.DataFrame()
    cov.index.name='refi'
    for pileupcol in aligned.pileup('amplicon', 0,len(reference2seq['amplicon']),max_depth=10000000):
        cov.loc[pileupcol.pos, 'amplicon']=pileupcol.n
    # target_region
    ini=findall(reference2seq['amplicon'],reference2seq['target'])[0]
    end=ini+len(reference2seq['target'])
    for pileupcol in aligned.pileup('amplicon', ini,end,max_depth=10000000):
        cov.loc[pileupcol.pos, 'target']=pileupcol.n    
    return cov
