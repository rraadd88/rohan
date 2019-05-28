from rohan.global_imports import *
import os, sys, re
import numpy as np
from collections import defaultdict

def codemltxtp2nums(codemltxtp):  
#     print("Gene_1\tGene_2\tdnds\tdN\tdS")
    status = 0
    genome_dnds = defaultdict(list)
    #print "first\tsecond\tdnds\tdn\tds"
    for i in open(codemltxtp,'r').readlines():
        if i.startswith("pairwise comparison, codon frequencies"):
            status = 1

        if status == 1:
            if i[0].isdigit():
                line = i.rstrip()
                line2 = re.sub("\(", "", line)
                line3 = re.sub("\)", "", line2)
                spaces = line3.split(" ")
                first = spaces[1]
                second = spaces[4]

                first_split = first.split("..")
                second_split = second.split("..")
                g1 = first_split[0]
                g2 = second_split[0]

            if i.startswith("t="):
                line = i.rstrip()
                line1 = re.sub("=", "", line)
                line2 = re.sub("\s+", "\t", line1)
                tabs = line2.split("\t")
                dnds = tabs[7]
                dn = tabs[9]
                ds = tabs[11]
    #           if float(ds) < 2 and float(ds) > 0.01 and float(dnds) < 10:
#                 print(first +"\t"+ second +"\t"+ dnds +"\t"+ dn +"\t"+ ds)
                to_table(pd.DataFrame(pd.Series({'gene ids':f"{first}_{second}",
                            "dN/dS":dnds,
                            "dN":dn,
                            "dS":ds,                             
                            })).T,
                        f"{codemltxtp}.tsv")

def seqs2afasta(ids2seqs,fastap):
    from Bio import SeqIO,SeqRecord,Seq,Alphabet
    seqs = (SeqRecord.SeqRecord(Seq.Seq(ids2seqs[id], Alphabet.ProteinAlphabet), id) for id in ids2seqs)
    SeqIO.write(seqs, fastap, "fasta")

def get_dnds(x,pal2nald,clustalop,codemlp,dndsd,fastad,test=False):
    """
    git clone https://github.com/faylward/dnds
    wget http://www.bork.embl.de/pal2nal/distribution/pal2nal.v14.tar.gz
    tar xvzf pal2nal.v14.tar.gz
    """
#     seqs=dparalogs.loc[0,['gene1 sequence','gene2 sequence']].tolist()
    seqtype2fastap={}
    for seqtype in ['protein','transcript']:
#         fastap=f"codeml/data/{dparalogs.loc[0,'gene ids']}.{seqtype}.fasta"
        fastap=f"{fastad}/{x['protein1 id']}_{x['protein2 id']}.{seqtype}.fasta"
        seqtype2fastap[seqtype]=fastap
#         seqs2afasta(ids2seqs=dict(zip(dparalogs.loc[0,[f'{seqtype}1 id',f'{seqtype}2 id']].tolist(),
#         dparalogs.loc[0,[f'{seqtype}1 sequence',f'{seqtype}2 sequence']].tolist())),
#                    fastap=fastap)
        seqs2afasta(ids2seqs=dict(zip([x[f'protein1 id'],x[f'protein2 id']],
                                      [x[f'{seqtype}1 sequence'],x[f'{seqtype}2 sequence']])),
                   fastap=fastap)

    from rohan.dandage.io_sys import runbashcmd
    from os.path import abspath
    # tmpd='codeml/tmp/'
    codemltxtp=f"{abspath(seqtype2fastap['protein'])}.codeml.txt"
    if not exists(codemltxtp):
        coms=[f"cd {dndsd}/;git clean -f;",
              f"conda activate human_paralogs;{clustalop} --force -i {abspath(seqtype2fastap['protein'])} -o {abspath(seqtype2fastap['protein'])}.aln.faa;",
              f"perl {pal2nald}/pal2nal.pl {abspath(seqtype2fastap['protein'])}.aln.faa {abspath(seqtype2fastap['transcript'])} -output paml -nogap > {dndsd}/cluster_1.pal2nal;",
              f"cd {dndsd};{codemlp};mv {dndsd}/codeml.txt {codemltxtp};",
        ]
#         ", #python2 parse_codeml_output.py codeml.txt

        import subprocess
        for com in coms:
#             print(com)
#             subprocess.call(com,shell=True)
            runbashcmd(com,test=test)
    codemltxtp2nums(codemltxtp)
    #     runbashcmd(com)
