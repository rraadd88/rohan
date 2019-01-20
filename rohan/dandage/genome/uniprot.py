from rohan.dandage.io_sys import runbashcmd
def uniproitid2seq(id,fap):
    runbashcmd(f"wget https://www.uniprot.org/uniprot/{id}.fasta -O tmp.fasta")
    from Bio import SeqIO
    for record in SeqIO.parse(fap, "fasta"):
        return str(record.seq)
        break
