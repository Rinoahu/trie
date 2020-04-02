#!usr/bin/env python
## python this.py srst2
# this script is used to merge srst2 and blastn
import sys

try:
    srst2, blast8, fasta = sys.argv[1:4]
except:
    print('python this.py srst2 blast8 fasta')
    raise SysemExit()

# get the sequences
seqs_dct = SeqIO.to_dict(SeqIO.parse(fasta))


# filter blast
blast_dct = {}
f = open(blast8, 'r')
for i in f:
    j = i[:-1].split('\t')
    qid, sid, idy, aln, mis, gop, qst, qed, sst, sed, evl, sco = j
    qst, qed = map(int, [qst, qed])
    qln = blast_dct[qid]
    if idy >= 90 and cov >= .9:

        tpd = qid.split('__')[1]
        blast_dct.add(tpd)

f.close()

f = open(srst2, 'r')
for i in f:



