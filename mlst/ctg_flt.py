#!usr/bin/env python
import sys
from Bio import SeqIO

qry, m8 = sys.argv[1:3]

allow_set = {}
f = open(m8, 'r')
for i in f:
    j = i[:-1].split('\t')
    qid, sid, idy, aln = j[:4]
    idy = float(idy)
    aln = int(aln)
    if idy >= 95 and aln > 300:
        allow_set[qid] = sid

f.close()

"NODE_1_length_72570_cov_49.113528"
# filter short contig with low coverage
seqs = SeqIO.parse(qry, 'fasta')
_o = open(qry+'_flt.fsa', 'w')
for i in seqs:
    length =int(i.id.split('_length_')[1].split('_cov_')[0])
    cov = float(i.id.split('_cov_')[1])
    if i.id in allow_set and length >= 500 and cov >= 2:
        #SeqIO.write([i], _o, 'fasta')
        print('>' + i.id)
        print(str(i.seq))





