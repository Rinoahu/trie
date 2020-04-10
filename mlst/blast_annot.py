#!usr/bin/env python
from Bio import SeqIO
import os
import sys

try:
    qry, dir=sys.argv[1: 3]
except:
    print('python this.py qry dir')
    raise SystemExit()

try:
    outfile = sys.argv[3]
except:
    outfile = None

def overlap(s1, e1, s2, e2):
    over_lap = min(e1, e2) - max(s1, s2) + 1
    return over_lap


f = open(qry, 'r')
seqs_dict = SeqIO.to_dict(SeqIO.parse(f, 'fasta'))
f.close()


output = []
for ref in [dir + '/' + elem for elem in os.listdir(dir) if elem.endswith('.fasta') or elem.endswith('_scf.fsa')]:
    os.system('legacy_blast.pl formatdb -i %s -p F | tee log.txt'%ref)
    os.system('legacy_blast.pl blastall -m 8 -e 1e-10 -a 8 -p blastn -i %s -d %s -o %s.blast8 | tee log.txt'%(qry, ref, ref))

    blast_flt = {}
    f = open(ref + '.blast8', 'r')
    for i in f:
        j = i[:-1].split('\t')
        qid, sid, idy, aln, mis, gop, qst, qed, sst, sed, evl, sco = j
        qst, qed, sst, sed = map(int, [qst, qed, sst, sed])
        qln = len(seqs_dict[qid])
        cov = abs(qed-qst) * 1./ qln
        idy = float(idy)
        if idy >= 70 and cov >= .5:
            tpd = qid.split('__')[1:3]
            tpd = tuple(tpd)
            try:
                blast_flt[sid].append([tpd, sid, idy, qst, qed, sst, sed])
            except:
                blast_flt[sid] = [[tpd, sid, idy, qst, qed, sst, sed]]

    genes = []
    for values in blast_flt.values():
        values.sort(key = lambda x: x[5])
        gene = []
        for i in values:
            if len(gene) == 0:
                gene.append(i)
            else:
                s1, e1 = gene[-1][-2:]
                s2, e2 = i[-2:]
                if overlap(s1, e1, s2, e2) > 0 and gene[-1][2] < i[2]:
                    gene[-1] = i
                else:
                    gene.append(i)

        genes.extend(gene)

    #print('break 60', len(set([elem[0] for elem in genes])))

    output.append([ref, set([elem[0] for elem in genes])])

f.close()

#print('break 67', output)
header = set()
for i, j in output:
    header = header.union([elem[0] for elem in j])

#print(header)
header = ['Sample'] + list(header)

header_id = {}
flag = 0
for i in header:
    header_id[i] = flag
    flag += 1

if outfile:
    _o = open(outfile, 'w')

#print('\t'.join(header))

hd = '\t'.join(header)
if outfile:
    _o.write(hd + '\n')

else:
    print(hd)

for i, j in output:
    out = ['-'] * len(header)
    out[0] = i.split(os.sep)[-1].split('_')[0].split('.')[0]
    for (k, k1) in j:

        out[header_id[k]] = k1

    #print('\t'.join(out))

    if outfile:
        #_o.write()
        _o.write('\t'.join(out) + '\n')
    else:
        print('\t'.join(out))

if outfile:
    _o.close()
