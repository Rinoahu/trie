#!usr/bin/env python
import sys

try:
    meta, batch = sys.argv[1: 3]
except:
    print('python this.py metadata batch_number')
    raise SystemExit()

isu_dict = {}

f = open(meta, 'r')
f.readline()
for i in f:
    j = i[:-1].split('\t')
    qid, case = j[0], j[2]
    sample = 'M-' + batch + '-' + qid
    isu = case + '--' + batch
    if isu in isu_dict:
        isu_dict[isu] += 1
        isu = isu + '-' + chr(isu_dict[isu])
    else:
        isu_dict[isu] = 97

    print(sample + '\t' + isu)
f.close()

