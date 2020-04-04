#!usr/bin/env python
import sys

try:
    qry, ref = sys.argv[1:3]
except:
    print('python this.py qry ref')
    raise SystemExit()


# get header
f0 = open(qry, 'r')
h0 = f0.readline()[:-1].split('\t')

f1 = open(ref, 'r')
h1 = f1.readline()[:-1].split('\t')


h = ['Sample'] + sorted(set(h0[1:] + h1[1:]))
h_idx = {}
flag = 0
for i in h:
    h_idx[i] = flag
    flag += 1

table = {}
for i in f0:
    j = i[:-1].split('\t')
    if j[0] not in table:
        table[j[0]] = ['-'] * len(h)
        table[j[0]][0] = j[0]

    for k in j[1:]:
         table[j[0]][0] = j[0]
       

