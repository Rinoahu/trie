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

    for k, k1 in zip(h0[1:], j[1:]):
        if k1 != '-':
         table[j[0]][h_idx[k]] = k1


print('\t'.join(h))
tab = list(table.values())
tab.sort(key = lambda x: x[0])
for i in tab:
    print('\t'.join(i))


f0.close();
f1.close();
       

