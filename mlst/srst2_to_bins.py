#!usr/bin/env python
import sys


qry = sys.argv[1]
f = open(qry, 'r')
print(f.readline()[:-1])
for i in f:
    j = i[:-1].split('\t')
    for k in range(1, len(j)):
        #if j[k] != '-':
        if j[k] != '-' and j[k] != '-f' and j[k] != '-?':
            j[k] = '+'
        else:
            j[k] = '-'

    print('\t'.join(j))


f.close()

