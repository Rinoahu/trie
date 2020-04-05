#!usr/bin/env python
# convert binary to itol gene typing

import sys

qry = sys.argv[1]
try:
    _type = sys.argv[2]
except:
    _type = 're'


if _type == 're':
    f = open(qry, 'r')
    hd = f.readline()[:-1].split('\t')
    print(hd)

