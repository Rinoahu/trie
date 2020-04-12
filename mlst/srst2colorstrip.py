#!usr/bin/env python
# convert srst2 to color strip
import sys

Map, sero = sys.argv[1: 3]

map_dict = {}
f = open(Map, 'r')
for i in f:
    j = i[:-1].split('\t')
    k, v = j[:2]
    map_dict[k] = v

f.close()

f = open(sero, 'r')
hd = [elem.split('_')[0].split('sero')[-1] for elem in f.readline()[:-1].split('\t')]

for i in f:
    j = i[:-1].split('\t')
    sero = 'undetected'
    for k in range(1, len(j)):
        if not j[k].startswith('-'):
            sero = hd[k]

    color = map_dict.get(sero, '#D3D3D3')
    out = [j[0], color, sero]

    print(' '.join(out))



raise SystemExit()

for i in f:
    j = i[:-1].split('\t')
    k, v = j[:2]
    v = v.split('*')[0]
    v = 'failed' == v and 'unknown' or v
    color = map_dict[v]
    out = [k, color, v]
    print(' '.join(out))
