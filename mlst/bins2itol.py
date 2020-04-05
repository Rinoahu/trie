#!usr/bin/env python
# convert binary to itol gene typing

import sys

qry = sys.argv[1]
try:
    _type = sys.argv[2]
except:
    _type = 're'


header="""DATASET_BINARY

SEPARATOR COMMA

#label is used in the legend table (can be changed later)
DATASET_LABEL,%s

#dataset color (can be changed later)
COLOR,%s

SHOW_LABELS,1



FIELD_SHAPES,%s

FIELD_LABELS,%s

FIELD_COLORS,%s


MARGIN,50

DATA



"""
red='#E41A1C'
orange='#f76116'
blue='#377EB8'

first3 = set('EF,Muramidase-released_protein,sly'.split(','))

f = open(qry, 'r')
hd = f.readline()[:-1].split('\t')
n = len(hd[1:])
if _type == 're':
    idx = list(range(len(hd)))
    a = ','.join(['2'] * n)
    b = ','.join(hd[1:])
    c = ','.join([red] * n)
    print(header%('antibiotic', '#00ff00', a, b, c))
    

elif _type == 'vf':
    fst3 = []
    rest = []
    for i in range(1, len(hd)):
        if hd[i] in first3:
            fst3.append(i)
        else:
            rest.append(i)

    idx = [0] + fst3 + rest

    a = ','.join(['2'] * n)

    hd_new = [hd[elem] for elem in idx]
    b = ','.join(hd_new[1:])

    c = [orange] * len(fst3) + [blue] * len(rest)
    c = ','.join(c)
    print(header%('VF', '#0000ff', a, b, c))

else:
    raise SystemExit()



for i in f:
    j = i[:-1].split('\t')
    out = [j[elem] for elem in idx]
    for i in range(1, len(out)):
        if out[i] == '+':
            out[i] = 1
        elif out[i] == '-':
            out[i] = -1
        else:
            out[i] = 0


    print(','.join(map(str, out)))

f.close()

