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


%sLEGEND_TITLE,Gene Typing
%sLEGEND_SHAPES,2,2,2
%sLEGEND_COLORS,#377EB8,#f76116,#00b300
%sLEGEND_LABELS,Virulence Factor,vtaA1-A11,vtaAA12-A13



MARGIN,50

DATA



"""
red='#E41A1C'
orange='#f76116'
blue='#377EB8'
green='#00b300'

first3 = set('EF,Muramidase-released_protein,sly'.split(','))

first3 = 'vtaA1,vtaA2,vtaA3,vtaA4,vtaA5,vtaA6,vtaA7,vtaA8,vtaA9,vtaA10,vtaA11,vtaA12,vtaA13'.split(',')

cols = {}
for i in first3[:-2]:
    cols[i] =  orange

for i in first3[-2:]:
    cols[i] = green

first3_dict = {}
for i in range(len(first3)):
    first3_dict[first3[i]] = i + 1

f = open(qry, 'r')
hd = f.readline()[:-1].split('\t')
n = len(hd[1:])
if _type == 're':
    idx = list(range(len(hd)))
    a = ','.join(['2'] * n)
    b = ','.join(hd[1:])
    c = ','.join([red] * n)
    print(header%('antibiotic', '#00ff00', a, b, c, '#', '#', '#', '#'))
    

elif _type == 'vf':

    first3_dict[hd[0]] = 0

    hd_new = [[first3_dict.get(y, 10000), y, z]for z, y in enumerate(hd)]
    hd_new.sort(key=lambda x: x[0])

    #print('hd', hd)
    #print('new', hd_new)
    #print(hd[59], hd[64], hd[65])

    a = ','.join(['2'] * n)
    b = ','.join([y for x,y,z in hd_new[1:]])
    idx = [z for x,y,z in hd_new]
    c = [blue] * (len(hd))
    for i in range(len(hd)):
        x, y, z = hd_new[i]
        c[i] = cols.get(y, blue)

    c = c[1:]

    c = ','.join(c)
    print(header%('VF', '#0000ff', a, b, c, '', '', '', ''))

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

