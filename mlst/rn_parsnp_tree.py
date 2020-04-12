#!usr/bin/env python
import re
import sys


qry = sys.argv[1]

#pt = re.compile(r'\'_S\d+\':')
pt = re.compile(r'\'[\w|\_|\-|\.]+\'')

tree = open(qry, 'r').read().strip()

new_tree = tree
for i in pt.findall(tree):
    j = i[1:-1].split('_S')[0]
    new_tree = new_tree.replace(i, j, 1)

print(new_tree)

#print(tree)
#new_tree = pt.sub(':', tree)
#print(new_tree.replace('Ssuis', 'S.suis').replace('_ISU', 'ISU').replace('_LDO', 'LDO').replace('S.suis_SRD478_LGKK00000000', 'S.suisSRD478LGKK00000000'))
