#!usr/bin/env python
import re
import sys


qry = sys.argv[1]

pt = re.compile(r'_S\d+:')


tree = open(qry, 'r').read().strip()
#print(tree)
new_tree = pt.sub(':', tree)
print(new_tree.replace('Ssuis', 'S.suis').replace('_ISU', 'ISU').replace('_LDO', 'LDO').replace('S.suis_SRD478_LGKK00000000', 'S.suisSRD478LGKK00000000'))
