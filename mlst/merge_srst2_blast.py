#!usr/bin/env python
# this script is used to merge srst2 and blastn
import sys

try:
    srst2, blast8, fasta = sys.argv[1:4]
except:
    print('python this.py srst2 blast8 fasta")
    raise SysemExit()


