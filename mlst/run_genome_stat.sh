#!/bin/bash

genome=$1
reads=$2

#for i in `find ../../../data/ss_b*_raw_reads/*_output -name *_scaffolds.fsa`
for i in `find $genome -name *_scf.fsa`
do
    x=`getN50 $i | awk '{printf "%s" (NR%5==0?RS:FS),$1" "$2}'`
    j=`echo $i | awk -F"_scf.fsa" '{print $1}' | awk -F"/" '{print $NF}'`
    hd=`echo $x | cut -f1,3,5,7,9,11 -d\ `
    value=`echo $x | cut -f2,4,6,8,10,12 -d\ `
    echo "Plate_num Raw_Seq_Name Read_Pair_Counts Coverage Assembly_file" $hd
    break
done


#for i in `find ../../../data/ss_b*_raw_reads/*_output -name *_scaffolds.fsa`
for i in `find $genome -name *_scf.fsa`
do
    fn=`echo $i | awk -F"/" '{print $NF}'`
    x=`getN50 $i | awk '{printf "%s" (NR%5==0?RS:FS),$1" "$2}'`
    j=`echo $i | awk -F"_scf.fsa" '{print $1}' | awk -F"/" '{print $NF}'`
    raw_name=`echo $j | awk -F"/" '{print $NF}'`
    value=`echo $x | cut -f2,4,6,8,10,12 -d\ `
    size=`echo $x | cut -f4 -d\ `

    M=`zcat $reads/$j\_*fastq.gz | awk 'NR%4==2' | wc -c -l`
    rows=`echo $M | cut -f1 -d\ `
    N=`echo $M | cut -f2 -d\ `
    #echo $rows $N
    cov=`echo $N/$size | bc`
    row=`echo $rows/2 | bc`
    batch=`echo $raw_name | awk -F"_" '{print $1}' | awk -F"-" '{print $NF}'`
    echo $batch $raw_name $row $cov\X $fn $value
    #break
done
