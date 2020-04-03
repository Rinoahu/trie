#!/bin/bash

shell_folder=$(dirname $(readlink -f "$0"))

#echo $shell_folder
#exit 0

# raw sequencing reads 
folder=$1

# $2 is the reference sequences
reference=$2

# $3 is the sero gene database
sero_db=$3

# $4 is vf gene database
vf_db=$4

# $5 is resist gene database
res_db=$5

echo 22 $res_db

# $6 is the ssuis_pipeline
ssuis_exe=$6

#if [[ $folder == "" ]]
if [[ "" ]]
then
    echo 'fold' $folder
else
    echo 'nope' $folder
fi
#exit 0

# install packages
#conda install -y -c bioconda -c conda-forge multiqc
#conda install -y -c bioconda fastqc trimmomatic srst2 spades blast parsnp amos velvet
#conda install -y genblasta genblastg
#conda install -c dfornika resfinder
#conda install -c bioconda staramr
#python -mpip install ncbi-genome-download
#exit 0

# download vf
mkdir -p vf_db
srst2_cls=$PWD/tools/srst2/database_clustering
#taxon=Haemophilus
#cd vf_db
#wget -c http://www.mgc.ac.cn/VFs/Down/VFDB_setB_nt.fas.gz
#gunzip VFDB_setB_nt.fas.gz
#python $srst2_cls/VFDBgenus.py --infile VFDB_setB_nt.fas --genus $taxon
#cd-hit -i $taxon\.fsa -o $taxon\_cdhit90 -c 0.9 > $taxon\_cdhit90.stdout
#python $srst2_cls/VFDB_cdhit_to_csv.py --cluster_file $taxon\_cdhit90.clstr \
# --infile $taxon\.fsa --outfile $taxon\_cdhit90.csv
#python $srst2_cls/csv_to_gene_db.py -t $taxon\_cdhit90.csv \
# -o $taxon\_VF_clustered.fasta -s 5
#cd ..

# install ssuis st
#git clone https://github.com/streplab/SsuisSerotyping_pipeline

# test srst2
#srst2 --output test --input_pe $F_pe $R_pe --mlst_db ./mlst_db/Streptococcus_suis.fasta --mlst_definitions ./mlst_db/ssuis.txt --mlst_delimiter '_'


# get mlst database
mkdir -p mlst_db
cd mlst_db
getmlst.py --species "Streptococcus suis"
cd ..

# adapter
ad_seq=$PWD/data/supporting_DB/adapters/NexteraPE-PE.fa

cdir=$PWD
# trim the reads
for i in $folder/*_R1_*fastq.gz
do
    j=`echo $i | sed 's/_R1_/_R2_/g'`
    prefix=`echo $i | awk -F"_R1_" '{print $1}'`
    #echo $prefix

    is=`echo $i | awk -F"/" '{print $NF}'`
    js=`echo $j | awk -F"/" '{print $NF}'`

    # create a folder for the results
    cd $cdir
    mkdir -p $prefix\_output
    cd $prefix\_output
    ln -sf ../$is ./
    ln -sf ../$js ./
    F_pe=$is
    R_pe=$js

    # trim the data
    phred=`zcat $is | head -n 40 | awk '{if(NR%4==0) printf("%s",$0);}' |  od -A n -t u1 | awk 'BEGIN{min=100;max=0;}{for(i=1;i<=NF;i++) {if($i>max) max=$i; if($i<min) min=$i;}}END{if(max<=74 && min<59) print "phred33"; else if(max>73 && min>=64) print "phred64"; else if(min>=59 && min<64 && max>73) print "Solexa+64"; else print "Unknown score encoding";}'`
    #echo $phred
    trimmomatic PE -threads 16 -$phred $F_pe $R_pe $F_pe\_clean.fq.gz $F_pe\_unpaired.fq.gz $R_pe\_clean.fq.gz $R_pe\_unpaired.fq.gz ILLUMINACLIP:$ad_seq:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36 HEADCROP:15

    # prepare reads
    #echo $PWD
    fn=`echo $prefix | awk -F"/" '{print $NF}'`
    echo $fn
    #zcat $PWD/$F_pe\_clean.fq.gz > $fn\_R1.fastq
    #zcat $PWD/$R_pe\_clean.fq.gz > $fn\_R2.fastq

    ln -sf $fn\_R1.fastq $fn\_R1_001.fastq
    ln -sf $fn\_R2.fastq $fn\_R2_001.fastq

    # assembly
    #spades.py --isolate -o spades_asm_dir -1 $fn\_R1.fastq -2 $fn\_R2.fastq
    #velveth velvet_asm_dir 31 -shortPaired -fastq -separate $fn\_R1.fastq $fn\_R2.fastq



    # blast to filter
    #ls -sh $PWD
    #legacy_blast.pl megablast -i ./spades_asm_dir/scaffolds.fasta -d $2 -o ./spades_asm_dir/scaffolds.fasta.m8 -m 8 -a 8 -e 1e-10
    #python $shell_folder/ctg_flt.py ./spades_asm_dir/scaffolds.fasta ./spades_asm_dir/scaffolds.fasta.m8 > $fn\_scaffolds.fsa
    legacy_blast.pl formatdb -i $fn\_scaffolds.fsa -p F


    # srst2 search against sero, vf, and resist gene.
    if [[ !$ssuis_exe ]]
    then
        if [[ $sero_db ]]
        then
            #srst2 --gene_db $sero_db --input_pe $fn\_R1_001.fastq $fn\_R2_001.fastq --output $fn\_srst2_sero --threads 8
            #legacy_blast.pl blastall -p blastn -i $sero_db -d $fn\_scaffolds.fsa -o $fn\_sero.m8 -m 8 -a 8 -e 1e-10
            tmp=`ls $fn\_srst2_sero__genes__*__results.txt`
            echo hello
        fi

        if [[ $vf_db ]]
        then
            srst2 --gene_db $vf_db --input_pe $fn\_R1_001.fastq $fn\_R2_001.fastq --output $fn\_srst2_vf --threads 8
            #legacy_blast.pl blastall -p blastn -i $vf_db -d $fn\_scaffolds.fsa -o $fn\_vf.m8 -m 8 -a 8 -e 1e-10
            tmp=`ls $fn\_srst2_vf__genes__*__results.txt`
            echo hello $tmp
        fi
    fi

    if [[ $res_db ]]
    then
        echo 155 $res_db 
        echo 156 $PWD

        echo "srst2 --gene_db $res_db --input_pe $fn\_R1.fastq $fn\_R2.fastq --output $fn\_srst2_res --threads 8"
        #srst2 --gene_db $res_db --input_pe $fn\_R1_001.fastq $fn\_R2_001.fastq --output $fn\_srst2_res --threads 8
        #legacy_blast.pl blastall -p blastn -i $res_db -d $fn\_scaffolds.fsa -o $fn\_res.m8 -m 8 -a 8 -e 1e-10
        tmp=`ls $fn\_srst2_res__genes__*__results.txt`
        echo hello $tmp
    fi

    # ssuis pipeline
	ssuis_exe=$cdir/SsuisSerotyping_pipeline
    if [[ $ssuis_exe ]]
    then
        #echo "perl $cdir/SsuisSerotyping_pipeline/Ssuis_serotypingPipeline.pl \
        """perl $ssuis_exe/Ssuis_serotypingPipeline.pl \
            --fastq_directory $PWD --scoreName $fn\_ssuis \
            --serotype_db $ssuis_exe/Ssuis_Serotyping.fasta \
            --serotype_definitions $ssuis_exe/Ssuis_Serotyping_Definitions.txt \
            --cps2K $ssuis_exe/Ssuis_cps2K.fasta \
	        --MLST_db $ssuis_exe/Streptococcus_suis.fasta \
	        --MLST_definitions $ssuis_exe/ssuis.txt \
    	    --recN_db $ssuis_exe/recN_full.fasta \
	        --Virulence_db $ssuis_exe/Virulence.fasta \
            --forward _R1 --reverse _R2 &> log.txt
        """
    fi
    # search against customer's database
    #ggsearch36 -T 8 -Q -q -n -m 8 -E 1e-10 $gene $genome > out.txt

    #break
done

exit 0
# ksnp3 to build tree
cd $folder
mkdir -p ksnp3_output/genomes
cd ksnp3_output/genomes
for i in `find ../../ -name *_scaffolds.fsa`
do
    ln -sf $i
done

cd ../
MakeKSNP3infile genomes inlist A
MakeFasta inlist fasta_genomes
Kchooser fasta_genomes
bestK=`grep optimum Kchooser.report | awk '{print $NF}' | cut -f1 -d\.`
kSNP3 -in inlist -outdir SNPs_genomes -k $bestK -ML -NJ -vcf  -CPU 8 -core -min_frac 0.5 | tee Log.txt
#kSNP3 -in inlist -outdir SNPs_genomes -k $bestK -ML -NJ -vcf  -CPU 8 -core -min_frac 0.5

