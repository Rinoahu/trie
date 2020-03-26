#!/bin/bash

folder=$1

# install packages
#conda install -y -c bioconda -c conda-forge multiqc
#conda install -y -c bioconda fastqc trimmomatic srst2 spades blast parsnp amos velvet
#conda install -y genblasta genblastg
#conda install -c dfornika resfinder
#conda install -c bioconda staramr
python -mpip install ncbi-genome-download
exit 0

# download vf
mkdir -p vf_db
srst2_cls=$PWD/tools/srst2/database_clustering
taxon=Bacillus
cd vf_db
#wget -c http://www.mgc.ac.cn/VFs/Down/VFDB_setB_nt.fas.gz
gunzip VFDB_setB_nt.fas.gz
python $srst2_cls/VFDBgenus.py --infile VFDB_setB_nt.fas --genus $taxon
cd-hit -i $taxon\.fsa -o $taxon\_cdhit90 -c 0.9 > $taxon\_cdhit90.stdout
python $srst2_cls/VFDB_cdhit_to_csv.py --cluster_file $taxon\_cdhit90.clstr \
 --infile $taxon\.fsa --outfile $taxon\_cdhit90.csv
python $srst2_cls/csv_to_gene_db.py -t $taxon\_cdhit90.csv \
 -o $taxon\_VF_clustered.fasta -s 5
cd ..

exit 0;

# install ssuis st
#git clone https://github.com/streplab/SsuisSerotyping_pipeline

# test ssuis st
rm -rf recN/ Serotyping/ MLST/ Virulence/
F_pe=_R1
R_pe=_R2
echo "perl ./SsuisSerotyping_pipeline/Ssuis_serotypingPipeline.pl --fastq_directory /home/xiaohu/research/mlst_test/data/gp_b4_raw_reads/M-B4-161_S24_L001_output --scoreName Score_output_type1 --forward $F_pe --reverse $R_pe &> log.txt" > run_test.sh


F_pe=_S3_L001_R1_001
R_pe=_S3_L001_R2_001
#perl ./SsuisSerotyping_pipeline/Ssuis_serotypingPipeline.pl --fastq_directory $PWD/serotyping_testing/type1over2 --scoreName Score_output_type1over2 --forward $F_pe --reverse $R_pe &> log.txt

#exit 0;

# test srst2
#srst2 --output test --input_pe $F_pe $R_pe --mlst_db ./mlst_db/Streptococcus_suis.fasta --mlst_definitions ./mlst_db/ssuis.txt --mlst_delimiter '_'


# get mlst database
mkdir -p mlst_db
cd mlst_db
#getmlst.py --species "Streptococcus suis"
cd ..

# adapter
ad_seq=/home/xiaohu/research/mlst_test/data/supporting_DB/adapters/NexteraPE-PE.fa

cdir=$PWD
# trim the reads
for i in $folder/*_R1_*fastq.gz
do
    j=`echo $i | sed 's/_R1_/_R2_/g'`
    prefix=`echo $i | awk -F"_R1_" '{print $1}'`
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
    echo $phred
    trimmomatic PE -threads 16 -$phred $F_pe $R_pe $F_pe\_clean.fq.gz $F_pe\_unpaired.fq.gz $R_pe\_clean.fq.gz $R_pe\_unpaired.fq.gz ILLUMINACLIP:$ad_seq:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36 HEADCROP:15

    # assembly
    #spades.py --isolate -o spades_asm_dir -1 $F_pe\_clean.fq.gz -2 $R_pe\_clean.fq.gz

    # srst2

    # ssuis pipeline
    echo $PWD
    zcat $PWD/$F_pe\_clean.fq.gz > reads_R1.fastq
    zcat $PWD/$R_pe\_clean.fq.gz > reads_R2.fastq
	dbs=$cdir/SsuisSerotyping_pipeline
    echo "perl $cdir/SsuisSerotyping_pipeline/Ssuis_serotypingPipeline.pl \
        --fastq_directory $PWD --scoreName result \
        --serotype_db $dbs/Ssuis_Serotyping.fasta \
        --serotype_definitions $dbs/Ssuis_Serotyping_Definitions.txt \
        --cps2K $dbs/Ssuis_cps2K.fasta \
	    --MLST_db $dbs/Streptococcus_suis.fasta \
	    --MLST_definitions $dbs/ssuis.txt \
	    --recN_db $dbs/recN_full.fasta \
	    --Virulence_db $dbs/Virulence.fasta \
        --forward _R1 --reverse _R2 &> log.txt" > run_test.sh

    # search against customer's database
    #ggsearch36 -T 8 -Q -q -n -m 8 -E 1e-10 $gene $genome > out.txt

    break
done

