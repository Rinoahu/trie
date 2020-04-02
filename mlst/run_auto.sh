#!/bin/bash

rm log.txt

for i in `ls data/ | grep ss_`
do
    echo $i
    sero_db=$PWD/data/supporting_DB/DB/SSuis/sero_DB/HPS_sero.fasta
    vf_db=$PWD/data/supporting_DB/DB/SSuis/VF_DB/Strep_VF_srst2_2020-02-07.fasta
    res_db=$PWD/data/supporting_DB/DB/Resistance_RE/oneline_ARGannot_r3_mcr8_2019-06-28.fasta
    ssuis_exe=$PWD/SsuisSerotyping_pipeline
    bash run_all.sh $PWD/data/$i $PWD/data/reference/Streptococcus.fsa $sero_db $vf_db $res_db $ssuis_exe &>> log.txt
    #bash run_all.sh ./data/$i $PWD/data/reference/Streptococcus.fsa

done

exit 0

for i in `ls data/ | grep gp_`
do
    echo $i

    sero_db=./data/supporting_DB/DB/GPS/sero_DB/HPS_sero.fasta
    vf_db=./data/supporting_DB/DB/GPS/VF_DB/HPS_sero.fasta
    res_db=./data/supporting_DB/DB/Resistance_RE/oneline_ARGannot_r3_mcr8_2019-06-28.fasta
    ssuis_db=./SsuisSerotyping_pipeline

    bash run_all.sh ./data/$i $PWD/data/reference/Glaesserella.fsa $sero_db $vf_db $res_db $ssuis_db
done
