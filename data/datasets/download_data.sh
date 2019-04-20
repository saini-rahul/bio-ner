#!/bin/bash

Datasets="BC4CHEMD BC5CDR-IOB NCBI-disease-IOB JNLPBA"

for val in $Datasets; do
    mkdir -p $val
    cd $val
    wget "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/"$val"/train.tsv"
    wget "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/"$val"/test.tsv"
    wget "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/"$val"/devel.tsv"
    cd ../
done
 
