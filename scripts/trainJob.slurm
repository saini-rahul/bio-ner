#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobExample4       #Set the job name to "JobExample4"
#SBATCH --time=10:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1                   #Request 1 task
#SBATCH --mem=16G                  #Request 2560MB (2.5GB) per node
#SBATCH --output=Example4Out.%j      #Send stdout/err to "Example4Out.[jobID]"
##SBATCH --gres=gpu:1                 #Request 1 GPU per node can be 1 or 2
##SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
ml Python/3.5.2-intel-2017A
#source /scratch/user/rssaini/keras-contrib/tdenv/bin/activate
source /scratch/user/rssaini/multi_out_ner/scripts/rssaini/bin/activate
python train.py
deactivate
