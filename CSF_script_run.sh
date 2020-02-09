#!/bin/bash --login
#$ -cwd
#$ -j y
#$ -o logs
#$ -l nvidia_v100
#$ -pe smp.pe 8


export OMP_NUM_THREADS=$NSLOTS
mkdir -p logs

# check nvidia-smi
nvidia-smi 

## activate conda environment
# this also includes cuda and cuda toolkits

source activate /mnt/iusers01/jw01/mdefscs4/ra_challenge/tensorflow2.0_gpu

rm ~/localscratch/* -r
# run python training script

python pretrain_test_feet_landmarks.py
