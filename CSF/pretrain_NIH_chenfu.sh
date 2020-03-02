#!/bin/bash --login
#$ -cwd
#$ -j y
#$ -o logs
#$ -l nvidia_v100
#$ -pe smp.pe 8

module load apps/anaconda3/5.2.0/bin

export OMP_NUM_THREADS=$NSLOTS
mkdir -p logs

# check nvidia-smi
nvidia-smi 

## activate conda environment
# this also includes cuda and cuda toolkits

source activate /mnt/jw01-aruk-home01/projects/ra_challenge/tensorflow2.0_gpu


rm ~/localscratch/* -r
# run python training script
cd /mnt/jw01-aruk-home01/projects/ra_challenge/RA_challenge/RA2_alpine_lads/ra_joint_predictions
python pretrain_chest_CSF.py
