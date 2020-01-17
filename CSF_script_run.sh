#!/bin/bash --login
#$ -cwd
#$ -o logs
#$ -l nvidia_v100
#$ -pe smp.pe 8


export OMP_NUM_THREADS=$NSLOTS
mkdir -p logs

# check nvidia_smi
nvidia_smi 

## activate conda environment
# this also includes cuda and cuda toolkits




# run python training script

python run_training_CSF.py
