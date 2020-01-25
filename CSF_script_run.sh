#!/bin/bash --login
#$ -cwd
#$ -j y
#$ -o logs
#$ -l nvidia_v100
#$ -pe smp.pe 8


export OMP_NUM_THREADS=$NSLOTS
mkdir -p logs

# check nvidia_smi
nvidia_smi 

## activate conda environment
# this also includes cuda and cuda toolkits

source activate ~/communal_software/tensorflow_gpu


# run python training script

python pretrain_chest_CSF.py
