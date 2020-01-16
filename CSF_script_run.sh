#!/bin/bash --login
#$ -cwd
#$ -o logs
#$ -l nvidia_v100
#$ -pe smp.pe 8


export OMP_NUM_THREADS=$NSLOTS


## activate conda environment
# this also includes cuda and cuda toolkits


