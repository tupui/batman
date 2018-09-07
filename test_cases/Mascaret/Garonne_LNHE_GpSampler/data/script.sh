#!/bin/sh
#SBATCH --partition debug
#SBATCH --time=00:00:10
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --job-name=1Dfunction
#SBATCH --share

# mkdir cfd-output-data

python function.py > function.out
