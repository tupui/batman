#!/bin/sh
#SBATCH --partition debug
#SBATCH --time=00:01:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --job-name=1Dfunction
#SBATCH --mail-user=jouhaud@cerfacs.fr
#SBATCH --share

mkdir cfd-output-data

python function.py > function.out
