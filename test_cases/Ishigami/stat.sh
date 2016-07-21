#!/bin/bash
#SBATCH --partition debug
#SBATCH --time=00:10:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --job-name=JPOD
##SBATCH --mail-user ...@cerfacs.fr
##SBATCH --mail-type all
#SBATCH --share

module load application/openturns/1.7

cd ${SLURM_SUBMIT_DIR}

python ~/JPOD/jpod/ui.py scripts/task.py -o output/task -u -n 

