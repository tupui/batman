#!/bin/bash
#SBATCH --partition prod
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --job-name=JPOD
##SBATCH --mail-user ...@cerfacs.fr
##SBATCH --mail-type all
#SBATCH --share

module load application/openturns/1.7
module load python/3.3.6

cd ${SLURM_SUBMIT_DIR}

python ~/JPOD/kernel/jpod/ui.py scripts/task.py -o output/task -u -n

