#!/bin/bash
#SBATCH --partition debug
#SBATCH -J c80_8
# Nombre de noeud : NBNODE
#SBATCH -N 1
# Nombre de processus MPI par noeud : ntaskpernode
#SBATCH --ntasks-per-node=2
# Mettre adresse email
#SBATCH --share

cd $SLURM_SUBMIT_DIR

source ~elsA/ELSA/setupelsa_v3502.sh

export OMP_NUM_THREADS=1
export MALLOC_MMAP_MAX_=0
export MALLOC_TRIM_THRESHOLD_=-1
export FORT_BUFFERED=true
export decfort_dump_flag=true

export ELSA_NOLOG=ON
export ELSA_MPI_LOG_FILES=OFF
export ELSA_ALLOW_OBSOLETE=ON

mpirun -np $SLURM_NTASKS -env PYTHONPATH=$PYTHONPATH -env LD_LIBRARY_PATH=$LD_LIBRARY_PATH `which elsA.x` rae_mg.py
