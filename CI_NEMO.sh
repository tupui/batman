#!/bin/sh
#SBATCH --partition prod
#SBATCH --job-name batci
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --share

# On spike: send the job to the HPC scheduler and wait for completion
if [ ${HOSTNAME:0:4} != 'nemo' ]
then
    nojob=$(ssh roy@nemo 'sbatch --share CI_NEMO.sh' | awk '{print $NF}') 
    echo $nojob
    
    r=0
    while [ $r -eq 0 ]
    do
      resu=$(squeue -j $nojob -h -o %T)
      if [ ! -z "$resu" ] ; then
         sleep 30
      else
         r=1
      fi
    done

    ssh roy@nemo 'cat slurm*'

    exit 0
fi

# On HPC: launch test suite and coverage
source activate bat_ci
python --version

SCRATCH='/scratch/cfd/roy'
mkdir $SCRATCH/CI_BATMAN
tar -xf batman_ci.tar -C $SCRATCH/CI_BATMAN/.
cd $SCRATCH/CI_BATMAN/batman

python setup.py build_fortran
python setup.py install
which batman

coverage run --omit=. -m pytest --ignore=test_cases/Mascaret
if [ $? -ne 0 ]
then
    exit 1
fi

coverage report

pip uninstall -y batman
rm -r $SCRATCH/CI_BATMAN
