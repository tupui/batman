#!/bin/sh
#SBATCH --partition prod
#SBATCH --job-name batci
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --share

# On spike: send the job to the HPC scheduler and wait for completion
if [ ${HOSTNAME:0:4} != 'nemo' ] && [ ${HOSTNAME:0:4} != 'node' ] ; then
    nojob=$(ssh roy@nemo 'sbatch --share CI_NEMO.sh' | awk '{print $NF}') 
    echo 'Job number: ' $nojob
    
    r=0
    while [ $r -eq 0 ]
    do
      resu=$(ssh roy@nemo "squeue -j $nojob -h -o %T")
      if [ ! -z "$resu" ] ; then
         sleep 30
      else
         r=1
      fi
    done
    
    ssh roy@nemo 'cat slurm*'
    status=$(ssh roy@nemo "sacct --format=state -j $nojob | awk 'NR>3 {print $1}'")
    if [ $status = 'FAILED' ] ; then
        exit 1
    else
        exit 0
    fi
fi

# On HPC:
source activate bat_ci
python --version

SCRATCH='/scratch/cfd/roy'
mkdir $SCRATCH/CI_BATMAN
tar -xf batman_ci.tar -C $SCRATCH/CI_BATMAN/.
cd $SCRATCH/CI_BATMAN/batman

python setup.py build_fortran
python setup.py install
which batman

# launch test suite and coverage
coverage run -m pytest --basetemp=./TMP_CI --ignore=test_cases/Mascaret .
if [ $? -ne 0 ] ; then
    fail=1
else
    fail=0
fi
coverage report -m

pip uninstall -y batman
rm -r $SCRATCH/CI_BATMAN

if [ $fail -eq 1 ] ; then
    exit 1
fi