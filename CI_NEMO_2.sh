#!/bin/sh
#SBATCH --partition prod
#SBATCH --job-name batCI2
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --share

# On spike: send the job to the HPC scheduler and wait for completion
if [ ${HOSTNAME:0:4} != 'nemo' ] && [ ${HOSTNAME:0:4} != 'node' ] ; then
    nojob=$(ssh roy@nemo 'sbatch --share CI_NEMO_2.sh' | awk '{print $NF}') 
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
    
    ssh roy@nemo "cat slurm-$nojob.out"
    status=$(ssh roy@nemo "sacct --format=state -j $nojob | awk 'NR>3 {print $1}'")
    if [ $status = 'FAILED' ] ; then
        exit 1
    else
        exit 0
    fi
fi

# On HPC:
source activate bat_ci_2
export PYTHONPATH=
python --version

SCRATCH='/scratch/cfd/roy'
mkdir $SCRATCH/CI_BATMAN_2
tar -xf batman_ci_2.tar -C $SCRATCH/CI_BATMAN_2/.
rm batman_ci_2.tar CI_NEMO_2.sh
cd $SCRATCH/CI_BATMAN_2/batman

python setup.py build_fortran
python setup.py install
which batman

# launch test suite and coverage
pytest --cov=batman --cov-report term-missing --basetemp=./TMP_CI .
if [ $? -ne 0 ] ; then
    fail=1
else
    fail=0
fi

pip uninstall -y batman
rm -r $SCRATCH/CI_BATMAN_2

if [ $fail -eq 1 ] ; then
    exit 1
fi
