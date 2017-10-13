#!/bin/sh
#SBATCH --partition prod
#SBATCH --job-name batCI2
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --share

commit=$1
CI_HOME="$HOME/.ci_batman_2/$commit"
CI_SCRATCH="/scratch/cfd/roy/CI_BATMAN_2_$commit"


# On spike: send the job to the HPC scheduler and wait for completion
if [ ${HOSTNAME:0:4} != 'nemo' ] && [ ${HOSTNAME:0:4} != 'node' ]; then
    nojob=$(ssh roy@nemo "sbatch --share $CI_HOME/CI_NEMO_2.sh $commit" | awk '{print $NF}') 
    echo "Job number: $nojob"
    
    resu=$(ssh roy@nemo "squeue -j $nojob -h -o %T 2>/dev/null")
    while [ "$resu" ]; do
        sleep 30
        resu=$(ssh roy@nemo "squeue -j $nojob -h -o %T 2>/dev/null")
    done
    
    ssh roy@nemo "cat slurm-$nojob.out"
    stat=$(ssh roy@nemo "sacct --format=state -j $nojob | awk 'NR>3 {print $1}'")
    if [ $stat = 'FAILED' ]; then
        exit 1
    else
        exit 0
    fi
fi

# On HPC:
conda create --name bat_ci_2_$commit --clone bat_ci_2
source activate bat_ci_2_$commit
export PYTHONPATH=
python --version

mkdir $CI_SCRATCH
tar -xf $CI_HOME/batman_ci_2.tar -C $CI_SCRATCH
rm $CI_HOME/batman_ci_2.tar $CI_HOME/CI_NEMO_2.sh
rmdir --ignore-fail-on-non-empty $CI_HOME

cd $CI_SCRATCH/batman
python setup.py build_fortran
python setup.py install
which batman

# launch test suite and coverage
pytest --cov --cov-report term-missing --basetemp=./TMP_CI .
if [ $? -ne 0 ] ; then
    fail=1
else
    fail=0
fi

source deactivate
conda remove --name bat_ci_2_$commit --all
rm -r $CI_SCRATCH

if [ $fail -eq 1 ] ; then
    exit 1
fi
