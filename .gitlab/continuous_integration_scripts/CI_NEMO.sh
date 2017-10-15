#!/bin/sh
#SBATCH --partition prod
#SBATCH --job-name batCI3
#SBATCH --time=01:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --share

commit=$1
CI_HOME="$HOME/.ci_batman_$commit"
CI_SCRATCH="/scratch/cfd/roy/CI_BATMAN_$commit"


# On spike: send the job to the HPC scheduler and wait for completion
if [ ${HOSTNAME:0:4} != 'nemo' ] && [ ${HOSTNAME:0:4} != 'node' ]; then
    nojob=$(ssh roy@nemo "sbatch --share .ci_batman_$commit/CI_NEMO.sh $commit" | awk '{print $NF}') 
    echo "Job number: $nojob"
    
    resu=$(ssh roy@nemo "squeue -j $nojob -h -o %T 2>/dev/null")
    while [ "$resu" ]; do
        sleep 30
        resu=$(ssh roy@nemo "squeue -j $nojob -h -o %T 2>/dev/null")
    done
    
    ssh roy@nemo "cat slurm-$nojob.out"
    ssh roy@nemo "sacct --format=state -j $nojob | grep COMPLETED | tail -n 1"
    if [ $? -ne 0 ]; then
        exit 1
    else
        exit 0
    fi
fi

# On HPC:
conda create -yqf --name bat_ci_$commit --clone bat_ci
source activate bat_ci_$commit
export PYTHONPATH=
which python
python --version

mkdir $CI_SCRATCH
tar -xf $CI_HOME/batman_ci.tar -C $CI_SCRATCH
rm $CI_HOME/batman_ci.tar $CI_HOME/CI_NEMO.sh
rmdir --ignore-fail-on-non-empty $CI_HOME

cd $CI_SCRATCH/batman
python setup.py build_fortran
python setup.py install
which batman

# launch test suite and coverage
coverage run -m pytest --basetemp=./TMP_CI batman/tests test_cases
if [ $? -ne 0 ] ; then
    fail=1
else
    fail=0
fi
coverage report -m

source deactivate
conda remove -yq --name bat_ci_$commit --all
rm -r $CI_SCRATCH

if [ $fail -eq 1 ] ; then
    exit 1
fi
