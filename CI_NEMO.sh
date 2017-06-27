#!/bin/sh

source activate bat_ci
python --version

SCRATCH='/scratch/cfd/roy'
mkdir $SCRATCH/CI_BATMAN
tar -xf batman_ci.tar -C $SCRATCH/CI_BATMAN/.
cd $SCRATCH/CI_BATMAN/batman

python setup.py build_fortran
python setup.py install
which batman

coverage run --omit=. -m pytest
if [ $? -ne 0 ]
then
    exit 1
fi

coverage report

pip uninstall -y batman
rm -r $SCRATCH/CI_BATMAN
