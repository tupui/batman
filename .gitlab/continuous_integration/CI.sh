which python
python --version

export MPLBACKEND="Agg"

find . -regex "\(.*__pycache__.*\|*.py[co]\)" -delete

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

if [ $fail -eq 1 ] ; then
    exit 1
fi
