which python
python --version

export MPLBACKEND="Agg"

python setup.py install
which batman

# launch test suite and coverage
coverage run -m pytest --basetemp=./TMP_CI batman/tests # test_cases
if [ $? -ne 0 ] ; then
    fail=1
else
    fail=0
fi
coverage html
mkdir test-reports
mv htmlcov test-reports/coverage-report

if [ $fail -eq 1 ] ; then
    exit 1
fi
