. venv/bin/activate

which python
python --version

python setup.py develop
which batman

# launch test suite and coverage

circleci tests glob tests/*.py | circleci tests split --split-by=timings | xargs coverage run -m pytest --basetemp=./TMP_CI
#coverage run -m pytest --basetemp=./TMP_CI tests # test_cases

status=$?

coverage html
mkdir test-reports
mv htmlcov test-reports/coverage-report

exit $status
