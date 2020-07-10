which python
python --version

#python setup.py install
#which batman

# launch test suite and coverage

circleci tests glob batman/tests/*.py | circleci tests split --split-by=timings | xargs coverage run -m pytest --basetemp=./TMP_CI
#coverage run -m pytest --basetemp=./TMP_CI batman/tests # test_cases

status=$?

coverage html
mkdir test-reports
mv htmlcov test-reports/coverage-report

exit $status
