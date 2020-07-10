#!/usr/bin/env bash

input=$(cat coupling-dir/sample-space.json)
echo "# F1,F2,F3" > coupling-dir/sample-data.csv
echo "42,87,74,74" >> coupling-dir/sample-data.csv
echo "# 1,1,2" >> coupling-dir/sample-data.csv

echo 'done'
