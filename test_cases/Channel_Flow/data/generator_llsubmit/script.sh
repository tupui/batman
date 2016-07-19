#!/bin/bash 
module load python

mkdir cfd-output-data

python function.py > function.out
