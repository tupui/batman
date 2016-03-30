cat -t ../output/static_sobol_task_0/predictions/*/header.py > HEAD.dat 
cat -t ../output/static_sobol_task_0/predictions/*/function.dat > FUNC.dat
cat $(find ../output/static_sobol_task_0/snapshots/*/jpod-data/header.py -name "*" | sort -V) > SAMP.dat
