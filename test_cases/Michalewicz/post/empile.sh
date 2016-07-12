cat -t ../output/task/predictions/*/header.py > HEAD.dat 
cat -t ../output/task/predictions/*/function.dat > FUNC.dat
cat $(find ../output/task/snapshots/*/jpod-data/header.py -name "*" | sort -V) > SAMP.dat
