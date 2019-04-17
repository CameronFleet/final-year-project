#!/bin/bash

for ((i=0; i<=$2; i++))
do
echo "Starting Node : $i"
qsub -v SAVE=$1_$i,EPS=$3 -l walltime=24:00:00 run.sh 
done

