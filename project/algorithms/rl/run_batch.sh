#!/bin/bash

SAVE_DIR=$1
NODES=$2
EPS_PER_NODE=$3

LR = 0.00005
MEMORY_SIZE = 100000
UPDATE_TARGET = 10000

echo "CREATING weights/$SAVE_DIR"
mkdir weights/$SAVE_DIR
echo "==== HYPERPARAMETERS ===="
echo "LR $LR"
echo "MEMORY SIZE $MEMORY_SIZE"
echo "UPDATE TARGET $UPDATE_TARGET"
echo "========================="

echo "STARTING $NODES NODES"

for ((i=0; i<$NODES; i++))
do
echo "NODE $i STARTED"
qsub -v SAVE_DIR=$SAVE_DIR,JOB=$i,EPS=$EPS_PER_NODE,LR=$LR,MEMORY_SIZE=$MEMORY_SIZE,UPDATE_TARGET=$UPDATE_TARGET -l walltime=24:00:00 run.sh 
done
