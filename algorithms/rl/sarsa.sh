#!/bin/bash

cd project/algorithms/rl
python -m sarsa --save-dir $SAVE_DIR --job $JOB --episodes $EPS --learning-rate $LR 
