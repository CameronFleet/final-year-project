#!/bin/bash

cd project/algorithms/rl
python -m advanced_deep_q --save-dir $SAVE_DIR --job $JOB --episodes $EPS --learning-rate $LR --memory-size $MEMORY_SIZE --update-target $UPDATE_TARGET
