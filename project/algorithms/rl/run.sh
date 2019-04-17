#!/bin/bash

pwd
cd project/algorithms/rl
python -m advanced_deep_q --save $SAVE --episodes $EPS
