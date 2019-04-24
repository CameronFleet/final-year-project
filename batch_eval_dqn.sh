#!/bin/bash

SAVE=$1
DQNPATH=$2
python evaluation.py --save $SAVE -r -dqn $DQNPATH
python evaluation.py -env-n --save $SAVE -r -dqn $DQNPATH
