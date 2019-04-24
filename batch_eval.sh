#!/bin/bash

$SAVE=$1
python evaluation.py --save-dir $1 -r $2
python evaluation.py -env-n --save-dir $1 -r $2
python evaluation.py -env-b --save-dir $1 -r $2
