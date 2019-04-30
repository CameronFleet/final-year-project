#!/bin/bash

SAVE=$1
EST=$2
EST_PATH=$3

ECHO "NOMINAL TEST" >> evaluation/$SAVE.txt
python evaluation.py --save $SAVE -r -e $EST -p $EST_PATH

ECHO "NOISY TEST" >> evaluation/$SAVE.txt
python evaluation.py -env-n --save $SAVE -r -e $EST -p $EST_PATH

ECHO "ACCELERATION = 1 TEST" >> evaluation/$SAVE.txt
python evaluation.py -env-b --save $SAVE -r -e $EST -p $EST_PATH
