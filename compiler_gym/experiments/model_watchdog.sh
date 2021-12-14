#!/bin/bash
COUNT=1
if [ $1 = "DQN"]
then
    while :
    do
        python3 train_model_DQN.py $COUNT $2
        COUNT = COUNT + 1
    done
fi

if [ $1 = "A2C"]
then
    while :
    do
        python3 train_model_A2C.py $COUNT $2
        COUNT = COUNT + 1
    done
fi
