#!/bin/bash
if [ $1 == "DQN"]
then
    for COUNT in {1..2200}
    do
        python3 train_model_DQN.py $COUNT $2
        COUNT=$(( $COUNT + 1 ))
        echo "One loop done, Count is now at $COUNT"
    done
fi

if [ $1 == "A2C"]
then
    for COUNT in {1..2200}
    do
        python3 train_model_A2C.py $COUNT $2
        COUNT=$(( $COUNT + 1 ))
        echo "One loop done, Count is now at $COUNT"
    done
fi
