#!/bin/bash
for COUNT in {1..2200}
do
    python3 train_model_DQN.py $COUNT $1
    COUNT=$(( $COUNT + 1 ))
    echo "One loop done, Count is now at $COUNT"
done
