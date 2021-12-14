#!/bin/bash
COUNT = 1
while :
do
	python3 train_model.py COUNT $1
    COUNT = COUNT + 1
done
