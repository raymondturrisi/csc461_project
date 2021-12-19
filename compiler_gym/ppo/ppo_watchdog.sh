#!/bin/bash

for i in {1..2200}
do
	python3 optuna_ppo_1.py $i
done
