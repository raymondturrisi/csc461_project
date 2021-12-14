#!/bin/bash
COUNT = 1
while :
do
	./dqn_tests.py COUNT $1
    COUNT = COUNT + 1
done
