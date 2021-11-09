#!/bin/bash

##DEPENDENT ON CSMITH : https://github.com/csmith-project/csmith
echo "lower bound: $1"
echo "upper bound: $2"

read -p 'y/n:' EXECUTE

if [[ "$EXECUTE" == "y" ]] 
then
TIMESTAMP=$(date +"%Y-%m-%d_%H%M")
mkdir "${TIMESTAMP}_dir"
	for (( i=$1; i<=$2; i++ ))
	do
		echo "${TIMESTAMP}_dir/rand_c_idx_${i}_${TIMESTAMP}.c"
		csmith > "${TIMESTAMP}_dir/rand_c_idx_${i}_${TIMESTAMP}.c"
	done

	else
	echo "Not executed.."
fi

