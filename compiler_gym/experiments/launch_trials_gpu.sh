NAME="DQN_OPTUNA_GPU"
for i in {1..4}
do	
	screen -S "${NAME}_${i}" -d -m
	screen -S "${NAME}_${i}" -X stuff './model_watchdog_gpu.sh\r'
done
