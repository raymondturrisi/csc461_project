NAME="DQN_OPTUNA_"
for i in {1..8}
do	
	screen -S "${NAME}_${i}" -d -m
	screen -S "${NAME}_${i}" -X stuff './model_watchdog.sh\r'
done
