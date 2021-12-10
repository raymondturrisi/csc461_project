NAME="DQN_OPTUNA_SESSION"
for i in {1..5}
do	
	screen -S "${NAME}_${i}" -d -m
	screen -S "${NAME}_${i}" -X stuff './model_watchdog.sh\r'
done
