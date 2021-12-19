NAME="PPO_OPTUNA_"
for i in {1..5}
do	
	screen -S "${NAME}_${i}" -d -m
	screen -S "${NAME}_${i}" -X stuff 'bash ppo_watchdog.sh\r'
done