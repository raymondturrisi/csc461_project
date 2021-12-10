import compiler_gym
from compiler_gym.datasets import benchmark
import gym
import calendar
import time

from stable_baselines3 import DQN
from compiler_gym.wrappers import TimeLimit
from compiler_gym.wrappers import CycleOverBenchmarks

def train_env():
    env = compiler_gym.make(
        "llvm-v0",
        reward_space="IrInstructionCountOz",
        observation_space="InstCountNorm"
    )

    del env.datasets["cbench-v1"]
    del env.datasets["generator://csmith-v0"]
    del env.datasets["generator://llvm-stress-v0"]
    train_benchmarks = env.datasets.benchmarks()

    env = TimeLimit(env, max_episode_steps=500)
    env = CycleOverBenchmarks(env, train_benchmarks)
    return env

def test_env():
    env = compiler_gym.make(
        "llvm-v0",
        reward_space="IrInstructionCountOz",
        observation_space="InstCountNorm"
    )
    dataset = env.datasets["cbench-v1"]
    return env
    


def train(env, model):
    episodes = 1000000 # The number of episodes used to learn
    episode_length = 500 # The maximum number of transformations
    for i in range(1, episodes+1):
        model.learn(total_timesteps=episode_length)
        
        if i % 5000 == 0:
            ts = calendar.timegm(time.gmtime())
            print ("Average Reward for Step " + str(i))
            model.save("models/DQN_model_1"+str(ts))
    
    print("DONE")

            


if __name__ == "__main__":
    env = train_env()
    hyperparams = { 
        "env": env,
        "policy": 'MlpPolicy',
        "gamma": 0.999,
        "learning_rate": 0.0020846018394760695,
        "batch_size": 32,
        "buffer_size": 10000,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0.49275615218804686,
        "exploration_final_eps": 0.031248308796828307,
        "target_update_interval": 1,
        "learning_starts": 20000,
        "policy_kwargs": dict(net_arch=[64]),
        "device": "cuda",
        "verbose": 1,
    }
    model = DQN(**hyperparams)
    train(env,model)