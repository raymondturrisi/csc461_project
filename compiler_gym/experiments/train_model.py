import compiler_gym
from compiler_gym.datasets import benchmark
import gym
import calendar
import time
from datetime import datetime
import torch
import sys

from stable_baselines3 import DQN
from compiler_gym.wrappers import TimeLimit
from compiler_gym.wrappers import CycleOverBenchmarks

def train_env():
    env = compiler_gym.make(
        "llvm-v0",
        reward_space="IrInstructionCountOz",
        observation_space="Autophase"
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
        observation_space="Autophase"
    )
    dataset = env.datasets["cbench-v1"]
    return env
    


def train(model, file):
    episodes = 1000000 # The number of episodes used to learn
    episode_length = 500 # The maximum number of transformations
    error_count = 0
    for i in range(1, episodes+1):
        try:
            model.learn(total_timesteps=episode_length)
        except:
            print("Error running model. Most likely failed to parse the LLVM bitcode for some reason")
            error_count += 1
        
        if i % 1000 == 0:
            print ("Step " + str(i))
            current_time = datetime.now().strftime("%Y_%m_%d_%H%M")
            print("Current Time =", current_time)
            model.save(file)
    
    print("DONE")
    print("There were " + str(error_count) + " error(s) when trying to parse the LLVM bitcode")

            


if __name__ == "__main__":
    env = train_env()
    # 74%
    # hyperparams = { 
    #     "env": env,
    #     "policy": 'MlpPolicy',
    #     "gamma": 0.999,
    #     "learning_rate": 0.0020846018394760695,
    #     "batch_size": 32,
    #     "buffer_size": 10000,
    #     "train_freq": 4,
    #     "gradient_steps": 1,
    #     "exploration_fraction": 0.49275615218804686,
    #     "exploration_final_eps": 0.031248308796828307,
    #     "target_update_interval": 1,
    #     "learning_starts": 20000,
    #     "policy_kwargs": dict(net_arch=[64]),
    #     "device": "cuda",
    #     "verbose": 1,
    # }
    # 84%
    hyperparams = { 
        "env": env,
        "policy": 'MlpPolicy',
        "gamma": 0.9838176115174185,
        "learning_rate": 0.0009579913102614245,
        "batch_size": 256,
        "buffer_size": 8192,
        "train_freq": 128,
        "gradient_steps": 128,
        "exploration_fraction": 0.21236553254173923,
        "exploration_final_eps": 0.058250066257957485,
        "target_update_interval": 100,
        "learning_starts": 5000,
        "policy_kwargs": dict(net_arch=[128, 128, 128, 128, 128], activation_fn=torch.nn.Tanh),
        "device": "cuda",
        "verbose": 1,
    }

    model = None
    file = "models/DQN_model_4_84"
    if len(sys.argv) > 1:
        model = DQN.load(sys.argv[1], env=env)
        file = sys.argv[1]
    else:
        model = DQN(**hyperparams)
    train(model,file)
