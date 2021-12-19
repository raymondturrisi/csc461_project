import compiler_gym
from compiler_gym.datasets import benchmark
from compiler_gym.wrappers.datasets import RandomOrderBenchmarks
import gym
import calendar
import time
from datetime import datetime
import torch
import sys

from stable_baselines3 import A2C
from compiler_gym.wrappers import TimeLimit
from compiler_gym.wrappers import CycleOverBenchmarks

from itertools import islice


def train_env(count):
    env = compiler_gym.make(
        "llvm-v0",
        reward_space="IrInstructionCountOz",
        observation_space="Autophase"
    )

    del env.datasets["cbench-v1"]
    del env.datasets["generator://csmith-v0"]
    del env.datasets["generator://llvm-stress-v0"]
    train_benchmarks = env.datasets.benchmarks()
    train_benchmarks = list(islice(train_benchmarks, (count-1)*5000, count*5000))

    env = TimeLimit(env, max_episode_steps=500)
    env = RandomOrderBenchmarks(env, train_benchmarks)
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
    episodes = 5000 # The number of episodes used to learn
    episode_length = 500 # The maximum number of transformations
    error_count = 0
    for i in range(1, episodes+1):
        try:
            model.learn(total_timesteps=episode_length)
        except:
            print("Error running model. Most likely failed to parse the LLVM bitcode for some reason")
            error_count += 1
        print(i)
        if i % 1000 == 0:
            print ("Step " + str(i))
            current_time = datetime.now().strftime("%Y_%m_%d_%H%M")
            print("Current Time =", current_time)
            model.save(file)
    
    print("DONE")
    print("There were " + str(error_count) + " error(s) when trying to parse the LLVM bitcode")

            


if __name__ == "__main__":
    env = train_env(int(sys.argv[1]))
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
        "ent_coef": 0.0002060122177296989,
        "gae_lambda": 0.8,
        "gamma": 0.99,
        "learning_rate": 2.9050653862434115e-05,
        "lr_schedule": linear, 
        "max_grad_norm": 5, 
        "normalize_advantage": True,
        "n_steps": 128, 
        "ortho_init": True, 
        "use_rms_prop": True, 
        "vf_coef": 0.38376565620472664,
        "policy_kwargs": dict(net_arch=[128, 128, 128, 128, 128], activation_fn=torch.nn.Tanh),
        "device": "cuda",
        "verbose": 1,
    }

    model = None
    file = "models/A2C_model_"+ str(int(sys.argv[2])%5)
    if len(sys.argv) > 2:
        model = A2C.load(sys.argv[2], env=env)
        file = sys.argv[2]
    else:
        model = A2C(**hyperparams)
    train(model,file)
