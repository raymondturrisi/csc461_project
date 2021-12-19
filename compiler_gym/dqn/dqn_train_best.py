#general imports
from datetime import datetime
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import compiler_gym                      # imports the CompilerGym environments
from compiler_gym.wrappers import CycleOverBenchmarks

from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy # Evaluation method used by FB for leaderboard
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import TimeLimit, CycleOverBenchmarks, RandomOrderBenchmarks
import torch as tc
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_device
from itertools import islice
import names

def get_trial_params():
    # configuring the environment parameters with big tuna
    env_config = {"id":"llvm-ic-v0", 
                  "observation_space":"InstCountNorm", 
                  "reward_space":"IrInstructionCountOz"
                  }

    env = compiler_gym.make(**env_config)

    datasets = env.datasets.benchmarks()
    training_data = list(islice(datasets, 1))
    
    env = CycleOverBenchmarks(env,training_data)
    env = TimeLimit(env, 100)

    # Currently, parameters only for an MLP, I don't think another is appropriate atm

    #number of layers
    net_arch = [128, 128]
    #activation function
    activation_fn = tc.nn.Tanh
    #optimizer - read below
    optimizer_class = tc.optim.SGD 

    #for mlp
    policy_kwargs = {
        "net_arch":net_arch,
        "activation_fn":activation_fn,
        "optimizer_class":optimizer_class,
        "optimizer_kwargs":None
    }
    # Read: https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6

    # Params for DQN Model
    model_config = {
        "env":env,
        "learning_rate":0.0006760923694541781,
        "batch_size":256,
        "buffer_size":100000,
        "gamma":0.995,
        "exploration_final_eps": 0.1796559659051645,
        "exploration_fraction":0.48572600600279464,
        "learning_starts":0,
        "target_update_interval":15000,
        "device":"auto",
        "policy":"MlpPolicy",
        "policy_kwargs":policy_kwargs
    }
    #return env_config, model_config, policy_kwargs
    return model_config, env_config


from stable_baselines3.common.monitor import Monitor

def make_test_env(env_config, model_config) -> compiler_gym.envs.CompilerEnv:

    env = compiler_gym.make(**env_config)

    env = TimeLimit(env, max_episode_steps=100)

    dataset = env.datasets["cbench-v1"] # Small dataset

    train_benchmarks = list(dataset) # N_bechmarks total benchmarks the dataset

    env = CycleOverBenchmarks(env, train_benchmarks)
    #env = DummyVecEnv([lambda: env])
    env = Monitor(env)
    return env 


def eval_model_on_compilergym_benchmark(model, env_config, model_config):
    # Define the test environment.
    #test_env = make_test_env(env_config, model_config)

    # Run the evaluation helper method from SB3.
    mean_reward_per_episode, _ = evaluate_policy(model, make_test_env(env_config, model_config))

    return mean_reward_per_episode


def main():
    ts = datetime.now().strftime("%Y_%m_%d_%H%M")
    trial_name = names.get_full_name().replace(" ", "_")

    print(f"{ts}: Made new trial. Hello, {trial_name}.")

    model_config, env_config = get_trial_params()

    print(f"Model Params:\n{model_config}")
    print(f"Env. Params:\n{env_config}")
    model = DQN(**model_config)


    total_epochs = 100000
    training_stride = 5000

    for epochs in range(training_stride,total_epochs+1,training_stride):
        model.learn(total_timesteps=training_stride)

        score = eval_model_on_compilergym_benchmark(model, env_config=env_config, model_config=model_config)

        ts = datetime.now().strftime("%Y_%m_%d_%H%M")
        print(f"{ts}: {trial_name} is on: {epochs}/{total_epochs}")
        print(f"{trial_name} evaluation Score: {score}")
    model.save(f"models/{trial_name}-{ts}")


main()