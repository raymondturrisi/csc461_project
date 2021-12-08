#!/usr/bin/python3
import os

import gym

import compiler_gym
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy # Evaluation method used by FB for leaderboard
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import TimeLimit

import numpy as np
import pandas as pd

from stable_baselines3 import DQN, SAC
from stable_baselines3.common.evaluation import evaluate_policy

import optuna

from itertools import islice
from compiler_gym.wrappers import CycleOverBenchmarks

from typing import Any, Dict

import calendar
import time
import logging
import sys

import random

with open(f"../func/not_cbench_cg.txt", "r") as benchmarks_files:
  benchmarks = benchmarks_files.readlines()


def make_env(env_config=None) -> compiler_gym.envs.CompilerEnv:
    """Make the reinforcement learning environment for this experiment.
    
      From FB example.
    """
    env = compiler_gym.make(
        "llvm-ic-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )

    
    # Finally, we impose a time limit on the environment so that every episode
    # for 5 steps or fewer. This is because the environment's task is continuous
    # and no action is guaranteed to result in a terminal state. Adding a time
    # limit means we don't have to worry about learning when an agent should 
    # stop, though again this limits the potential improvements that the agent
    # can achieve compared to using an unbounded maximum episode length.
    
    env = TimeLimit(env, max_episode_steps=1000)

    #del env.datasets["cbench-v1"]
    #del env.datasets["generator://csmith-v0"]
    #del env.datasets["generator://llvm-stress-v0"]
    #dataset = env.datasets.benchmarks() # Every dataset besides cbench

    # Each dataset has a `benchmarks()` method that returns an iterator over the
    # benchmarks within the dataset. Here we will use iterator sliceing to grab a 
    # handful of benchmarks for training and validation.

    #N_benchmarks = 5000

    #train_benchmarks = list(islice(dataset, N_benchmarks)) # N_bechmarks total benchmarks the dataset
    # train_benchmarks = list(dataset)
    # len(train_benchmarks) # , val_benchmarks = train_benchmarks[:50], train_benchmarks[50:]
    test_set = benchmarks[[random.randrange(0,len(benchmarks)) for i in range(0,5000)]]
    env = CycleOverBenchmarks(env, test_set)
    return env

def make_test_env(env_config=None) -> compiler_gym.envs.CompilerEnv:
    """
    Make the testing environment for evaluating approximate performance on the test set.
    
    Defining this function because the eval method by Facebook did not function as desired
    in tests.
    """
    env = compiler_gym.make(
        "llvm-ic-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )

    # Finally, we impose a time limit on the environment so that every episode
    # for 5 steps or fewer. This is because the environment's task is continuous
    # and no action is guaranteed to result in a terminal state. Adding a time
    # limit means we don't have to worry about learning when an agent should 
    # stop, though again this limits the potential improvements that the agent
    # can achieve compared to using an unbounded maximum episode length.
    env = TimeLimit(env, max_episode_steps=1000)

    dataset = env.datasets["cbench-v1"] # Small dataset

    # Each dataset has a `benchmarks()` method that returns an iterator over the
    # benchmarks within the dataset. Here we will use iterator sliceing to grab a 
    # handful of benchmarks for training and validation.

    train_benchmarks = list(dataset) # N_bechmarks total benchmarks the dataset

    env = CycleOverBenchmarks(env, train_benchmarks)

    return env


def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    NOTE: Comes from: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py
    Sampler for DQN hyperparams.
    :param trial:
    :return:
    """
    policy = 'MlpPolicy' # trial.suggest_categorical("policy", ["MlpPolicy", "CnnPolicy"])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(5e4), int(1e5), int(1e6)])
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 1000, 5000, 10000, 15000, 20000])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 5000, 10000, 20000])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 128, 256, 1000])
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)

    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

    net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]

    hyperparams = {
        "env": make_env(),
        "policy": policy,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "target_update_interval": target_update_interval,
        "learning_starts": learning_starts,
        "policy_kwargs": dict(net_arch=net_arch),
        "device": "cuda"
    }

    # if trial.using_her_replay_buffer:
    #     hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams

def eval_model_on_compilergym_benchmark(model):
    # Define the test environment.
    test_env = make_test_env()

    # Run the evaluation helper method from SB3.
    mean_reward_per_episode, std_dev_of_reward_per_episode = evaluate_policy(model, make_test_env())

    return mean_reward_per_episode

def objective(trial):
    """
    Calls helper functions to choose hyperparameters with Optuna and trains a model iteratively.
    Set prune=True to 
    """
    ts = calendar.timegm(time.gmtime()) # Timestamp for uniqueness of the model when saving.

    model = DQN(**(sample_dqn_params(trial))) # Instantiate the model with sampled hyperparameters.


    # Iteratively train the model on the training environment.
    total_steps = 5000
    step_size = trial.suggest_int("step_size", 200, 2000, step=100)
    for steps in range(1000, total_steps, step_size): # Steps goes up by step_size until it reaches total_steps.

      model.learn(total_timesteps=step_size) # Train

      score = eval_model_on_compilergym_benchmark(model) # Evaluate

      # Pruning. Example here: https://github.com/optuna/optuna-examples/blob/6a6b20ad634627eebb3e7e104f73b70b45c6e624/simple_pruning.py
      prune = True
      if prune == True:
        trial.report(score, steps)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
          
      
      # model.save(f"dqn_llvm_model_{ ts }")

      print("Model saved. . .")

    return score

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    NOTE: Comes from: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py
    Sampler for SAC hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    learning_starts = trial.suggest_categorical("learning_starts", [0, 1000, 10000, 20000])
    # train_freq = trial.suggest_categorical('train_freq', [1, 10, 100, 300])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    gradient_steps = train_freq
    # ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    # You can comment that out when not using gSDE
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "large": [256, 256, 256],
        # "verybig": [512, 512, 512],
    }[net_arch]

    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

    hyperparams = {
        'env': training_env,
        'policy': 'MlpPolicy',
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
        "device": "cuda"
    }

    # if trial.using_her_replay_buffer:
    #     hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def multi_model_objective(trial):
    """
    UNDER DEVELOMENT!
    Objective function for running trials with multiple models at once.
    Inspired by this example: https://github.com/optuna/optuna-examples/blob/main/kubernetes/simple/sklearn_distributed.py
    """
    ts = calendar.timegm(time.gmtime()) # Timestamp for uniqueness of the model when saving.

    model_type = trial.suggest_categorical("model_type", ["DQN", "SAC"]) # Use Optuna helper function to choose which model should be used in this trial.

    if model_type == 'DQN':
      model = DQN(**(sample_dqn_params(trial))) # Instantiate the model with sampled hyperparameters.
    elif model_type == 'SAC':
      model = SAC(**(sample_sac_params(trial)))

    # Iteratively train the model on the training environment.
    total_steps = 3000
    step_size = [1000, 10000, 50000, 100000]
    #for steps in range(1000, total_steps, step_size): # Steps goes up by step_size until it reaches total_steps.
    for steps in step_size:
      model.learn(total_timesteps=steps) # Train

      score = eval_model_on_compilergym_benchmark(model) # Evaluate

      # Pruning. Example here: https://github.com/optuna/optuna-examples/blob/6a6b20ad634627eebb3e7e104f73b70b45c6e624/simple_pruning.py
      prune = True
      if prune == True:
        trial.report(score, steps)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
          
      
      model.save(f"dqn_llvm_model_{ ts }")

      print("Model saved. . .")

    return score

if __name__ == "__main__":
    database_url = "postgresql://yzvxgwluxjnkap:8cd45bfa27d5df1577be2e2b20a35c90cf154d272c8b5975bb28266852c7dbd9@ec2-3-231-112-124.compute-1.amazonaws.com:5432/d1mqml0sjdqj22"

    ts = calendar.timegm(time.gmtime()) # Timestamp for uniqueness of the study name.

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(study_name=f"dqn_test_all", direction="maximize", storage=database_url, load_if_exists=True)
    study.optimize(objective, n_trials=25, show_progress_bar=True, n_jobs = -1)
