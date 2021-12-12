#!/usr/bin/python3
import os
from compiler_gym.datasets import benchmark
from compiler_gym.wrappers.datasets import RandomOrderBenchmarks

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

import torch

from typing import Any, Dict

import calendar
import time
import logging
import sys

import random

from datetime import datetime
#with open(f"../func/not_cbench_cg.txt", "r") as benchmarks_files:
#  benchmarks = benchmarks_files.readlines()


def make_env(env_config=None) -> compiler_gym.envs.CompilerEnv:
    """Make the reinforcement learning environment for this experiment.
    
      From FB example.
    """
    #global benchmarks
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
    
    env = TimeLimit(env, max_episode_steps=300)
    del env.datasets["generator://csmith-v0"]
    del env.datasets["generator://llvm-stress-v0"]
    del env.datasets["cbench-v1"]

    # Each dataset has a `benchmarks()` method that returns an iterator over the
    # benchmarks within the dataset. Here we will use iterator sliceing to grab a 
    # handful of benchmarks for training and validation.

    #test_set = random.choices(benchmarks,k=5000)
    
    dataset = env.datasets.benchmarks()
    
    train_benchmarks = list(islice(dataset, 5000))

    env = RandomOrderBenchmarks(env, train_benchmarks)
    #env = RandomOrderBenchmarks(env, ["generator://csmith-v0"])
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

    env = TimeLimit(env, max_episode_steps=300)

    dataset = env.datasets["cbench-v1"] # Small dataset

    train_benchmarks = list(dataset) # N_bechmarks total benchmarks the dataset

    env = CycleOverBenchmarks(env, train_benchmarks)

    return env


def sample_dqn_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    NOTE: Comes from: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/utils/hyperparams_opt.py
    Sampler for DQN hyperparams.
    :param trial:
    :retur/
    """
    policy = 'MlpPolicy' # trial.suggest_categorical("policy", ["MlpPolicy", "CnnPolicy"])
    gamma = trial.suggest_uniform("gamma", 0.97, 0.99999)
    learning_rate = trial.suggest_uniform("learning_rate", 0.0005, 0.01)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    buffer_size = trial.suggest_categorical("buffer_size", [1024, 2048, 4096, 8192, 16384])
    exploration_final_eps = trial.suggest_uniform("exploration_final_eps", 0, 0.2)
    exploration_fraction = trial.suggest_uniform("exploration_fraction", 0, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [1, 100, 500, 1000, 1500, 5000, 10000])
    learning_starts = trial.suggest_categorical("learning_starts", [5000, 10000, 20000])

    train_freq = trial.suggest_categorical("train_freq", [4, 8, 16, 128, 256])
    subsample_steps = trial.suggest_categorical("subsample_steps", [1, 2, 4, 8])
    gradient_steps = max(train_freq // subsample_steps, 1)

    net_arch = trial.suggest_categorical("net_arch", ["medium", "large", "larger", "deep"])

    net_arch = {"medium": [256, 256], "large": [128, 128, 128], "larger":[128, 128, 128, 128], "deep": [128, 128, 128, 128, 128]}[net_arch]

    activation_fn = trial.suggest_categorical("activation_fn", ["relu", "leaky_relu", "tanh"])
    activation_fn = {"relu": torch.nn.ReLU, "leaky_relu":torch.nn.LeakyReLU, "tanh": torch.nn.Tanh}[activation_fn]
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
        "policy_kwargs": dict(net_arch=net_arch, activation_fn=activation_fn)#,
        "device": f"cuda}"
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
    
    ts = datetime.now().strftime("%Y_%m_%d_%H%M") # Timestamp for uniqueness of the model when saving.
    print(f"{ts}: Made new trial")
    model = DQN(**(sample_dqn_params(trial))) # Instantiate the model with sampled hyperparameters.

    # Iteratively train the model on the training environment.
    total_epochs = 100000
    
    evaluation_spread = 5000

    for epochs in range(1000, total_epochs, evaluation_spread): # Steps goes up by step_size until it reaches total_steps.

      model.learn(total_timesteps=evaluation_spread) # Train

      score = eval_model_on_compilergym_benchmark(model) # Evaluate

      # Pruning. Example here: https://github.com/optuna/optuna-examples/blob/6a6b20ad634627eebb3e7e104f73b70b45c6e624/simple_pruning.py
      prune = True
      if prune == True:
        trial.report(score, epochs)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

      ts = datetime.now().strftime("%Y_%m_%d_%H%M")
      print(f"{ts} Model on: {epochs}/{total_epochs}")
      print(f"Evaluation Score: {score}")

    return score

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y_%m_%d_%H%M")
    print(f"Program started at: {ts}")
    database_url = "postgresql://uuiqr5ei8ll3q:pd0a212fdb3d2379d6b849a598a75c6825267df881bb20ac19d97f2afd24a4aa3@ec2-3-218-203-60.compute-1.amazonaws.com:5433/d6h21bev0n8gtf"

    ts = calendar.timegm(time.gmtime()) # Timestamp for uniqueness of the study name.

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(study_name=f"dqn_test_deep", direction="maximize", storage=database_url, load_if_exists=True)
    study.optimize(objective, n_trials=3, n_jobs = -1)
    ts = datetime.now().strftime("%Y_%m_%d_%H%M")
    print(f"Program ended at: {ts}")


