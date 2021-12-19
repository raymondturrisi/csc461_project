import os

import gym

import compiler_gym
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy # Evaluation method used by FB for leaderboard
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import TimeLimit

import numpy as np
import pandas as pd

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_device

import optuna
from torch import nn as nn # Needed for sample_a2c params.
from typing import Callable

from itertools import islice
from compiler_gym.wrappers import CycleOverBenchmarks, RandomOrderBenchmarks

from typing import Any, Dict

import calendar
import time

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
    env = TimeLimit(env, max_episode_steps=500)

    train_benchmarks = list(env.datasets["cbench-v1"])

    # Each dataset has a `benchmarks()` method that returns an iterator over the
    # benchmarks within the dataset. Here we will use iterator sliceing to grab a 
    # handful of benchmarks for training and validation.

    print(f"Using { len(train_benchmarks) } benchamrks\n")

    env = RandomOrderBenchmarks(env, train_benchmarks)

    env = DummyVecEnv([lambda: env for _ in range(4)])

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
    env = TimeLimit(env, max_episode_steps=500)

    dataset = list(env.datasets["cbench-v1"]) # Small dataset

    # Each dataset has a `benchmarks()` method that returns an iterator over the
    # benchmarks within the dataset. Here we will use iterator sliceing to grab a 
    # handful of benchmarks for training and validation.

    train_benchmarks = list(dataset) # N_bechmarks total benchmarks the dataset

    env = CycleOverBenchmarks(env, train_benchmarks)

    return env


def eval_model_on_compilergym_benchmark(model):
    # Define the test environment.
    test_env = make_test_env()

    # Run the evaluation helper method from SB3.
    mean_reward_per_episode, std_dev_of_reward_per_episode = evaluate_policy(model, make_test_env())

    return mean_reward_per_episode

# From stable-baselines3 docs. Used in sample_a2c params
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.
    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    # Toggle PyTorch RMS Prop (different from TF one, cf doc)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
    ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # sde_net_arch = trial.suggest_categorical("sde_net_arch", [None, "tiny", "small"])
    # full_std = trial.suggest_categorical("full_std", [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
    }[net_arch]

    # sde_net_arch = {
    #     None: None,
    #     "tiny": [64],
    #     "small": [64, 64],
    # }[sde_net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "env": make_env(),
        "policy": 'MlpPolicy',
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # full_std=full_std,
            activation_fn=activation_fn,
            # sde_net_arch=sde_net_arch,
            ortho_init=ortho_init,
        ),
    }


def objective(trial):
    """
    Calls helper functions to choose hyperparameters with Optuna and trains a model iteratively.
    Set prune=True to 
    """

    model = A2C(**(sample_a2c_params(trial))) # Instantiate the model with sampled hyperparameters.

    # Iteratively train the model on the training environment.
    total_steps = 50000
    step_size = 5000
    for steps in range(step_size, total_steps, step_size): # Steps goes up by step_size until it reaches total_steps.

      model.learn(total_timesteps=step_size) # Train

      score = eval_model_on_compilergym_benchmark(model) # Evaluate

      # Pruning. Example here: https://github.com/optuna/optuna-examples/blob/6a6b20ad634627eebb3e7e104f73b70b45c6e624/simple_pruning.py
      prune = True
      if prune == True:
        trial.report(score, steps)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()
          
    return score

if __name__ == '__main__':
    database_url = "postgresql://uuiqr5ei8ll3q:pd0a212fdb3d2379d6b849a598a75c6825267df881bb20ac19d97f2afd24a4aa3@ec2-3-218-203-60.compute-1.amazonaws.com:5432/d6h21bev0n8gtf"

    device_type = get_device(device="cuda") # Stable-baselines3 select device to run on.
    print(f"Using { device_type } device!")

    study = optuna.create_study(study_name=f"a2c_tests_bridges", direction="maximize", storage=database_url, load_if_exists=True)
    study.optimize(objective, n_trials=50, n_jobs=-1)
