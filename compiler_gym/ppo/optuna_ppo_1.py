#!/usr/bin/python3

import compiler_gym
# Evaluation method used by FB for leaderboard
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import TimeLimit

from stable_baselines3 import DQN, SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy

import optuna

from itertools import islice
from compiler_gym.wrappers import CycleOverBenchmarks, RandomOrderBenchmarks

from typing import Any, Dict

import calendar
import time
import logging
import sys
from torch import nn

from datetime import datetime
print(" - - - - - - STARTING PPO - - - - - - ")

def make_env(env_config=None) -> compiler_gym.envs.CompilerEnv:
    print("Training Environment..")
    env = compiler_gym.make(
        "llvm-ic-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )

    #env = TimeLimit(env, max_episode_steps=1000)

    del env.datasets["cbench-v1"]
    del env.datasets["generator://csmith-v0"]
    del env.datasets["generator://llvm-stress-v0"]
    dataset = env.datasets.benchmarks()  # Every dataset besides cbench

    N_benchmarks = 5000
    num = int(sys.argv[1])

    train_benchmarks = list(islice(dataset, (num-1)*N_benchmarks, num*N_benchmarks))

    env = RandomOrderBenchmarks(env, train_benchmarks)
    return env


def make_test_env(env_config=None) -> compiler_gym.envs.CompilerEnv:
    print("Test Environment..")
    env = compiler_gym.make(
        "llvm-ic-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
    )

    # env = TimeLimit(env, max_episode_steps=1000)

    dataset = env.datasets["cbench-v1"]  # Small dataset

    train_benchmarks = list(dataset)

    env = CycleOverBenchmarks(env, train_benchmarks)

    return env


def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    print("Gathering Parameters..")

    policy = 'MlpPolicy'
    learning_rate = trial.suggest_uniform("learning_rate", 0.0005, 0.01)
    n_steps = trial.suggest_int("n_steps", 100, 1500)
    batch_size = trial.suggest_categorical("batch_size", [128, 256])
    n_epochs = trial.suggest_int("n_epochs", 7, 30)
    gamma = trial.suggest_uniform("gamma", 0.97, 0.99999)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.93, 1)
    clip_range = trial.suggest_uniform("clip_range", 0.15, 0.3)
    ent_coef = trial.suggest_uniform("ent_coef", 0, 0.2)
    vf_coef = trial.suggest_uniform("vf_coef", 0.3, 0.7)
    max_grad_norm = trial.suggest_uniform("max_grad_norm", 0.3, 0.7)
    use_sde = trial.suggest_categorical("use_sde", [True, False])
    net_arch = trial.suggest_categorical("net_arch", ["128..1","128..2","128..4","128..5","128..6","128..8",
                                                        "256..4","256..5","256..6","256..8"])

    net_arch = {"128..1":[128],
        "128..2":[128, 128],
        "128..4":[128, 128, 128, 128], 
        "128..5": [128, 128, 128, 128, 128],
        "128..6":[128, 128, 128, 128, 128, 128], 
        "128..8": [128, 128, 128, 128, 128, 128, 128, 128, 128],
        "256..4":[256, 256, 256, 256], 
        "256..5": [256, 256, 256, 256, 256],
        "256..6":[256, 256, 256, 256, 256, 256], 
        "256..8": [256, 256, 256, 256, 256, 256, 256, 256, 256]}[net_arch]

    activation_fn_idx = trial.suggest_int("activation_fn", 0,2)

    activation_fn = [nn.ReLU, nn.LeakyReLU, nn.Tanh][activation_fn_idx]
    
    hyperparams = {
        "env": make_env(),
        "policy": policy,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "use_sde": use_sde,
        "policy_kwargs": dict(net_arch=net_arch, activation_fn=activation_fn)
    }

    return hyperparams


def eval_model_on_compilergym_benchmark(model):
    # Define the test environment.
    test_env = make_test_env()

    # Run the evaluation helper method from SB3.
    mean_reward_per_episode, std_dev_of_reward_per_episode = evaluate_policy(
        model, make_test_env())

    return mean_reward_per_episode

study_counter = 0

internal_counter = 0

def objective(trial):
    """
    Calls helper functions to choose hyperparameters with Optuna and trains a model iteratively.
    Set prune=True to 
    """
    
    ts = datetime.now().strftime("%Y_%m_%d_%H%M") # Timestamp for uniqueness of the model when saving.
    print(f"{ts}: Made new trial")
    model = PPO(**(sample_ppo_params(trial))) # Instantiate the model with sampled hyperparameters.

    # Iteratively train the model on the training environment.
    total_epochs = 150000
    
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

    ts = calendar.timegm(time.gmtime())

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = optuna.create_study(study_name=f"dqn_test_deeper", direction="maximize", storage=database_url, load_if_exists=True)
    study.optimize(objective, n_trials=3, n_jobs = -1)
    ts = datetime.now().strftime("%Y_%m_%d_%H%M")
    print(f"Program ended at: {ts}")
