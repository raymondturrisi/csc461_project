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
from compiler_gym.wrappers import CycleOverBenchmarks

from typing import Any, Dict

import calendar
import time
import logging
import sys
from torch import nn


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

    train_benchmarks = list(islice(dataset, N_benchmarks))

    env = CycleOverBenchmarks(env, train_benchmarks)
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
    learning_rate = trial.suggest_uniform("learning_rate", 0.00001, 0.0005)
    n_steps = trial.suggest_int("n_steps", 100, 1500)
    batch_size = trial.suggest_int("batch_size", 50, 80)
    n_epochs = trial.suggest_int("n_epochs", 7, 30)
    gamma = trial.suggest_uniform("gamma", 0.99, 1)
    gae_lambda = trial.suggest_uniform("gae_lambda", 0.93, 1)
    clip_range = trial.suggest_uniform("clip_range", 0.15, 0.3)
    ent_coef = trial.suggest_uniform("ent_coef", 0, 0.2)
    vf_coef = trial.suggest_uniform("vf_coef", 0.3, 0.7)
    max_grad_norm = trial.suggest_uniform("max_grad_norm", 0.3, 0.7)
    use_sde = trial.suggest_categorical("use_sde", [True, False])

    # "net_arch_idx = trial.suggest_int("net_arch", 1, 5)"
    # "net_arch = {1: [64], 2: [64, 64], 3: [256, 256], 4: [128, 128, 128], 5: [256, 128, 64]}[net_arch_idx]"
    
    net_arch = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])

    net_arch = {"tiny": [64], "small": [64, 64], "medium": [256, 256]}[net_arch]

    #"activation_fn_idx = trial.suggest_int("activation_fn", 1,3)"

    #"activation_fn = [nn.ReLU, nn.LeakyReLU, nn.Tanh][activation_fn_idx]"
    
    #policy_kwargs = {"net_arch":net_arch, "activation_fn":activation_fn}
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
        "policy_kwargs": dict(net_arch=net_arch)
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
    global internal_counter
    # Instantiate the model with sampled hyperparameters.
    model = PPO(**(sample_ppo_params(trial)))

    print("- - Started training - -")
    model.learn(total_timesteps=1000*10**study_counter)  # Train
    score = eval_model_on_compilergym_benchmark(model)  # Evaluate
    trial.report(score, internal_counter)
    if trial.should_prune():
        raise optuna.TrialPruned()
    print(f"{time.asctime(time.localtime())}: IC {internal_counter}, SC {study_counter}, SCORE {score}")

    return score

study_trials = 3
if __name__ == "__main__":
    database_url = "postgresql://yzvxgwluxjnkap:8cd45bfa27d5df1577be2e2b20a35c90cf154d272c8b5975bb28266852c7dbd9@ec2-3-231-112-124.compute-1.amazonaws.com:5432/d1mqml0sjdqj22"
    optuna.logging.get_logger("optuna").addHandler(
        logging.StreamHandler(sys.stdout))
    study = optuna.create_study(
        study_name=f"ppo_test", direction="maximize", storage=database_url, load_if_exists=True)
    
    while study_counter <= study_trials:
        print(f"On study {study_counter}")
        study.optimize(objective, n_trials=1, show_progress_bar=False, n_jobs=-1)
        study_counter+=1
