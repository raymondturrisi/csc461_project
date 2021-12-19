import os

import compiler_gym
from compiler_gym.leaderboard.llvm_instcount import eval_llvm_instcount_policy # Evaluation method used by FB for leaderboard
from compiler_gym.envs import LlvmEnv
from compiler_gym.wrappers import TimeLimit, CycleOverBenchmarks, RandomOrderBenchmarks
import optuna
import time
from datetime import datetime
import torch as hot_like_fire
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_device
from itertools import islice
import names

#Define a parameter selection function with optuna trials

def get_trial_params(trial: optuna.Trial):
    # configuring the environment parameters with big tuna
    env_config = {
        "id":"llvm-ic-v0",
        "observation_space":trial.suggest_categorical("observation_space",["InstCount", "Autophase", "InstCountNorm"]),
        "reward_space":"IrInstructionCountOz"
        }

    env = compiler_gym.make(**env_config)

    datasets = env.datasets.benchmarks()
    training_data = list(islice(datasets, 5000))
    
    env = CycleOverBenchmarks(env,training_data)

    # Currently, parameters only for an MLP, I don't think another is appropriate atm

    #number of layers
    net_arch = trial.suggest_categorical("net_arch", ["128..2","128..4","128..5","128..6","128..8",
                                                        "256..2", "256..4","256..5","256..6","256..8"])
    net_arch = {
        "128..2":[128, 128],
        "128..4":[128, 128, 128, 128], 
        "128..5": [128, 128, 128, 128, 128],
        "128..6":[128, 128, 128, 128, 128, 128], 
        "128..8": [128, 128, 128, 128, 128, 128, 128, 128, 128],
        "256..2":[256, 256], 
        "256..4":[256, 256, 256, 256], 
        "256..5": [256, 256, 256, 256, 256],
        "256..6":[256, 256, 256, 256, 256, 256], 
        "256..8": [256, 256, 256, 256, 256, 256, 256, 256, 256]}[net_arch]

    #activation function
    activation_fn = trial.suggest_categorical("activation_fn", ["relu", "leaky_relu", "tanh"])
    activation_fn = {"relu": hot_like_fire.nn.ReLU, "leaky_relu":hot_like_fire.nn.LeakyReLU, "tanh": hot_like_fire.nn.Tanh}[activation_fn]
    
    #optimizer - read below
    optimizer_class = trial.suggest_categorical("optimizer", ["Adam", "SGD", "RMSProp"])


    optimizer_class = {"Adam":hot_like_fire.optim.Adam, 
                        "SGD": hot_like_fire.optim.SGD, 
                        "RMSProp": hot_like_fire.optim.RMSprop}[optimizer_class]
    
    #for mlp
    policy_kwargs = {
        "net_arch":net_arch,
        "activation_fn":activation_fn,
        "ortho_init":trial.suggest_categorical("ortho_init", [True, False]),
        "optimizer_class":optimizer_class,
        "optimizer_kwargs":None
    }
    # Read: https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6


    # Params for PPO Model
    model_config = {
        "env":env,
        "learning_rate":trial.suggest_uniform("learning_rate", 0.0005, 0.03),
        "n_steps":trial.suggest_int("n_steps", 100, 500),
        "batch_size":trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "n_epochs":trial.suggest_int("n_epochs", 100, 100000),
        "gamma":trial.suggest_uniform("gamma", 0.97, 0.999999),
        "gae_lambda":trial.suggest_uniform("gae_lambda", 0,2.6),
        "clip_range":trial.suggest_uniform("clip_range", 0,1),
        "clip_range_vf":trial.suggest_uniform("clip_range_vf", 0,1),
        "ent_coef":trial.suggest_uniform("ent_coef", 0, 4.37),
        "max_grad_norm":trial.suggest_uniform("max_grad_norm", 0,5.1),
        "sde_sample_freq":trial.suggest_int("sde_sample_freq", -1, 20),
        "device":"auto",
        "policy":"MlpPolicy",
        "policy_kwargs":policy_kwargs
    }
    #return env_config, model_config, policy_kwargs
    return model_config, env_config


def make_test_env(env_config, model_config) -> compiler_gym.envs.CompilerEnv:

    env = compiler_gym.make(**env_config)

    env = TimeLimit(env, max_episode_steps=model_config["n_steps"])

    dataset = env.datasets["cbench-v1"] # Small dataset

    train_benchmarks = list(dataset) # N_bechmarks total benchmarks the dataset

    env = CycleOverBenchmarks(env, train_benchmarks)
    env = DummyVecEnv([lambda: env])
    return env 

def eval_model_on_compilergym_benchmark(model, env_config, model_config):
    # Define the test environment.
    #test_env = make_test_env(env_config, model_config)

    # Run the evaluation helper method from SB3.
    mean_reward_per_episode, _ = evaluate_policy(model, make_test_env(env_config, model_config))

    return mean_reward_per_episode

def objective_fn(trial):
    ts = datetime.now().strftime("%Y_%m_%d_%H%M")
    trial_name = names.get_full_name()

    print(f"{ts}: Made new trial. Hello, {trial_name}.")
    
    model_config, env_config = get_trial_params(trial)
    
    print(f"Model Params:\n{model_config}")
    print(f"Env. Params:\n{env_config}")
    model = PPO(**model_config)

    total_epochs = 100000
    training_stride = 2500

    for epochs in range(training_stride,total_epochs+1,training_stride):
        model.learn(total_timesteps=training_stride)

        score = eval_model_on_compilergym_benchmark(model, env_config=env_config, model_config=model_config)

        trial.report(score, epochs)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.TrialPruned()

        ts = datetime.now().strftime("%Y_%m_%d_%H%M")
        print(f"{ts}: {trial_name} is on: {epochs}/{total_epochs}")
        print(f"{trial_name} evaluation Score: {score}")
        return score 



# Define study
def main():
    database_url = "postgresql://uuiqr5ei8ll3q:pd0a212fdb3d2379d6b849a598a75c6825267df881bb20ac19d97f2afd24a4aa3@ec2-3-218-203-60.compute-1.amazonaws.com:5432/d6h21bev0n8gtf"
    study = optuna.create_study(study_name=f"ppo_1", direction="maximize", storage=database_url, load_if_exists=True)
    study.optimize(objective_fn, n_trials=1, n_jobs=-1)


# Launch study
main()