import compiler_gym
import gym
import os

from compiler_gym.wrappers import CycleOverBenchmarks
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import DQN
from compiler_gym.wrappers import TimeLimit



def make_test_env(env_config=None) -> compiler_gym.envs.CompilerEnv:
    """
    Make the testing environment for evaluating approximate performance on the test set.
    
    Defining this function because the eval method by Facebook did not function as desired
    in tests.
    """
    env = compiler_gym.make(
        "llvm-ic-v0",
        observation_space="InstCountNorm",
        reward_space="IrInstructionCountOz",
    )

    # Finally, we impose a time limit on the environment so that every episode
    # for 5 steps or fewer. This is because the environment's task is continuous
    # and no action is guaranteed to result in a terminal state. Adding a time
    # limit means we don't have to worry about learning when an agent should 
    # stop, though again this limits the potential improvements that the agent
    # can achieve compared to using an unbounded maximum episode length.
    env = TimeLimit(env, max_episode_steps=2000)

    dataset = env.datasets["cbench-v1"] # Small dataset

    # Each dataset has a `benchmarks()` method that returns an iterator over the
    # benchmarks within the dataset. Here we will use iterator sliceing to grab a 
    # handful of benchmarks for training and validation.

    train_benchmarks = list(dataset) # N_bechmarks total benchmarks the dataset

    env = CycleOverBenchmarks(env, train_benchmarks)

    return env

def eval_model_on_compilergym_benchmark(model):
    # Run the evaluation helper method from SB3.
    mean_reward_per_episode, std_dev_of_reward_per_episode = evaluate_policy(model, model.get_env())

    return mean_reward_per_episode


if __name__ == "__main__":
    for file in os.listdir("models/"):
        env = make_test_env()
        model = DQN.load("models/" + file, env=env)
        mean_reward = eval_model_on_compilergym_benchmark(model)
        print(file + " has a mean reward of " + str(mean_reward))