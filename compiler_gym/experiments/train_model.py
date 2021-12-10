import compiler_gym
import gym

from stable_baselines3 import DQN

def train_env():
    env = compiler_gym.make(
        "llvm-v0",
        reward_space="IrInstructionCountOz",
        observation_space="InstCountNorm"
    )
    return env

def test_env():
    env = compiler_gym.make(
        "llvm-v0",
        reward_space="IrInstructionCountOz",
        observation_space="InstCountNorm"
    )
    dataset = env.datasets["cbench-v1"]
    return env

def create_dqn_model(**kwargs):
    model = DQN(kwargs)


def train(env, model):
    
    del env.datasets["cbench-v1"]
    del env.datasets["generator://csmith-v0"]
    del env.datasets["generator://llvm-stress-v0"]
    train_benchmarks = env.datasets
    episodes = 50000 # The number of episodes used to learn
    episode_length = 12 # The maximum number of transformations
    patience = 6 # The maximum transformations with zero change tolerated before a new episode
    for i in range(1, episodes+1):
        obs = env.reset(benchmark = train_benchmarks.random_benchmark())
        done = False
        action_count = 0
        actions_since_last_change = 0
        episode_reward = 0

        while not done and action_count < episode_length and actions_since_last_change < patience: 
            action, _ = model.predict(obs)
            new_obs, reward, done, info = env.step(action)
            action_count += 1
            episode_reward += reward

            if reward == 0:
                actions_since_last_change += 1
            else:
                actions_since_last_change = 0
            
            model.learn(gradient_steps=1, batch_size = 32)

            obs = new_obs

            print("Step: " + str(i) + " Episode Total: " + "{:.4f}".format(episode_reward) + " Action: " + action)

            


if __name__ = "__main__":
    env = train_env()
    model = create_dqn_model(
        env = env,
        policy = 'MlpPolicy',
        gamma = 0.999,
        learning_rate = 0.0020846018394760695,
        batch_size = 32,
        buffer_size = 10000,
        train_freq = 4,
        gradient_Steps = 1,
        exploration_fraction = 0.49275615218804686,
        exploration_final_eps = 0.031248308796828307,
        target_update_interval = 1,
        learning_starts = 20000,
        policy_kwargs = dict(net_arch='tiny'),
        device = "cuda")
    train(env,model)