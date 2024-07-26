import gymnasium as gym
import numpy as np
import UAV_env_custom as UAV_env

from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []
        self.delays = []
        self.episode_max_delay = []
        self.episode_rewards = []
        self.episode_reward_count = 0

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        delay  = 1/reward
        #delay = -np.log(reward)
        self.delays.append(delay)
        self.rewards.append(reward)
        self.episode_reward_count += reward

        done = self.locals['dones'][0]
        if done:
            #self.episode_rewards.append(min(self.rewards)) #trains to minimize reward maximizing the minimum delay
            self.episode_rewards.append(min(self.rewards)) #trains to maximize reward minimizing the maximum delay)
            self.episode_max_delay.append(max(self.delays))
            self.rewards = []
            self.delays = []
        return True

# initlize the environment and verify it
env = UAV_env.UAVEnv()
#check_env(env, warn=True)

# Set parameters for the replay buffer
buffer_size = 100000  # Size of the replay buffer
learning_starts = 1000  # Number of steps before learning starts

# Wrap the environment using DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

# Define the fine noise for the agent.
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Create and train the DDPG model
#model = DDPG('MultiInputPolicy', vec_env, verbose=1)
model = TD3('MultiInputPolicy', vec_env, buffer_size=buffer_size, learning_starts=learning_starts,action_noise=action_noise, verbose=1)
#model = DDPG('MultiInputPolicy', vec_env,action_noise=action_noise, verbose=1)
#model = PPO('MultiInputPolicy', vec_env, verbose=1)
reward_logger = RewardLoggerCallback()
model.learn(total_timesteps=3000, callback=reward_logger)

# Plot the rewards


plt.plot(reward_logger.episode_max_delay)
plt.xlabel('Episodes')
plt.ylabel('Max Delay')
plt.title('Max Delay over Episodes')
plt.show()
#print(reward_logger.episode_rewards)
plt.plot(reward_logger.episode_rewards)
plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Rewards over Episodes')
plt.show()

# model.save("ddpg_uav")
# del model

# # Load trained model
# model = DDPG.load("ddpg_uav")

# # Reset environment
# obs = vec_env.reset()

# #evaluate model
# episodes = 10
# for episode in range(episodes):
#     obs, _ = env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, done, truncated, info = env.step(action)
#         total_reward += reward
#     print(f"Episode {episode + 1}: Total Reward: {total_reward}")
