import gymnasium as gym
import numpy as np
import UAV_env_custom as UAV_env

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]
        self.rewards.append(reward)
        return True

# initlize the environment and verify it
env = UAV_env.UAVEnv()
check_env(env, warn=True)

# Wrap the environment using DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

# Define the fine noise for the agent.
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Create and train the DDPG model
model = DDPG('MultiInputPolicy', vec_env, action_noise=action_noise, verbose=1)
reward_logger = RewardLoggerCallback()
model.learn(total_timesteps=100, callback=reward_logger)

# Plot the rewards
plt.plot(reward_logger.rewards)
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.title('Rewards over Timesteps')
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
