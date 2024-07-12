import gymnasium as gym
import numpy as np
import UAV_env_custom as UAV_env

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv

# Create and check the custom environment
env = UAV_env.UAVEnv()
check_env(env, warn=True)

# Wrap the environment using DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

# Define the action noise for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Create and train the DDPG model
model = DDPG('MultiInputPolicy', vec_env, action_noise=action_noise, verbose=1)
model.learn(total_timesteps=1000)
model.save("ddpg_uav")
del model

# Load the trained model
model = DDPG.load("ddpg_uav")

# Reset the environment
obs = vec_env.reset()

# Run the trained model in the environment
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    # Uncomment the following lines if you want to render the environment
    # env.render()
    if dones:
        obs = vec_env.reset()
