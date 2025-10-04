# lite6_test.py

import envs  # 👈 this triggers registration

import gymnasium as gym
env = gym.make("Lite6Reach-v1")
obs, _ = env.reset()
print(obs)
