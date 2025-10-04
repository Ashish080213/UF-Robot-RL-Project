import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../panda-gym')) # the directory of 'panda-gym'
import time
import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import envs.uf_tasks.lite6_tasks  # Force registration to run

# target test environment
env_name = "Lite6Reach-v1"

env = gym.make(env_name, render_mode="human")
env = DummyVecEnv([lambda : env])

# Load model For Lite6:
model = DDPG.load("./model/ddpg-Lite6Reach-v1.pkl", device="cuda:0", env=env) # DDPG + HER

# test for 50 episodes:
episodes = 50
sum_score = 0.0
for episode in range(1, episodes + 1): 
    state = env.reset()
    done = False
    score = 0
    steps = 0
    while not done:
        steps = steps +1
        action, _states = model.predict(state)
        # print("Actions: ", action)
        state, reward, done, info = env.step(action)
        # print("States: ",state)
        score += reward
        env.render()
        time.sleep(0.2)
    print("Episode : {}, Score : {}".format(episode, score))
    sum_score = sum_score + score

print("***************************\nAverage score: {}\n***************************".format(sum_score/episodes))

env.close()