import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../panda-gym')) # the directory of 'panda-gym'
import time
import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
import envs.uf_tasks.lite6_tasks  # Force registration to run

def start_sim():
    # target test environment
    env_name = "Lite6Reach-v1"

    env = gym.make(env_name, render_mode="human")
    env = DummyVecEnv([lambda : env])

    # Load model For Lite6:
    model = DDPG.load("./model/multi_point/ddpg-Lite6Reach-v3.pkl", device="cuda:0", env=env) # DDPG + HER ---> Multi Point Model
    # model = DDPG.load("./model/single_point/ddpg-Lite6Reach-v2.pkl", device="cuda:0", env=env) # DDPG + HER ---> Single Point Model

    # test for 50 episodes:
    # episodes = 50
    episodes = 1
    sum_score = 0.0

    traj_list = []
    ee_pose_list = []

    for episode in range(1, episodes+1): 
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
            joint_angles = [env.envs[0].unwrapped.robot.get_joint_angle(i) for i in range(6)]
            print("***************************")
            print(f"Step {steps}:")
            print(f"  Joint Angles (rad): {joint_angles}")
            
            # Get TCP (end-effector) pose: position and orientation
            tcp_position = env.envs[0].unwrapped.robot.get_link_position(6)
            tcp_position[0] += 0.5

            # Print everything
            print(f"  TCP Position (x, y, z): {tcp_position}")
            
            traj_list.append(joint_angles)
            ee_pose_list.append(tcp_position)
        
            time.sleep(0.2)
        
        print("***************************")
        print("Episode : {}, Score : {}".format(episode, score))
        sum_score = sum_score + score

    print("***************************\nAverage score: {}\n***************************".format(sum_score/episodes))

    env.close()
    
    return traj_list, ee_pose_list