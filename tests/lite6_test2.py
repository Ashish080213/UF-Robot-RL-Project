import envs
import gymnasium as gym

env = gym.make('Lite6Reach-v1', render_mode='human')
obs = env.reset()

input("Press Enter to close the window...")

env.close()
