import time
import numpy as np
from panda_gym.pybullet import PyBullet
from custom_lite6_v1 import MyRobot

sim = PyBullet(render_mode="human")
robot = MyRobot(sim)

angles = np.zeros(6)

for i in range(1000):
    angles[1] = 0.5 * np.sin(i * 0.01)  # oscillate 2nd joint only
    robot.set_action(angles)
    sim.step()
    time.sleep(0.01)
