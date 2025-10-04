from panda_gym.pybullet import PyBullet
from lite6 import MyRobot
import numpy as np
import time

# Create simulation
sim = PyBullet(render_mode="human")
robot = MyRobot(sim)

# Run simulation and move joint
for i in range(1000):
    angle = np.array([np.sin(i * 0.01)])  # Oscillating joint movement
    robot.set_action(angle)
    sim.step()
    time.sleep(0.01)  # Slow down to make movement visible