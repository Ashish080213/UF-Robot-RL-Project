from panda_gym.pybullet import PyBullet
from src.custom_lite6_v1 import MyRobot
import numpy as np
import time

# Create simulation
sim = PyBullet(render_mode="human")

# Set camera zoom and angle
sim.physics_client.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

robot = MyRobot(sim)

angle = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Run simulation and move joint
for i in range(1000):
      # 6 joint targets
    robot.set_action(angle)
    sim.step()
    time.sleep(1)
