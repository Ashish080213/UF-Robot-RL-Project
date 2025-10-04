import numpy as np
from panda_gym.pybullet import PyBullet
from custom_lite6_v1 import MyRobot
import time
import pybullet as p

sim = PyBullet(render_mode="human")
robot = MyRobot(sim)

current_angles = np.zeros(6, dtype=np.float32)
selected_joint = None
increment = 0.01  # larger increment

# Set camera closer to the robot
sim.physics_client.resetDebugVisualizerCamera(
    cameraDistance=1.0,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0, 0, 0]
)

print("Press keys 1 to 5 to select a joint.")
print("Use UP/DOWN arrows to move the selected joint.")

while True:
    keys = p.getKeyboardEvents()

    # Detect joint selection keys (1-5)
    for k in keys:
        if keys[k] & p.KEY_WAS_TRIGGERED:
            if k in [ord(str(i)) for i in range(1, 6)]:
                selected_joint = k - ord('0')
                print(f"Selected joint: {selected_joint}")

    # Move selected joint with arrow keys
    if selected_joint is not None:
        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            current_angles[selected_joint] += increment
            print(f"Joint {selected_joint} angle increased to {current_angles[selected_joint]:.2f}")
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            current_angles[selected_joint] -= increment
            print(f"Joint {selected_joint} angle decreased to {current_angles[selected_joint]:.2f}")

    # Clamp angles - adjust limits if needed
    current_angles = np.clip(current_angles, -2*np.pi, 2*np.pi)

    # Apply joint angles to robot
    robot.set_action(current_angles)

    sim.step()
    time.sleep(0.01)
