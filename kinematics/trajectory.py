from xarm.wrapper import XArmAPI
import time
from rl.test_lite6_reach import start_sim
import numpy as np
from detection.call import Call

Call()
joint_traj, ee_pose = start_sim()

# === Connect to xArm ===
arm = XArmAPI('192.168.1.171')  # xArm IP
arm.connect()

if not arm.connected:
    print("Robot arm is not connected!")

err = arm.error_code  # <-- fixed here
print(f"Error code: {err}")

if err != 0:
    arm.clean_error()

# === Prepare the arm ===
arm.motion_enable(True)
arm.set_mode(0)   # Position control mode
arm.set_state(0)  # Set to ready state

# === Define target joint angles ===
# Angles are in degrees: [J1, J2, J3, J4, J5, J6]
# target_joints = [0, -45, 45, 0, 90, 0]

joint_traj = np.rad2deg(joint_traj)
# print(joint_traj)

# === Send joint command ===
speed=5 # speed in UF studio set to 50%
for i in range(len(joint_traj)-1):
    traj = np.delete(joint_traj[i], 0)
    traj = np.append(traj, 0)
    # traj = traj + [0, -45, 45, 0, 90, 0] # previous URDF initial pose offset
    
    # updated URDF initial pose offset
    traj[0] = traj[0] + 90
    traj[1] = traj[1] + 26.4
    traj[2] = traj[2] + 116.9
    traj[3] = traj[3] + 1
    traj[4] = traj[4] + 83.5
    traj[5] = 180
    print("***************************")
    print("Traj: ", traj)
    arm.set_servo_angle(angle=traj, speed=speed, wait=True)
    # print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

print("***************************")
print("Sim EndEffector_Pose in mm: ", 1000*(ee_pose[len(ee_pose)-2]))
print("Robot EndEffector_Pose in mm", (arm.get_position())[1][:3])
print("***************************")

# === Done ===
print("Joint angles set successfully!")
arm.disconnect()
