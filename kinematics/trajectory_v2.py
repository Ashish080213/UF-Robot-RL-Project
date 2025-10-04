from xarm.wrapper import XArmAPI
import time
from rl.test_lite6_reach import start_sim
import numpy as np

joint_traj = start_sim()

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
print(joint_traj)

# === Send joint command ===
speed=5

traj = np.delete(joint_traj[0], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[1], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[2], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[3], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[4], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[5], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[6], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[7], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[8], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[9], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[10], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[11], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

traj = np.delete(joint_traj[12], 0)
traj = np.append(traj, 0)
traj = traj + [0, -45, 45, 0, 90, 0]
print(traj)
arm.set_servo_angle(angle=traj, speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=False))

# === Done ===
print("Joint angles set successfully!")
arm.disconnect()
