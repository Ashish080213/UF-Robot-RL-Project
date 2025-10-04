from xarm.wrapper import XArmAPI
import time

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

# === Send joint command ===
speed=2
arm.set_servo_angle(angle=[0, 9.9, 31.8, 0, 21.9, 0], speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))
arm.set_servo_angle(angle=[0, -45, 31.8, 0, 21.9, 0], speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))
arm.set_servo_angle(angle=[0, -45, 45, 0, 21.9, 0], speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))
arm.set_servo_angle(angle=[0, -45, 45, 0, 90, 0], speed=speed, wait=True)
print(arm.get_servo_angle(), arm.get_servo_angle(is_radian=True))

# === Done ===
print("Joint angles set successfully!")
arm.disconnect()
