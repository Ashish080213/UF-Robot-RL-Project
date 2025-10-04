import os
import sys
import time

from xarm.wrapper import XArmAPI

# === Connect to xArm ===
arm = XArmAPI('192.168.1.171')  # xArm IP
arm.connect()

if not arm.connected:
    print("Robot arm is not connected!")

err = arm.error_code  # <-- fixed here
print(f"Error code: {err}")

if err != 0:
    arm.clean_error()
    
arm.motion_enable(enable=True)

arm.set_mode(1)
arm.set_state(0)
time.sleep(0.1)

while arm.connected and arm.state != 4:
    for i in range(100):
        angles = [i, 9.9, 31.8, 0, 21.9, 0]
        ret = arm.set_servo_angle_j(angles)
        print('set_servo_angle_j, ret={}'.format(ret))
        time.sleep(0.01)
    for i in range(100):
        angles = [100-i, 9.9, 31.8, 0, 21.9, 0]
        ret = arm.set_servo_angle_j(angles)
        print('set_servo_angle_j, ret={}'.format(ret))
        time.sleep(0.01)

arm.disconnect()