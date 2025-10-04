import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot


class MyRobot(PyBulletRobot):
    """My 6-joint lite6 robot"""

    def __init__(self, sim):
        action_dim = 6  # number of joints (link1 to link6)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="lite6",  # make sure this matches your URDF root link or robot name
            file_name="urdf/lite6.urdf",  # path to your URDF file
            base_position=np.zeros(3),
            action_space=action_space,
            joint_indices=np.arange(6),  # joint indices 0 to 5
            # joint_forces=np.array([32, 32, 18, 2.5, 2.5, 2.5])  # force array for all joints - v1
            joint_forces=np.array([50, 50, 32, 32, 32, 20])  # force array for all joints - v2

        )

    def set_action(self, action):
        # Set target angles for all joints
        self.control_joints(target_angles=action)

    def get_obs(self):
        # Get angles of all 6 joints
        return np.array([self.get_joint_angle(joint=i) for i in range(6)], dtype=np.float32)

    def reset(self):
        # Reset all joints to zero angles (neutral position)
        neutral_angles = np.zeros(6, dtype=np.float32)
        self.set_joint_angles(angles=neutral_angles)
        
    def print_joint_info(self):
        num_joints = self.sim.get_num_joints()
        print(f"Number of joints: {num_joints}")
        for i in range(num_joints):
            info = self.sim.get_joint_info(i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            print(f"Joint index: {i}, name: {joint_name}, type: {joint_type}")
