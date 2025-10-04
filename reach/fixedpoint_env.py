from typing import Any, Dict

import numpy as np

from panda_gym.envs.core import Task
from panda_gym.utils import distance

import os
import yaml

class Reach(Task):
    def __init__(
        self,
        sim,
        get_ee_position,
        reward_type="sparse",
        distance_threshold=0.05,
        goal_range=0.3,
    ) -> None:
        super().__init__(sim)
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.get_ee_position = get_ee_position
        self.goal_range_low = np.array([-goal_range / 2, -goal_range / 2, 0])
        self.goal_range_high = np.array([goal_range / 2, goal_range / 2, goal_range])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_sphere(
            body_name="target",
            radius=0.02,
            mass=0.0,
            ghost=True,
            position=np.zeros(3),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        return np.array([])  # no tasak-specific observation

    def get_achieved_goal(self) -> np.ndarray:
        ee_position = np.array(self.get_ee_position())
        return ee_position

    def reset(self) -> None:
        self.goal = self._sample_goal()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Randomize goal."""
        # goal = self.np_random.uniform(self.goal_range_low, self.goal_range_high) # Original Code - Far from robot
        
        # Get current end-effector position
        ee_pos = np.array(self.get_ee_position())

        # Fixed goal close to EE — 3cm in each direction (X, Y), slightly above in Z
        
        # V-1 Single Target Model Trained Point
        # noise = np.array([0.2, -0.2, -0.4])

        # V-2 Single Target Model Trained Point
        # noise = np.array([0.05, 0.05, -0.4])
        
        # V-2 Single Target Model Testing Point
        # noise = np.array([0.1, 0.03, -0.4])
        
        # V-2 Single Target Model Hardware Testing Point
        yaml_file_path = 'detection/yaml/coordinates.yaml'

        # Read the existing x, y, z coordinates from the YAML file
        if os.path.exists(yaml_file_path):
            with open(yaml_file_path, 'r') as file:
                pose_dict = yaml.safe_load(file)
        else:
            raise FileNotFoundError(f"YAML file {yaml_file_path} not found")

        target = np.array([pose_dict['x'], pose_dict['y'], pose_dict['z']])
        noise = np.array([pose_dict['y'], -pose_dict['x'], pose_dict['z']]) # this change because hardware is -90 degrees off in joint1
        
        goal = (noise/1000) + np.array([-0.5, 0, 0]) # converting to m, and adding x=-0.5 offset which is in sim.
        print("Goal Point: ", target)
        # goal = ee_pos + noise # commented to run hardware logic
        
        # print("EE POSE", ee_pos)
        
        return goal

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -np.array(d > self.distance_threshold, dtype=np.float32)
        else:
            return -d.astype(np.float32)
