import os
import yaml
import numpy as np  # Assuming target_pose might be a NumPy array
from detection.detect import start_detection

def Call():
    # Get the target pose from the detection function
    target_pose = start_detection()

    # Print the target pose to check its structure
    print(target_pose)

    # If target_pose is a NumPy array, convert it to a list
    if isinstance(target_pose, np.ndarray):
        target_pose = target_pose.tolist()

    # Ensure target_pose has exactly 3 elements
    if len(target_pose) == 3:
        # Map the values to x, y, z
        pose_dict = {
            'x': target_pose[0],
            'y': target_pose[1],
            'z': target_pose[2]
        }
    else:
        raise ValueError("Target pose does not contain exactly 3 coordinates")

    # Ensure the directory exists before saving the file
    output_dir = 'detection/yaml'
    os.makedirs(output_dir, exist_ok=True)

    # Save the coordinates into a YAML file
    yaml_file_path = os.path.join(output_dir, 'coordinates.yaml')
    with open(yaml_file_path, 'w') as file:
        yaml.dump(pose_dict, file, default_flow_style=False)

    print(f"Coordinates saved to '{yaml_file_path}'.")
