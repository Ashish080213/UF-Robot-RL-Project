# Author: Jenson
# Modified: Ashish, Srini
# Modified Date: 6-Oct-2025

import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import time
import torch
import torchvision
from math import atan2, degrees, cos, sin
from torchvision.transforms import functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from ultralytics import YOLO


def start_detection():
    # === CONFIGURATION ===
    Z_OFFSET = 0.06  # meters
    MIN_Z_MM = 94  # safety lower bound for Z in mm
    xarm_ip = "192.168.1.171"
    clicked_pixel = None
    target_pos = None
    rot_target = None
    selected_class_id = 1  # 👈 ADDED: default selected class
    matching_objects = []
    current_selection_idx = 0

    # === MASK R-CNN CONFIG ===
    crop_size = 640
    move_step = 20
    zoom_step = 0.1
    min_zoom = 0.5
    max_zoom = 3.0
    view_params = {'zoom': 1.0, 'center_x': 640, 'center_y': 360}
    class_names = ["background", "medicine", "pointer",
                "glue", "marker"]

    # === LOAD CALIBRATION ===
    data = np.load("detection/dataDrawer/intrinsics.npz")
    K = data["K"]
    dist = data["dist"]
    he = np.load("detection/dataDrawer/handeye_result1.npz")
    R_cam2ee = he["R"]
    t_cam2ee = he["t"]

    # === ROBOT SETUP ===

    arm = XArmAPI(xarm_ip)
    arm.clean_error()
    arm.motion_enable(True)
    arm.set_mode(0)
    arm.set_state(0)
    # home = [83.4, 3.4, 409.7, 179.2, 0.0, 0.9] # actual home
    # home = [-6.0, 322.5, 448, 179.7, 6.7, -89.1] # for testing with 90 offset in joint1
    home = [-5.2, 85.5, 409.3, 179.0, 0.3, 90.9] # for testing multi point with 90 offset in joint1
    # home = [-156.2, 283.4, 447.4, 176.9, 5.9, -85.7] # for testing
    # rdet = [-118, -87, -152] # No where used in code
    arm.set_position(x=home[0], y=home[1], z=home[2],
                    roll=home[3], pitch=home[4], yaw=home[5],
                    speed=25, wait=True)

    # === REALSENSE SETUP ===
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # === INTRINSICS ===
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    depth_intrinsics = profile.get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()

    # === MASK R-CNN MODEL ===


    def get_custom_model(num_classes=5):
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes)
        return model


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_custom_model().to(device)
    model.load_state_dict(torch.load(
        "detection/model/best_maskrcnn_overlapped_40Epochs.pth", map_location=device))
    model.eval()

    # === UTILS ===

    # Transformation matrix define
    def build_T(R, t):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()
        return T


    def euler_to_rotation_matrix(rx, ry, rz):
        Rz = cv2.Rodrigues(np.array([0, 0, rz]))[0]
        Ry = cv2.Rodrigues(np.array([0, ry, 0]))[0]
        Rx = cv2.Rodrigues(np.array([rx, 0, 0]))[0]
        return Rz @ Ry @ Rx


    def rotation_matrix_to_euler(R):
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        if sy < 1e-6:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        else:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        return np.degrees([x, y, z])


    def mouse_callback(event, x, y, flags, param):
        global clicked_pixel
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_pixel = (x, y)
            print(f"[🖱️] Clicked Pixel: (u={x}, v={y})")

    # Rotation joint 6 conversion
    def convert_angle_to_rz(angle_deg):
        # Step 1: Convert original angle to rz format
        if 0 <= angle_deg <= 180:
            rz = -angle_deg
        else:
            rz = 360 - angle_deg

        # Step 2: Rotate the rz output by 180 degrees
        rz_rotated = ((rz + 180 + 180) % 360) - 180
        return rz_rotated

    # Safe call error handling
    def safe_position_call(arm, x, y, z, roll, pitch, yaw, speed=25, wait=True, max_retries=3, delay=0.5):
        for attempt in range(max_retries):
            code = arm.set_position(
                x=x, y=y, z=z,
                roll=roll, pitch=pitch, yaw=yaw,
                speed=speed, wait=wait
            )
            if isinstance(code, tuple):
                code = code[0]

            if code == 0:
                return True  # ✅ Success
            elif code == 9:
                print(
                    f"[⚠️] IK failed (code 9), attempt {attempt+1}/{max_retries}. Retrying...")
            else:
                print(f"[❌] Unhandled error code {code} from set_position.")
                break

            time.sleep(delay)

        print("[🛑] Failed to reach target pose after retries.")
        return False

    # For overlapped condition
    def compute_iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    # Convertion from image pixel to robot pose
    def compute_tcp_from_uv_depth(u, v, depth, K, R_cam2ee, t_cam2ee, current_pose):
        x = (u - K[0, 2]) * depth / K[0, 0]
        y = (v - K[1, 2]) * depth / K[1, 1]
        point_cam = np.array([x, y, depth, 1.0])

        pos_mm = np.array(current_pose[:3])
        angles_rad = np.radians(current_pose[3:])
        R_ee2base = euler_to_rotation_matrix(*angles_rad)

        T_ee2base = build_T(R_ee2base, pos_mm / 1000.0)
        T_cam2ee = build_T(R_cam2ee, t_cam2ee)
        point_base = T_ee2base @ T_cam2ee @ point_cam

        tcp_mm = point_base[:3] * 1000
        tcp_mm[2] += Z_OFFSET * 1000
        tcp_mm[2] = max(tcp_mm[2], MIN_Z_MM)

        return tcp_mm



    CLASS_COLORS = {
        1: (0, 255, 0),     # medicine - green
        2: (255, 0, 0),     # pointer - blue
        3: (0, 0, 255),     # glue - red
        4: (255, 255, 0),   # marker - cyan
        5: (255, 0, 255),   # extra class - magenta (if needed)
    }

    # === MAIN LOOP ===
    cv2.namedWindow("Click to Move")
    cv2.setMouseCallback("Click to Move", mouse_callback)

    print("🟢 Click a pixel to preview the robot target TCP pose")
    print("⏎ Press Enter to move, 'q' to quit")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            display = color_image.copy()

            cv2.putText(display, 'This is Home Menu',
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display, 'Press "o" for object picking and "q" for exit' ,
                        (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Click to Move", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
                        
    # -----------------------------------------------------OBJECT PICKING OPERATION------------------------------------------------------

            elif key == ord('o'):
                # === MASK R-CNN CONFIG ===
                crop_size = 640
                move_step = 20
                zoom_step = 0.1
                min_zoom = 0.5
                max_zoom = 3.0
                view_params = {'zoom': 1.0, 'center_x': 640, 'center_y': 360}
                class_names = ["background", "medicine", "pointer",
                            "glue", "marker"]
                # === ROBOT SETUP ===
                Z_OFFSET = 0.06  # meters
                MIN_Z_MM = 81  # safety lower bound for Z in mm
                arm.clean_error()
                arm.motion_enable(True)
                arm.set_mode(0)
                arm.set_state(0)
                # home = [-156.2, 283.4, 447.4, 176.9, 5.9, -85.7] # Not Working due to orientation issue -> Updated [Added offset to orientation in below code]
                # home = [-6.0, 322.5, 448, 179.7, 6.7, -89.1] # for testing in 90 degrees offset in Joint1
                home = [-5.2, 85.5, 409.3, 179.0, 0.3, 90.9] # for testing multi point with 90 offset in joint1
                # home = [-200, 246, 456, 180, 0, 180] # Jenson Coordinates
                drop = [320.7, 23, 13.1, 178.6, -1.6, 96.7]
                arm.set_position(x=home[0], y=home[1], z=home[2],
                                roll=home[3], pitch=home[4], yaw=home[5],
                                speed=25, wait=True)
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
                model = get_custom_model().to(device)
                model.load_state_dict(torch.load(
                    "detection/model/best_maskrcnn_overlapped_40Epochs.pth", map_location=device))
                model.eval()
                while True:
                    frames = pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    display = color_image.copy()

                    # === MASK R-CNN DETECTION (PRESERVED EXACTLY AS IS) ===
                    H, W = color_image.shape[:2]

                    # Define crop region
                    effective_size = int(crop_size / view_params['zoom'])
                    x1_crop = max(0, view_params['center_x'] - effective_size // 2) - 60
                    y1_crop = max(0, view_params['center_y'] - effective_size // 2) + 10
                    x2_crop = min(W, x1_crop + effective_size) - 175
                    y2_crop = min(H, y1_crop + effective_size) - 400

                    color_crop = color_image[y1_crop:y2_crop, x1_crop:x2_crop]
                    depth_crop = depth_image[y1_crop:y2_crop, x1_crop:x2_crop]

                    if color_crop.shape[:2] != (crop_size, crop_size):
                        color_crop_resized = cv2.resize(
                            color_crop, (crop_size, crop_size))
                        depth_crop_resized = cv2.resize(
                            depth_crop, (crop_size, crop_size), interpolation=cv2.INTER_NEAREST)
                    else:
                        color_crop_resized = color_crop
                        depth_crop_resized = depth_crop

                    alpha = 0.5
                    beta = -30
                    color_crop_resized = cv2.convertScaleAbs(
                        color_crop_resized, alpha=alpha, beta=beta)

                    input_tensor = F.to_tensor(
                        color_crop_resized).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = model(input_tensor)[0]

                    scale_x = (x2_crop - x1_crop) / crop_size
                    scale_y = (y2_crop - y1_crop) / crop_size

                    all_candidates = []

                    for i in range(len(outputs["masks"])):

                        score = outputs["scores"][i].item()
                        if score < 0.7:
                            continue

                        label_id = outputs["labels"][i].item()
                        label_name = class_names[label_id]

                        mask_resized = outputs["masks"][i, 0].cpu().numpy()
                        mask_resized = cv2.resize(
                            mask_resized, (x2_crop - x1_crop, y2_crop - y1_crop))
                        mask_full = np.zeros((H, W), dtype=np.uint8)
                        mask_full[y1_crop:y2_crop, x1_crop:x2_crop] = (
                            mask_resized > 0.5).astype(np.uint8)

                        contours, _ = cv2.findContours(
                            mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            continue

                        color = CLASS_COLORS.get(
                            label_id, (200, 200, 200))  # fallback gray

                        mask_rgb = np.zeros_like(display)
                        cv2.drawContours(mask_rgb, contours, -1, color, -1)
                        display = cv2.addWeighted(display, 1.0, mask_rgb, 0.6, 0)

                        x, y, w, h = cv2.boundingRect(contours[0])
                        cv2.rectangle(display, (x, y), (x + w, y + h),
                                    color, 2)  # ✅ Bounding Box Overlay
                        cx = int(x + w / 2)
                        cy = int(y + h / 2)

                        # Orientation
                        M = cv2.moments(mask_full)
                        if M["m00"] == 0:
                            continue
                        mu20, mu02, mu11 = M["mu20"], M["mu02"], M["mu11"]
                        theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

                        theta_orth = theta + np.pi / 2
                        dx_test = 20 * cos(theta_orth)
                        dy_test = 20 * sin(theta_orth)

                        ys, xs = np.nonzero(mask_full)
                        vx = xs - cx
                        vy = ys - cy
                        dots = vx * dx_test + vy * dy_test
                        side1 = np.count_nonzero(dots >= 0)
                        side2 = np.count_nonzero(dots < 0)

                        direction_sign = 1 if side1 > side2 else -1
                        dx = direction_sign * int(40 * cos(theta))
                        dy = direction_sign * int(40 * sin(theta))
                        angle = (degrees(atan2(dx, -dy)) + 360) % 360

                        cv2.arrowedLine(display, (cx, cy),
                                        (cx + dx, cy + dy), color, 2, tipLength=0.3)
                        cv2.circle(display, (cx, cy), 4, (255, 255, 255), -1)

                        depth_value = depth_crop_resized[
                            min(crop_size - 1, max(0, int((cy - y1_crop) / scale_y))),
                            min(crop_size - 1, max(0, int((cx - x1_crop) / scale_x)))
                        ] * 0.001
                        depth_cm = depth_value * 10

                        label_text = f"{label_name} | {score:.2f} | {depth_cm:.1f}cm"
                        orient = f"Angle: {angle:.1f}°"
                        coord = f"({cx},{cy})"
                        y_offset = y - 30
                        for text in [label_text, orient, coord]:
                            cv2.putText(display, text, (x, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            y_offset += 15

                        # Collect candidates if needed for logic

                        # Then you can filter best object of selected class
                        detected_center = None
                        best_score = 0
                        best_object = None

                        for obj in all_candidates:
                            if obj["label_id"] == selected_class_id and obj["score"] > best_score:
                                best_score = obj["score"]
                                detected_center = obj["center"]
                                best_object = obj  # Save the whole object

                            M = cv2.moments(mask_full)
                            if M["m00"] == 0:
                                continue
                            mu20, mu02, mu11 = M["mu20"], M["mu02"], M["mu11"]
                            theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

                            theta_orth = theta + np.pi / 2
                            dx_test = 20 * cos(theta_orth)
                            dy_test = 20 * sin(theta_orth)

                            ys, xs = np.nonzero(mask_full)
                            vx = xs - cx
                            vy = ys - cy
                            dots = vx * dx_test + vy * dy_test
                            side1 = np.count_nonzero(dots >= 0)
                            side2 = np.count_nonzero(dots < 0)

                            direction_sign = 1 if side1 > side2 else -1
                            dx = direction_sign * int(40 * cos(theta))
                            dy = direction_sign * int(40 * sin(theta))
                            angle = (degrees(atan2(dx, -dy)) + 360) % 360

                            cv2.arrowedLine(display, (cx, cy),
                                            (cx + dx, cy + dy), color, 2, tipLength=0.3)
                            cv2.circle(display, (cx, cy), 4, (255, 255, 255), -1)

                            depth_value = depth_crop_resized[
                                min(crop_size - 1, max(0, int((cy - y1_crop) / scale_y))),
                                min(crop_size - 1, max(0, int((cx - x1_crop) / scale_x)))
                            ] * 0.001
                            depth_cm = depth_value * 10

                            label = f"{class_names[outputs['labels'][i]]} | {score:.2f} | {depth_cm:.1f}cm"
                            orient = f"Angle: {angle:.1f}°"
                            coord = f"({cx},{cy})"
                            y_offset = y - 30
                            for text in [label, orient, coord]:
                                cv2.putText(display, text, (x, y_offset),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                y_offset += 15

                        all_candidates.append({
                            "label_id": label_id,
                            "score": score,
                            "center": (cx, cy),
                            "angle": angle,
                            "depth": depth_value,
                            "bbox": (x, y, w, h)
                        })

                    matching_objects = [
                        obj for obj in all_candidates if obj["label_id"] == selected_class_id]

                    if matching_objects:
                        current_selection_idx %= len(
                            matching_objects)  # clamp index
                        best_object = matching_objects[current_selection_idx]
                        detected_center = best_object["center"]
                    else:
                        best_object = None
                        detected_center = None

                    cv2.rectangle(display, (x1_crop, y1_crop),
                                (x2_crop, y2_crop), (0, 255, 0), 2)

                    target_pixel = clicked_pixel if clicked_pixel else detected_center

                    if target_pixel:
                        u, v = target_pixel
                        print(f"target Pixel: ", target_pixel)
                        z_depth = depth_image[v, u] * depth_scale

                        if z_depth == 0:
                            cv2.putText(display, "⚠️ No depth at target pixel", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            target_pos = None
                        else:
                            x = (u - K[0, 2]) * z_depth / K[0, 0]
                            y = (v - K[1, 2]) * z_depth / K[1, 1]
                            point_cam = np.array([x, y, z_depth, 1.0])

                            tcp = arm.position
                            pos_mm = np.array(tcp[:3])
                            angles_rad = np.radians(tcp[3:])
                            R_ee2base = euler_to_rotation_matrix(*angles_rad)
                            T_ee2base = build_T(R_ee2base, pos_mm / 1000.0)
                            T_cam2ee = build_T(R_cam2ee, t_cam2ee)
                            point_base = T_ee2base @ T_cam2ee @ point_cam

                            target_pos = point_base[:3] * 1000
                            target_pos[2] += Z_OFFSET * 1000
                            target_pos[2] = max(target_pos[2], MIN_Z_MM)
                            rot_target = rotation_matrix_to_euler(R_ee2base)
                            # Correct Rz based on detection angle

                            cv2.circle(display, (u, v), 5, (0, 255, 0), -1)
                            preview_text = [
                                f"Target TCP [mm]: x={target_pos[0]:.1f}, y={target_pos[1]:.1f}, z={target_pos[2]:.1f}",
                                f"Target Rot [deg]: Rx={rot_target[0]:.1f}, Ry={rot_target[1]:.1f}, Rz={rot_target[2]:.1f}"
                            ]
                            for i, line in enumerate(preview_text):
                                cv2.putText(display, line, (10, 30 + i * 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(display, "Click to select a 3D point or wait for detection", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

                    # === ADDED: show selected class overlay ===
                    cv2.putText(display, f"Selected Class [{selected_class_id}]: {class_names[selected_class_id]}",
                                (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    if matching_objects:
                        cv2.putText(display, f"Selected [{current_selection_idx+1}/{len(matching_objects)}] of '{class_names[selected_class_id]}'",
                                    (10, display.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    cv2.putText(display, "Press 'q' to go to menu", (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(display, "Press 'h' to go home position", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow("Click to Move", display)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break
                    elif key in [ord('+'), ord('=')]:
                        view_params['zoom'] = min(
                            view_params['zoom'] + zoom_step, max_zoom)
                    elif key in [ord('-'), ord('_')]:
                        view_params['zoom'] = max(
                            view_params['zoom'] - zoom_step, min_zoom)
                    elif key == ord('w'):
                        view_params['center_y'] = max(
                            effective_size // 2, view_params['center_y'] - move_step)
                    elif key == ord('s'):
                        view_params['center_y'] = min(
                            H - effective_size // 2, view_params['center_y'] + move_step)
                    elif key == ord('a'):
                        view_params['center_x'] = max(
                            effective_size // 2, view_params['center_x'] - move_step)
                    elif key == ord('d'):
                        view_params['center_x'] = min(
                            W - effective_size // 2, view_params['center_x'] + move_step)
                    elif key == ord('j'):
                        if matching_objects:
                            current_selection_idx = (
                                current_selection_idx - 1) % len(matching_objects)
                    elif key == ord('l'):
                        if matching_objects:
                            current_selection_idx = (
                                current_selection_idx + 1) % len(matching_objects)
                    elif key == ord('h'):
                        arm.clean_error()
                        safe_position_call(arm, x=home[0], y=home[1], z=home[2],
                                        roll=home[3], pitch=home[4], yaw=home[5],
                                        speed=25, wait=True)
                        time.sleep(1)
                    elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')]:
                        selected_class_id = int(chr(key))
                    elif key == 13 and target_pixel and target_pos is not None:
                        print("🚀 Moving to TCP:")
                        print(f"  Position [mm]: {target_pos}")
                        print(f"  Rotation [deg]: {rot_target}")
                        # Move gripper rotation
                        code, current_pose = arm.get_position(is_radian=False)
                        overlapped_obj = None
                        if best_object:
                            ready_to_pick = True
                            for obj in all_candidates:
                                if obj == best_object:
                                    continue
                                iou = compute_iou(best_object["bbox"], obj["bbox"])
                                if iou >= 0.05 and obj["depth"] < best_object["depth"]:
                                    overlapped_obj = obj
                                    break

                        if overlapped_obj:
                            print("⚠️ Overlap detected. Picking top object first.")
                            rz_temp = convert_angle_to_rz(overlapped_obj["angle"])

                            # Get center in full image space
                            cx_full = overlapped_obj["center"][0]
                            cy_full = overlapped_obj["center"][1]

                            # Get depth from original depth image
                            depth_value = depth_image[cy_full,
                                                    cx_full] * depth_scale

                            # Fallback if center has no depth
                            if depth_value == 0:
                                print("⚠️ No depth at center, sampling nearby pixels")
                                for dx, dy in [(0, 0), (5, 0), (-5, 0), (0, 5), (0, -5)]:
                                    u = cx_full + dx
                                    v = cy_full + dy
                                    if 0 <= u < W and 0 <= v < H:
                                        depth_value = depth_image[v,
                                                                u] * depth_scale
                                        if depth_value > 0:
                                            cx_full, cy_full = u, v
                                            break

                            # If still no depth, use bbox center (clamped to image bounds)
                            if depth_value == 0:
                                x, y, w, h = overlapped_obj["bbox"]
                                cx_full = min(max(x + w//2, 0), W - 1)
                                cy_full = min(max(y + h//2, 0), H - 1)
                                depth_value = depth_image[cy_full,
                                                        cx_full] * depth_scale

                            # Skip if depth is still invalid
                            if depth_value == 0 or depth_value < 0.05:
                                print(
                                    "❌ No valid depth found, skipping overlapped object")
                                continue

                            print(
                                f"Overlapped object - Center: ({cx_full}, {cy_full}), Depth: {depth_value:.3f}m")

                            # Visual feedback (optional)
                            cv2.circle(display, (cx_full, cy_full),
                                    5, (0, 255, 255), -1)

                            # Compute TCP
                            code, current_pose = arm.get_position(is_radian=False)
                            if code == 0:
                                overlapped_obj_tcp = compute_tcp_from_uv_depth(
                                    cx_full, cy_full, depth_value,
                                    K, R_cam2ee, t_cam2ee,
                                    current_pose
                                )

                                print("Target Overlap object tcp")
                                print(overlapped_obj_tcp)

                            else:
                                print(f"[ℹ️] Current pose: {current_pose}")
                            

                        print("✅ Move complete. Press Enter again or click a new point.")
                        clicked_pixel = None
                        # target_pos = None
                        return target_pos

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        arm.disconnect()
        
    
    