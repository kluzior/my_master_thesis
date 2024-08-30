from packages2.handeye_eye_in_hand_NEW import HandEyeCalibration
from packages2.robot_functions import RobotFunctions
from packages2.robot_positions import RobotPositions
import numpy as np
import cv2
import socket 
import math
import time
from packages2.start_communication import start_communication

def convert_pose(data):
    if len(data) != 6:
        raise ValueError("Input data must contain exactly 6 elements.")
    
    pose = {
        "base": data[0],
        "shoulder": data[1],
        "elbow": data[2],
        "wrist1": data[3],
        "wrist2": data[4],
        "wrist3": data[5]
    }
    
    return pose

c = None


c, s = start_communication()

robot_functions = RobotFunctions(c)


camera_params_path = "data/results/for_camera_calib/images_28-08_10-27/CameraParams.npz"

he = HandEyeCalibration(camera_params_path, c)

# files_path =  "data/results/for_hand_eye_calib/images_28-08_13-36" # DAJE OK WYNIKI, TCP 110mm
# files_path =  "data/results/for_hand_eye_calib/images_28-08_13-38"  # analogicznie, ale TCP 310mm

files_path =  "data/results/for_hand_eye_calib/images_30-08_14-04"

# he.run_get_data()

he.run_calibration(files_path)

# CALCULATE POSITIONS
pose1 = he.calculate_point_to_robot_base(files_path, point_shiftt=(0,0))
pose2 = he.calculate_point_to_robot_base(files_path, point_shiftt=(0,6))
pose3 = he.calculate_point_to_robot_base(files_path, point_shiftt=(7,6))
pose4 = he.calculate_point_to_robot_base(files_path, point_shiftt=(7,0))
pose5 = he.calculate_point_to_robot_base(files_path, point_shiftt=(4,3))

# SEND COMMANDS TO GO TO POSITIONS
robot_functions.moveJ_pose(convert_pose(pose1))
time.sleep(2)
robot_functions.moveJ_pose(convert_pose(pose2))
time.sleep(2)
robot_functions.moveJ_pose(convert_pose(pose3))
time.sleep(2)
robot_functions.moveJ_pose(convert_pose(pose4))
time.sleep(2)
robot_functions.moveJ_pose(convert_pose(pose5))



c.close()
s.close()
