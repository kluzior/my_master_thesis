from packages2.handeye_eye_in_hand_NEW import HandEyeCalibration
from packages2.robot_functions import RobotFunctions
from packages2.robot_positions import RobotPositions
import numpy as np
import cv2
import socket 
import math


def transformation_matrix_to_pose(matrix):
    position = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    pose = np.concatenate((position, rotation_vector.flatten()))
    return pose


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

HOST = "192.168.0.1"
PORT = 10000
print("Start listening...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(5)
c, addr = s.accept()

robot_functions = RobotFunctions(c)


camera_params_path = "data/results/for_camera_calib/images_28-08_10-27/CameraParams.npz"

he = HandEyeCalibration(camera_params_path, c)

# files_path =  "data/results/for_hand_eye_calib/images_28-08_13-36" # DAJE SPOKO WYNIKI, TCP 110mm
# files_path =  "data/results/for_hand_eye_calib/images_28-08_13-38"  # analogicznie, ale TCP 310mm

files_path =  "data/results/for_hand_eye_calib/images_28-08_13-36"

# he.run_get_data()

he.run_calibration(files_path)


# point1 = he.calculate_point_to_robot_base(files_path, point_shiftt=(0,0))
# point2 = he.calculate_point_to_robot_base(files_path, point_shiftt=(5,0))
# point3 = he.calculate_point_to_robot_base(files_path, point_shiftt=(0,5))
# point4 = he.calculate_point_to_robot_base(files_path, point_shiftt=(4,5))

# print(f"point1 (0,0):\n {point1}")
# print(f"point2 (1,0):\n {point2}")
# print(f"point3 (0,1):\n {point3}")
# print(f"point4 (1,1):\n {point4}")



# Convert the transformation matrix to a pose
# pose = transformation_matrix_to_pose(point1)

# print(f"pose: {pose}")

# conv_pose = convert_pose(pose)
# print(f"conv_pose: {conv_pose}")

# # robot_functions.moveJ(convert_pose(pose))
# robot_functions.moveJ(RobotPositions.pose_chess_0_0_point)