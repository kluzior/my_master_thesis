from packages2.camera_calibrator import CameraCalibrator
from packages2.robot_functions import RobotFunctions
from packages2.start_communication import start_communication

import numpy as np
import cv2
import socket 
import math

c = None

c, s = start_communication()

robot_functions = RobotFunctions(c)


# camera_params_path = "data/results/camera_params/CameraParams_inna_kamera_17-07_16-34.npz"

cc = CameraCalibrator(c)

files_path =  "data/results/for_camera_calib/images_28-08_10-27"


print("Socket closed")
ret, flow_files_path = cc.run_get_data()

if ret:
    print("ret is true")
    cc.run_calibration(flow_files_path)

c.close()
s.close()

