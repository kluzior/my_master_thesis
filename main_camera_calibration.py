from packages2.camera_calibrator import CameraCalibrator
from packages2.robot_functions import RobotFunctions
import numpy as np
import cv2
import socket 
import math

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

