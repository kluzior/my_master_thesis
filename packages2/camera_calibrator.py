from packages2.robot_positions import RobotPositions
from packages2.robot_functions import RobotFunctions
from packages2.common import show_camera
import time
import socket
import cv2
import threading
import numpy as np
import os
import logging
from pathlib import Path

class CameraCalibrator:
    def __init__(self, c=None, timestamp=None):
        self.c = c
        self.max_iter = 30
        self.min_accuracy = 0.001
        self.chess_size = (8,7)
        self.frame_size = (640,480)
        self.square_size_mm = 40
        self.square_size_m = self.square_size_mm / 1000
        self.timestamp = timestamp

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'CameraCalibrator({self}) was initialized.')

    def run(self, from_file=None):
        if from_file is None:
            ret, flow_files_path = self.run_get_data()
            if ret:
                camera_intrinsic_path= self.run_calibration(flow_files_path)
        else:
            camera_intrinsic_path= self.run_calibration(from_file)
        return camera_intrinsic_path
    

    def run_get_data(self):
        frame_event = threading.Event()
        stop_event = threading.Event()
        frame_storage = {}
        camera_thread = threading.Thread(target=show_camera, args=(frame_event, frame_storage, stop_event, "Camera calibration"))
        camera_thread.start()
        robot_functions = RobotFunctions(self.c)
        robot_poses = RobotPositions()

        try:
            robot_functions.moveJ(RobotPositions.look_at_chessboard)
            folder_with_time = "images_" + self.timestamp
            directory_with_time = Path("data/results/for_camera_calib/"+folder_with_time)
            directory_with_time.mkdir(parents=True, exist_ok=True)
            i = 0
            for pose in robot_poses.camera_calib_poses:
                robot_functions.moveJ(pose)
                frame_event.set()  # signal camera thread to capture frame
                while frame_event.is_set():
                    time.sleep(0.1)  # Wait a bit before checking again
                if 'frame' in frame_storage:
                    i+=1
                    _img = frame_storage['frame']
                    gray = _img.copy()
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                    ret, _ = cv2.findChessboardCorners(gray, self.chess_size , None)
                    if not ret:
                        self._logger.warning(f"Photo no. {i} was skipped!")
                        continue
                    cv2.imshow("Captured frame", _img)
                    _img_path = f"{directory_with_time}/img_{"{:04d}".format(i)}.jpg"
                    cv2.imwrite(_img_path, _img)
                    self._logger.info(f"Photo no. {i} was saved as: {_img_path}")
                    cv2.waitKey(500)
                    cv2.destroyWindow("Captured frame")
                time.sleep(1)
        except socket.error as socketerror:
            self._logger.error("Socket error: ", socketerror)
        finally:
            stop_event.set()
            camera_thread.join()
            self._logger.info("Procedure of taking pictures for camera calibration completed.")
            return True, directory_with_time

    def get_images(self, path=''):
        images = []
        if not os.path.exists(path):
            self._logger.warning(f"The provided path does not exist: {path}")
            return
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                else:
                    self._logger.error(f"Failed to load image: {img_path}")
        self._logger.info(f"Loaded {len(images)} images from {path}")
        return images    
    
    def run_calibration(self, files_path):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, self.min_accuracy)
        objp = np.zeros((self.chess_size[0] * self.chess_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chess_size[0],0:self.chess_size[1]].T.reshape(-1,2)
        objp = objp * self.square_size_m
        images = self.get_images(files_path)
        objpoints = [] 
        imgpoints = [] 
        num = 0
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chess_size, None)
            if ret == True:
                objpoints.append(objp) 
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2) 
                self._logger.debug(f'photo {num} analyzed!')
                num += 1 
        ret, self.camera_matrix, self.distortion_params, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.frame_size, None, None)
        if ret:
            self._logger.info('Camera successfully calibrated!')
            self._logger.info(f'Calculated camera matrix: \n{self.camera_matrix}')
            self._logger.info(f"Calculated distortion parameters: \n{self.distortion_params}")
        camera_intrinsic_path = f"{files_path}/CameraParams.npz"
        np.savez(camera_intrinsic_path, cameraMatrix=self.camera_matrix, 
                                        dist=self.distortion_params, 
                                        rvecs=rvecs, 
                                        tvecs=tvecs)
        self._logger.info(f'Intrinsic camera parameters stored under path: {camera_intrinsic_path}')
        mean_error = 0 
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], self.camera_matrix, self.distortion_params)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error 
        error_value = mean_error/len(objpoints)
        self._logger.info(f'Calculated total reprojection error: \n{error_value}')
        return camera_intrinsic_path