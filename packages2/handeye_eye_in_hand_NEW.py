import time
from packages2.robot_positions import RobotPositions
from packages2.robot_functions import RobotFunctions
from packages2.image_processor import ImageProcessor
from packages2.common import show_camera, pose2rvec_tvec, rvec_tvec2pose, pose_list_to_dict
import socket
import cv2
import threading
import numpy as np
import datetime
from pathlib import Path
import os
from scipy.spatial.transform import Rotation as R
import logging

class HandEyeCalibration:
    def __init__(self, camera_params_path, c=None):
        self.camera_params_path = camera_params_path
        self.image_procesor = ImageProcessor(self.camera_params_path)
        self.mtx, self.dist = self.image_procesor.load_camera_params(self.camera_params_path)
        self.c = c
        self.chess_size = (8,7)
        self.robot_functions = RobotFunctions(self.c)

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'HandEyeCalibration({self}) was initialized.')

    def run(self, from_file=None):
        if from_file is None:
            ret, flow_files_path = self.run_get_data()
            if ret:
                results_path = self.run_calibration(flow_files_path)
        else:
            results_path = self.run_calibration(from_file)
        return str(results_path)

    def run_get_data(self):
        frame_event = threading.Event()
        stop_event = threading.Event()
        frame_storage = {}
        camera_thread = threading.Thread(target=show_camera, args=(frame_event, frame_storage, stop_event))
        camera_thread.start()

        robot_functions = RobotFunctions(self.c)
        robot_poses = RobotPositions()

        try:
            robot_functions.moveJ(RobotPositions.look_at_chessboard)

            folder_with_time = "images_" + datetime.datetime.now().strftime("%d-%m_%H-%M")
            directory_with_time = Path("data/results/for_hand_eye_calib/"+folder_with_time)
            directory_with_time.mkdir(parents=True, exist_ok=True)

            waiting_pos_folder = folder_with_time + "/waiting_pos"
            directory_waiting = Path("data/results/for_hand_eye_calib/"+waiting_pos_folder)
            directory_waiting.mkdir(parents=True, exist_ok=True)

            frame_event.set()  # signal camera thread to capture frame
            while frame_event.is_set():
                time.sleep(0.1)  # wait a bit before checking again

            if 'frame' in frame_storage:
                _img = frame_storage['frame']
                _img_path = f"{directory_waiting}/WAITING_POSE.jpg"
                cv2.imwrite(_img_path, _img)
                self._logger.info(f"Photo of waiting pose was saved as: {_img_path}")

                robot_waiting_pose = robot_functions.give_pose()
                pose_waiting = list(robot_waiting_pose)

                # debug saving to .txt
                with open(os.path.join(directory_waiting, 'wait_pose.txt'), 'w') as file2:
                    file2.write(','.join(map(str, pose_waiting)) + '\n')

                _pose_path = f"{directory_waiting}/wait_pose.npz"
                np.savez(_pose_path, wait_pose=pose_waiting)
                self._logger.info(f"Robot waiting pose was saved as: {_pose_path}")
            
            robot_poses_read_from_robot = []
            i = 0
            n = 0
            for pose in robot_poses.poses_2:
                robot_functions.moveJ(pose)

                frame_event.set()
                while frame_event.is_set():
                    time.sleep(0.1)
                
                if 'frame' in frame_storage:
                    i+=1
                    _img = frame_storage['frame']
                    gray = _img.copy()
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                    ugray, _ = self.image_procesor.undistort_frame(gray, self.mtx, self.dist)
                    ret, _ = cv2.findChessboardCorners(ugray, self.chess_size, None)
                    if not ret:
                        self._logger.warning(f"Photo no. {i} was skipped!")
                        continue
                    cv2.imshow("Captured frame", _img)

                    _img_path = f"{directory_with_time}/img_{"{:04d}".format(i)}.jpg"
                    cv2.imwrite(_img_path, _img); n+=1
                    self._logger.info(f"Raw photo no. {i} was saved as: {_img_path}")
                    cv2.waitKey(500)
                    cv2.destroyWindow("Captured frame")
                robot_pose = robot_functions.give_pose()
                robot_poses_read_from_robot.append(list(robot_pose))
                self._logger.info(f"Robot pose no. {i} was appended to list")
                time.sleep(1)

            # debug saving to .txt
            with open(os.path.join(directory_with_time, 'robot_poses.txt'), 'w') as file:
                for pose in robot_poses_read_from_robot:
                    file.write(','.join(map(str, pose)) + '\n')
            
            _pose_path = f"{directory_with_time}/robot_poses.npz"
            np.savez(_pose_path, robot_poses_read_from_robot=robot_poses_read_from_robot, pose_waiting=pose_waiting)
            self._logger.info(f"Robot poses was saved as: {_pose_path}")
            
            self._logger.info(f"number of stored photos: {n} number of stored poses: {len(robot_poses_read_from_robot)}")
        except socket.error as socketerror:
            self._logger.error("Socket error: ", socketerror)
        finally:
            stop_event.set()
            camera_thread.join()            
            self._logger.info("Procedure of taking pictures for hand-eye calibration completed.")
        
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
        data = np.load(f"{files_path}/robot_poses.npz")
        robot_poses_read_from_robot = data['robot_poses_read_from_robot']
        images = self.get_images(files_path)
        self._logger.info(f"number of loaded photos: {len(images)} number of loaded poses: {len(robot_poses_read_from_robot)}")
        cam_rvecs = []
        cam_tvecs = []
        rob_rvecs = []
        rob_tvecs = []
        for robot_pose, image in zip(robot_poses_read_from_robot, images):
            u_image, _ = self.image_procesor.undistort_frame(image, self.mtx, self.dist)
            cam_rvec, cam_tvec, _ = self.image_procesor.calculate_rvec_tvec(u_image)
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
            rob_rvecs.append(robot_pose[3:])
            rob_tvecs.append(robot_pose[:3])

        R1, T1 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_TSAI)   
        self._logger.info("\nResults of hand-eye calibration with cv2.CALIB_HAND_EYE_TSAI")
        self._logger.info(f"R: {R1}")
        self._logger.info(f"T: {T1}")

        R2, T2 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_PARK )   
        self._logger.info("\nResults of hand-eye calibration with cv2.CALIB_HAND_EYE_PARK")
        self._logger.info(f"R: {R2}")
        self._logger.info(f"T: {T2}")

        R3, T3 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_HORAUD)
        self._logger.info("\nResults of hand-eye calibration with cv2.CALIB_HAND_EYE_HORAUD")
        self._logger.info(f"R: {R3}")
        self._logger.info(f"T: {T3}")

        R4, T4 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
        self._logger.info("\nResults of hand-eye calibration with cv2.CALIB_HAND_EYE_DANIILIDIS")
        self._logger.info(f"R: {R4}")
        self._logger.info(f"T: {T4}")

        np.savez(f"{files_path}/R_T_results_tsai.npz", camera_tcp_rmtx = R1, camera_tcp_tvec = T1)
        np.savez(f"{files_path}/R_T_results_park.npz", camera_tcp_rmtx = R2, camera_tcp_tvec = T2)
        np.savez(f"{files_path}/R_T_results_horaud.npz", camera_tcp_rmtx = R3, camera_tcp_tvec = T3)
        np.savez(f"{files_path}/R_T_results_daniilidis.npz", camera_tcp_rmtx = R4, camera_tcp_tvec = T4)
        self._logger.info(f"Results of hand-eye calibration from all methods saved to .npz file")
        return files_path

    def combine_transformations(self, R1, t1, R2, t2):
        R_combined = np.dot(R2, R1)
        t_combined = np.dot(R2, t1) + t2
        return R_combined, t_combined

    def determine_rvec_tvec_for_chess_point(self, img_path, point_shift=(0,0)):
        image = cv2.imread(img_path)
        uimg, _ = self.image_procesor.undistort_frame(image, self.mtx, self.dist)
        obj2cam_rvec, obj2cam_tvec, _ = self.image_procesor.calculate_rvec_tvec(uimg, point_shift=point_shift)
        obj2cam_rmtx, _ = cv2.Rodrigues(obj2cam_rvec)
        return obj2cam_rmtx, obj2cam_tvec

    def calculate_point_pose2robot_base(self, obj2cam_rmtx, obj2cam_tvec, handeye_path, handeye_type='tsai'):
        handeye_files = {
            'tsai'      : 'R_T_results_tsai.npz',
            'park'      : 'R_T_results_park.npz',
            'horaud'    : 'R_T_results_horaud.npz',
            'daniilidis': 'R_T_results_daniilidis.npz'
        }
        if handeye_type not in handeye_files:
            raise ValueError(f"Invalid handeye_type: {handeye_type}. Must be one of {list(handeye_files.keys())}")
        
        # Import calculated camera to TCP transformation matrix
        data = np.load(f"{handeye_path}/{handeye_files[handeye_type]}")
        camera2tcp_rmtx = data['camera_tcp_rmtx']
        camera2tcp_tvec = data['camera_tcp_tvec']

        # Get TCP to robot base transformation matrix
        data = np.load(f"{handeye_path}/waiting_pos/wait_pose.npz")
        wait_pose = data['wait_pose']
        tcp2base_rmtx, tcp2base_rvec, tcp2base_tvec = pose2rvec_tvec(wait_pose)

        # Calculate object to TCP transformation matrix
        obj2tcp_rmtx, obj2tcp_tvec = self.combine_transformations(obj2cam_rmtx, obj2cam_tvec, camera2tcp_rmtx, camera2tcp_tvec)

        # Calculate object to base transformation matrix
        obj2base_rmtx, obj2base_tvec = self.combine_transformations(obj2tcp_rmtx, obj2tcp_tvec, tcp2base_rmtx, tcp2base_tvec)
        
        obj2base_rvec, _ = cv2.Rodrigues(obj2base_rmtx)
        final_pose = rvec_tvec2pose(obj2base_rvec, obj2base_tvec)
        final_pose_dict = pose_list_to_dict(final_pose)
        return final_pose_dict, final_pose

    def generate_test_pose(self, files_path, point_shift=(0,0), handeye_type='tsai'):
        chess_path = files_path + "/waiting_pos/WAITING_POSE.jpg"
        obj2cam_rmtx, obj2cam_tvec = self.determine_rvec_tvec_for_chess_point(chess_path, point_shift)
        _, final_pose = self.calculate_point_pose2robot_base(obj2cam_rmtx, obj2cam_tvec, files_path, handeye_type)
        final_pose_dict = pose_list_to_dict(final_pose)
        return final_pose_dict
    
    def send_robot_to_test_poses(self, files_path, handeye_type='tsai'):
        test_chess_points = [(0,0), (0,6), (7,6), (7,0), (4,3)]

        for point_shift in test_chess_points:
            # calculate pose
            target_pose = self.generate_test_pose(str(files_path), point_shift, handeye_type)
            self._logger.info(f"calculated pose: {target_pose}")
            target_pose_waitpos = target_pose.copy()
            target_pose_waitpos["z"] += 0.1
            # send robot to poses
            self.robot_functions.moveJ_pose(target_pose_waitpos)
            time.sleep(1)
            self.robot_functions.moveL_pose(target_pose)
            time.sleep(1)
            self.robot_functions.moveJ_pose(target_pose_waitpos)

