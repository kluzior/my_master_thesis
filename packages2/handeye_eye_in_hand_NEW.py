import time
from packages2.robot_positions import RobotPositions
from packages2.robot_functions import RobotFunctions
from packages2.image_processor import ImageProcessor
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

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'HandEyeCalibration({self}) was initialized.')

    def run(self, from_file=None):
        if from_file is None:
            ret, flow_files_path = self.run_get_data()
            if ret:
                self.run_calibration(flow_files_path)
        else:
            self.run_calibration(from_file)

    def show_camera(self, frame_event, frame_storage, stop_event):
        cap = cv2.VideoCapture(1)
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Live camera view", frame)
            if frame_event.is_set():
                frame_storage['frame'] = frame
                frame_event.clear()
            if cv2.waitKey(1) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def run_get_data(self):
        frame_event = threading.Event()
        stop_event = threading.Event()
        frame_storage = {}
        camera_thread = threading.Thread(target=self.show_camera, args=(frame_event, frame_storage, stop_event))
        camera_thread.start()

        robot_functions = RobotFunctions(self.c)
        robot_poses = RobotPositions()

        try:
            robot_functions.moveJ(RobotPositions.look_at_chessboard)

            robot_pose_read_from_robot = []
            folder_with_time = "images_" + datetime.datetime.now().strftime("%d-%m_%H-%M")
            waiting_pos_folder = folder_with_time + "/waiting_pos"
            directory_with_time = Path("data/results/for_hand_eye_calib/"+folder_with_time)
            directory_with_time.mkdir(parents=True, exist_ok=True)

            directory_waiting = Path("data/results/for_hand_eye_calib/"+waiting_pos_folder)
            directory_waiting.mkdir(parents=True, exist_ok=True)


            frame_event.set()  # Signal camera thread to capture frame
            
            while frame_event.is_set():
                time.sleep(0.1)  # Wait a bit before checking again

            if 'frame' in frame_storage:
                _img = frame_storage['frame']
                img_name = f"{directory_waiting}/WAITING_POSE.jpg"
                cv2.imwrite(img_name, _img)

                robot_waiting_pose = robot_functions.give_pose()
                pose_waiting = list(robot_waiting_pose)
                print(f"pose_waiting: {pose_waiting}")

                with open(os.path.join(directory_waiting, 'wait_pose.txt'), 'w') as file2:
                    file2.write(','.join(map(str, pose_waiting)) + '\n')

                np.savez(f"{directory_waiting}/wait_pose.npz", wait_pose=pose_waiting)
            
            i = 0
            for pose in robot_poses.new_handeye_poses:
                robot_functions.moveJ(pose)

                frame_event.set()  # Signal camera thread to capture frame
                
                while frame_event.is_set():
                    time.sleep(0.1)  # Wait a bit before checking again

                if 'frame' in frame_storage:
                    i+=1
                    _img = frame_storage['frame']
                    gray = _img.copy()
                    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                    ugray = self.image_procesor.undistort_frame(gray, self.mtx, self.dist)
                    ret, _ = cv2.findChessboardCorners(ugray, (7,8), None)
                    if not ret:
                        print("****** PHOTO SKIPPED ******")
                        continue
                    cv2.imshow("Captured frame", _img)

                    img_name = f"{directory_with_time}/img_{"{:04d}".format(i)}.jpg"
                    cv2.imwrite(img_name, _img)
                    print(f"Saved {img_name}")
                    cv2.waitKey(500)
                    cv2.destroyWindow("Captured frame")


                robot_pose = robot_functions.give_pose()
                pose = list(robot_pose)
                robot_pose_read_from_robot.append(pose)
                print("ROBOT POSE APPENDED!")
                print(f"appended robot pose idx = {i}")
                

                time.sleep(1)

            print("DONE")
            print(f"robot_poses: {robot_pose_read_from_robot}")
            print(f"i: {i}")
            print(f"lenPose: {len(robot_pose_read_from_robot)}")
            # Save to npz file
            np.savez(f"{directory_with_time}/robot_poses.npz", robot_pose_read_from_robot=robot_pose_read_from_robot, pose_waiting=pose_waiting)

            # Save to txt file
            with open(os.path.join(directory_with_time, 'robot_poses.txt'), 'w') as file:
                for pose in robot_pose_read_from_robot:
                    file.write(','.join(map(str, pose)) + '\n')

        except socket.error as socketerror:
            print("Socket error: ", socketerror)
        finally:
            stop_event.set()
            camera_thread.join()            
        
        return True, directory_with_time



# CALIBRATION
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

        # Load from npz file
        data = np.load(f"{files_path}/robot_poses.npz")
        robot_pose_read_from_robot = data['robot_pose_read_from_robot']

        images = self.get_images(files_path)
        print(f"LEN OF IMAGES: {len(images)}")
        print(f"LEN OF ROBOT POSES: {len(robot_pose_read_from_robot)}")
        cam_rvecs = []
        cam_tvecs = []
        rob_rvecs = []
        rob_tvecs = []
        for robot_pose, image in zip(robot_pose_read_from_robot, images):
            u_image = self.image_procesor.undistort_frame(image, self.mtx, self.dist)

            cam_rvec, cam_tvec, _ = self.image_procesor.calculate_rvec_tvec(u_image)
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)

            rob_rvecs.append(robot_pose[3:])
            rob_tvecs.append(robot_pose[:3])

        print("CALIBRATION DONE")

        # proceed with OpenCv calibration
        R1, T1 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_TSAI)   
        print("RESULTS OF CALIBRATION (cv2.CALIB_HAND_EYE_TSAI)")
        print(f"R: {R1}")
        print(f"T: {T1}")

        R2, T2 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_PARK )   
        print("RESULTS OF CALIBRATION (cv2.CALIB_HAND_EYE_PARK )")
        print(f"R: {R2}")
        print(f"T: {T2}")

        R3, T3 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_HORAUD)
        print("RESULTS OF CALIBRATION (cv2.CALIB_HAND_EYE_HORAUD)")
        print(f"R: {R3}")
        print(f"T: {T3}")

        R5, T5 = cv2.calibrateHandEye(rob_rvecs, rob_tvecs, cam_rvecs, cam_tvecs, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
        print("RESULTS OF CALIBRATION (cv2.CALIB_HAND_EYE_DANIILIDIS)")
        print(f"R: {R5}")
        print(f"T: {T5}")


        # Save to npz files
        np.savez(f"{files_path}/R_T_results_tsai.npz", camera_tcp_rmtx = R1, camera_tcp_tvec = T1)
        np.savez(f"{files_path}/R_T_results_park.npz", camera_tcp_rmtx = R2, camera_tcp_tvec = T2)
        np.savez(f"{files_path}/R_T_results_horaud.npz", camera_tcp_rmtx = R3, camera_tcp_tvec = T3)
        np.savez(f"{files_path}/R_T_results_daniilidis.npz", camera_tcp_rmtx = R5, camera_tcp_tvec = T5)



    def pose_to_matrix(self, pose):
        tvec = np.array(pose[:3])
        rvec = np.array(pose[3:])
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        return rotation_matrix, rvec, tvec.reshape(-1, 1)

    def rvec_tvec2pose(self, rvec, tvec):
        x, y, z = tvec.flatten()
        Rx, Ry, Rz = rvec.flatten()
        pose = np.array([x, y, z, Rx, Ry, Rz])
        return pose

    def combine_transformations(self, R1, t1, R2, t2):
        R_combined = np.dot(R2, R1)
        t_combined = np.dot(R2, t1) + t2
        return R_combined, t_combined

# CALCULATING 
    def calculate_point_to_robot_base(self, files_path, point_shiftt=(0,0)):
        # Get object to camera transformation matrix
        image_path = files_path + "/waiting_pos/WAITING_POSE.jpg"
        image = cv2.imread(image_path)
        uimg = self.image_procesor.undistort_frame(image, self.mtx, self.dist)
        obj2cam_rvec, obj2cam_tvec, _ = self.image_procesor.calculate_rvec_tvec(uimg, point_shift=point_shiftt)
        obj2cam_rmtx, _ = cv2.Rodrigues(obj2cam_rvec)
        print(f"obj2cam_rvec: {obj2cam_rvec}")
        print(f"obj2cam_rmtx: {obj2cam_rmtx}")
        print(f"obj2cam_tvec: {obj2cam_tvec}")

        # Import calculated camera to TCP transformation matrix
        data = np.load(f"{files_path}/R_T_results_tsai.npz")
        camera2tcp_rmtx = data['camera_tcp_rmtx']
        camera2tcp_tvec = data['camera_tcp_tvec']
        print(f"camera2tcp_rmtx: {camera2tcp_rmtx}")
        print(f"camera2tcp_tvec: {camera2tcp_tvec}")

        # Calculate object to TCP transformation matrix
        obj2tcp_rmtx, obj2tcp_tvec = self.combine_transformations(obj2cam_rmtx, obj2cam_tvec, camera2tcp_rmtx, camera2tcp_tvec)
        print(f"obj2tcp_rmtx: {obj2tcp_rmtx}")
        print(f"obj2tcp_tvec: {obj2tcp_tvec}")

        # Get TCP to robot base transformation matrix
        data = np.load(f"{files_path}/waiting_pos/wait_pose.npz")
        wait_pose = data['wait_pose']
        print(f"wait pose: {wait_pose}")
        tcp2base_rmtx, tcp2base_rvec, tcp2base_tvec = self.pose_to_matrix(wait_pose)
        print(f"tcp2base_rmtx: {tcp2base_rmtx}")
        print(f"tcp2base_rvec: {tcp2base_rvec}")
        print(f"tcp_to_base_tvec: {tcp2base_tvec}")

        # Calculate object to base transformation matrix
        obj2base_rmtx, obj2base_tvec = self.combine_transformations(obj2tcp_rmtx, obj2tcp_tvec, tcp2base_rmtx, tcp2base_tvec)
        print(f"obj2base_rmtx: {obj2base_rmtx}")
        print(f"obj2base_tvec: {obj2base_tvec}")
        obj2base_rvec, _ = cv2.Rodrigues(obj2base_rmtx)
        print(f"obj2base_rvec: {obj2base_rvec}")

        return self.rvec_tvec2pose(obj2base_rvec, obj2base_tvec)
    











