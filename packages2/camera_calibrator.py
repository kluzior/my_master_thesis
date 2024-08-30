import time
from packages2.cmd_generator import CmdGenerator  # Ensure this module is available
from packages2.robot_positions import RobotPositions  # Ensure this module is available
from packages2.robot_functions import RobotFunctions  # Ensure this module is available
from packages2.image_processor import ImageProcessor  # Ensure this module is available
import socket
import cv2
import threading
import numpy as np
import pickle
import packages2.common as common
import datetime
from pathlib import Path
import os
from scipy.spatial.transform import Rotation as R


class CameraCalibrator:
    def __init__(self, c=None):
        self.c = c

        self.max_iter = 30
        self.min_accuracy = 0.001
        self.chess_size = (8,7)
        self.frame_size = (640,480)
        self.square_size_mm = 40
        self.square_size_m = self.square_size_mm / 1000

    def show_camera(self, frame_event, frame_storage):
        cap = cv2.VideoCapture(1)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("All Frames", frame)  # Display each frame
            if frame_event.is_set():
                frame_storage['frame'] = frame  # Store frame in dictionary on request
                frame_event.clear()  # Reset event after storing frame
            if cv2.waitKey(1) == ord('q'):  # Allow exiting the loop by pressing 'q'
                break

        cap.release()
        cv2.destroyAllWindows()

    def run_get_data(self):
        frame_event = threading.Event()
        frame_storage = {}
        robot_poses = RobotPositions()
        # Create a new thread for the camera function
        camera_thread = threading.Thread(target=self.show_camera, args=(frame_event, frame_storage))
        camera_thread.start()

        # HOST = "192.168.0.1"
        # PORT = 10000
        # print("Start listening...")
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # s.bind((HOST, PORT))
        # s.listen(5)
        # c, addr = s.accept()
        robot_functions = RobotFunctions(self.c)

        try:
            msg = self.c.recv(1024)
            print(msg)
            if msg == b"Hello server, robot here":
                print("Robot requested for data!")
                time.sleep(0.5)

                robot_functions.moveJ(RobotPositions.look_at_chessboard)

                folder_with_time = "images_" + datetime.datetime.now().strftime("%d-%m_%H-%M")
                waiting_pos_folder = folder_with_time + "/waiting_pos"
                directory_with_time = Path("data/results/for_camera_calib/"+folder_with_time)
                directory_with_time.mkdir(parents=True, exist_ok=True)

                directory_waiting = Path("data/results/for_camera_calib/"+waiting_pos_folder)
                directory_waiting.mkdir(parents=True, exist_ok=True)
                i = 0

                frame_event.set()  # Signal camera thread to capture frame
                
                # Wait for the frame to be stored
                while frame_event.is_set():
                    time.sleep(0.1)  # Wait a bit before checking again

                # Now the frame should be available in frame_storage['frame']
                if 'frame' in frame_storage:
                    _img = frame_storage['frame']
                    img_name = f"{directory_waiting}/WAITING_POSE.jpg"
                    cv2.imwrite(img_name, _img)

                for pose in robot_poses.camera_calib_poses:
                    robot_functions.moveJ(pose)

                    frame_event.set()  # Signal camera thread to capture frame
                    
                    # Wait for the frame to be stored
                    while frame_event.is_set():
                        time.sleep(0.1)  # Wait a bit before checking again

                    # Now the frame should be available in frame_storage['frame']
                    if 'frame' in frame_storage:
                        i+=1
                        test_img = frame_storage['frame']
                            
                        gray = test_img.copy()
                        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
                        ret, corners = cv2.findChessboardCorners(gray, (8,7), None)
                        if not ret:
                            print("****** PHOTO SKIPPED ******")
                            continue
                        cv2.imshow("Requested Frame", test_img)

                        img_name = f"{directory_with_time}/img_{"{:03d}".format(i)}.jpg"
                        cv2.imwrite(img_name, test_img)
                        print(f"Saved {img_name}")
                        cv2.waitKey(500)  # Wait for any key press
                        cv2.destroyWindow("Requested Frame")
                    else:
                        print("Frame was not captured")
                    time.sleep(1)

                print("DONE")
                # cv2.destroyAllWindows()
                return True, directory_with_time

        except socket.error as socketerror:
            print("Socket error: ", socketerror)
        # finally:
        #     stop_event.set()  # Signal the camera thread to stop
        #     camera_thread.join()  # Wait for the camera thread to finish




# CALIBRATION
    def get_images(self, path=''):
        images = []
        # Sprawdź, czy ścieżka jest poprawna
        if not os.path.exists(path):
            print(f"The provided path does not exist: {path}")
            return

        # Przeglądaj pliki w folderze
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                else:
                    print(f"Failed to load image: {img_path}")
        print(f"Loaded {len(images)} images from {path}")
        return images    
    
    def run_calibration(self, input_path):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, self.min_accuracy)

        objp = np.zeros((self.chess_size[0] * self.chess_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chess_size[0],0:self.chess_size[1]].T.reshape(-1,2)
        objp = objp * self.square_size_m



        np.set_printoptions(precision=6, suppress=True)

        #DATA TAKEN FROM HERE
        files_path = input_path


        # Load from npz file
        images = self.get_images(files_path)

        objpoints = [] 
        imgpoints = [] 
        num = 0

        for image in images:
            image_to_show = image.copy()
            # cv2.imshow("Image", image)

            gray = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.chess_size, None)
            if ret == True:
                objpoints.append(objp) 
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2) 
                # cv2.drawChessboardCorners(img, self.chess_size, corners2, ret)             
                # self.save_image(write_path, num, img)   
                # logging.debug(f'photo {num} analyzed!')
                num += 1 
            # cv2.drawChessboardCorners(image_to_show, (8, 7), corners, True)

            # cv2.imshow("Image with corners", image_to_show)

        ret, self.camera_matrix, self.distortion_params, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.frame_size, None, None)



            # cv2.waitKey(500)
            # cv2.destroyAllWindows()

        # Zapis wyników kalibracji do pliku
        np.savez(f"{files_path}/CameraParams.npz", 
                cameraMatrix=self.camera_matrix, 
                dist=self.distortion_params, 
                rvecs=rvecs, 
                tvecs=tvecs)

        print('Camera Calibrated!')
        print(f'Calculated camera matrix: {self.camera_matrix}')
        print(f"Calculated distortion parameters: {self.distortion_params}")
        # print(f'Calculated rotation vectors: {rvecs}')
        # print(f'Calculated translation vectors: {tvecs}')

        # calculate reprojection error
        mean_error = 0 
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], self.camera_matrix, self.distortion_params)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error 
        error_value = mean_error/len(objpoints)
        print(f'total reprojection error: {error_value}')
