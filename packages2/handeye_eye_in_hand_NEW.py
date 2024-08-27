import time
from packages2.cmd_generator import CmdGenerator  # Ensure this module is available
from packages2.robot_positions import RobotPositions  # Ensure this module is available
from packages2.robot_functions import RobotFunctions  # Ensure this module is available
from packages2.image_processor import ImageProcessor  # Ensure this module is available
# from packages2.handeye_eye_in_hand import HandEyeCalibration  # Ensure this module is available
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


class HandEyeCalibration:
    def __init__(self, camera_params_path):
        self.camera_params_path = camera_params_path
        self.image_procesor = ImageProcessor()
        self.mtx, self.dist = self.image_procesor.load_camera_params(self.camera_params_path)

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

    def calculate_rvec_tvec_from_robot_pose(self, pose):
        print(f"pose: {pose}")
        pos = list(pose)
        print(f"pos: {pos}")

        roll = pos[3]
        pitch = pos[4]
        yaw = pos[5]
        # rotation_matrix = common.RPY_to_rmtx(roll, pitch, yaw)

        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        rotation_matrix = r.as_matrix()

        # R_matrix[:3, :3] = rotation_matrix
        # print(f"rotation_matrix: {rotation_matrix}")
        rvec = cv2.Rodrigues(np.array(rotation_matrix))[0]
        tvec = np.zeros((3, 1))
        for i in range(3):
            tvec[i] = pos[i]
        # print(f"Calculated (rvec, tvec) for pose_{index}: {(rvec, tvec)}")

        return rvec, tvec, pos


    def run_get_data(self):
        frame_event = threading.Event()
        frame_storage = {}
        robot_poses = RobotPositions()
        # Create a new thread for the camera function
        camera_thread = threading.Thread(target=self.show_camera, args=(frame_event, frame_storage))
        camera_thread.start()

        HOST = "192.168.0.1"
        PORT = 10000
        print("Start listening...")
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(5)
        c, addr = s.accept()
        robot_functions = RobotFunctions(c)

        try:
            msg = c.recv(1024)
            print(msg)
            if msg == b"Hello server, robot here":
                print("Robot requested for data!")
                time.sleep(0.5)

                robot_functions.moveJ(RobotPositions.look_at_chessboard)

                robot_pose_read_from_robot = []
                folder_with_time = "images_" + datetime.datetime.now().strftime("%d-%m_%H-%M")
                waiting_pos_folder = folder_with_time + "/waiting_pos"
                directory_with_time = Path("images/for_hand_eye_calib/"+folder_with_time)
                directory_with_time.mkdir(parents=True, exist_ok=True)

                directory_waiting = Path("images/for_hand_eye_calib/"+waiting_pos_folder)
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

                    robot_waiting_pose = robot_functions.give_pose()
                    _, _, pose_waiting = self.calculate_rvec_tvec_from_robot_pose(robot_waiting_pose)

                    # Ensure the waiting_pos_folder exists
                    if not os.path.exists(waiting_pos_folder):
                        os.makedirs(waiting_pos_folder)

                    with open(os.path.join(directory_waiting, 'wait_pose.txt'), 'w') as file2:
                        file2.write(','.join(map(str, pose_waiting)) + '\n')

                    np.savez(f"{directory_waiting}/wait_pose.npz", wait_pose=pose_waiting)


                for pose in robot_poses.poses_2:
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

                        img_name = f"{directory_with_time}/img_{i}.jpg"
                        cv2.imwrite(img_name, test_img)
                        print(f"Saved {img_name}")
                        cv2.waitKey(500)  # Wait for any key press
                        cv2.destroyWindow("Requested Frame")
                    else:
                        print("Frame was not captured")
                    robot_pose = robot_functions.give_pose()
                    _, _, pose = self.calculate_rvec_tvec_from_robot_pose(robot_pose)
                    robot_pose_read_from_robot.append(pose)
                    print("ROBOT POSE APPENDED!")

                    

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
            c.close()
            s.close()
            print("Socket closed")
            cv2.destroyAllWindows()



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

    def prepare_one_mtx(self, rvec, tvec):
        mtx = np.zeros((4, 4))
        
        rmtx = cv2.Rodrigues(rvec)[0]
        mtx[:3, :3] = rmtx
        mtx[:3, 3] = tvec.ravel()
        mtx[3, 3] = 1
        return mtx 

    def prepare_one_mtx2(self, rmtx, tvec):
        mtx = np.zeros((4, 4))
        
        mtx[:3, :3] = rmtx
        mtx[:3, 3] = tvec.ravel()
        mtx[3, 3] = 1
        return mtx 

    def calculate_rvec_tvec_from_robot_pose2(self, pos):
        print(f"pos: {pos}")

        roll = pos[3]
        pitch = pos[4]
        yaw = pos[5]

        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
        rotation_matrix = r.as_matrix()


        rvec = cv2.Rodrigues(np.array(rotation_matrix))[0]
        tvec = np.zeros((3, 1))
        for i in range(3):
            tvec[i] = pos[i]

        return rvec, tvec
    
    def run_calibration(self, input_path):

        np.set_printoptions(precision=6, suppress=True)

        #DATA TAKEN FROM HERE
        # files_path = "images/for_hand_eye_calib/images_18-07_15-44"
        # files_path = "images/for_hand_eye_calib/images_27-08_11-32"
        files_path = input_path


        # Load from npz file
        data = np.load(f"{files_path}/robot_poses.npz")
        robot_pose_read_from_robot = data['robot_pose_read_from_robot']

        print(f"TEST: {len(data)}")

        images = self.get_images(files_path)


        print(len(robot_pose_read_from_robot))
        cam_rvecs = []
        cam_tvecs = []
        cam_mtxs = []
        rob_rvecs = []
        rob_tvecs = []
        rob_mtxs = []
        for robot_pose, image in zip(robot_pose_read_from_robot, images):
            image_to_show = image.copy()
            print(f"robot pose: {robot_pose}")
            # cv2.imshow("Image", image)

            u_image = self.image_procesor.undistort_frame(image, self.mtx, self.dist)
            
            # cv2.imshow("uImage", u_image)


            cam_rvec, cam_tvec = self.image_procesor.calculate_rvec_tvec(u_image)
            cam_mtx = self.prepare_one_mtx(cam_rvec, cam_tvec)
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
            cam_mtxs.append(cam_mtx)

            # print(f"Calculated object to camera rvec: {cam_rvec}")
            # print(f"Calculated object to camera tvec: {cam_tvec}")
            # print(f"Calculated object to camera mtx: {cam_mtx}")

            rob_rvec, rob_tvec = self.calculate_rvec_tvec_from_robot_pose2(robot_pose)
            rob_mtx = self.prepare_one_mtx(rob_rvec, rob_tvec)
            rob_rvecs.append(rob_rvec)
            rob_tvecs.append(rob_tvec)
            rob_mtxs.append(rob_mtx)

            # print(f"Calculated robot to base rvec: {rob_rvec}")
            # print(f"Calculated robot to base tvec: {rob_tvec}")
            # print(f"Calculated robot to base mtx: {rob_mtx}")

            gray = cv2.cvtColor(image_to_show, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (8,7), None)
            cv2.drawChessboardCorners(image_to_show, (8, 7), corners, True)

            # cv2.imshow("Image with corners", image_to_show)

            cv2.waitKey(500)
            cv2.destroyAllWindows()


        print("DONE")
        print(f"len of: {len(cam_rvecs)}")

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


        result_mtx_tsai = self.prepare_one_mtx2(R1, T1)
        result_mtx_park = self.prepare_one_mtx2(R2, T2)
        result_mtx_horaud = self.prepare_one_mtx2(R3, T3)
        result_mtx_daniilidis = self.prepare_one_mtx2(R5, T5)

        # Save to npz files
        np.savez(f"{files_path}/R_T_results_tsai.npz", camera_tcp_mtx = result_mtx_tsai)
        np.savez(f"{files_path}/R_T_results_park.npz", camera_tcp_mtx = result_mtx_park)
        np.savez(f"{files_path}/R_T_results_horaud.npz", camera_tcp_mtx = result_mtx_horaud)
        np.savez(f"{files_path}/R_T_results_daniilidis.npz", camera_tcp_mtx = result_mtx_daniilidis)







# CALCULATING 
    def calculate_point_to_robot_base(self, files_path, point_shiftt=(0,0)):
        data = np.load(f"R_T_results_tsai.npz")
        camera_tcp_mtx = data['camera_tcp_mtx']
        # image_procesor = ImageProcessor()
        # mtx, dist = image_procesor.load_camera_params("CameraParams_inna_kamera_17-07_16-34.npz")

        print(f"\ncamera_tcp_mtx: {camera_tcp_mtx}")


        data = np.load(f"{files_path}/waiting_pos/wait_pose.npz")
        wait_pose = data['wait_pose']


        # wait_pose = [-0.191371,0.109018,0.67845,-2.86535,-1.27539,-0.0543387]

        image_path = files_path + "/waiting_pos/WAITING_POSE.jpg"


        image = cv2.imread(image_path)



        uimg = self.image_procesor.undistort_frame(image, self.mtx, self.dist)






        cam_rvec, cam_tvec = self.image_procesor.calculate_rvec_tvec(uimg, point_shift=point_shiftt)
        obj_to_cam_mtx = self.prepare_one_mtx(cam_rvec, cam_tvec)

        print(f"\n obj_to_cam_mtx: {obj_to_cam_mtx}")



        # obj_to_tcp_mtx = np.dot(obj_to_cam_mtx, camera_tcp_mtx) #obj_to_cam_mtx @ camera_tcp_mtx
        # obj_to_tcp_mtx2 = np.dot(camera_tcp_mtx, obj_to_cam_mtx)

        # print(f"\nobj_to_tcp_mtx: {obj_to_tcp_mtx}\n")
        # print(f"\nobj_to_tcp_mtx2: {obj_to_tcp_mtx2}\n")

        rob_rvec, rob_tvec = self.calculate_rvec_tvec_from_robot_pose2(wait_pose)
        rob_base_mtx = self.prepare_one_mtx(rob_rvec, rob_tvec)


        cam_to_base = np.dot(rob_base_mtx, camera_tcp_mtx)



        obj_to_base_mtx = np.dot(cam_to_base, obj_to_cam_mtx)


        print(f"\n rob_mtx: {rob_base_mtx}")


        # obj_to_base_mtx = obj_to_tcp_mtx @ rob_mtx
        print(f"\nobj_to_base_mtx: {obj_to_base_mtx}")

        return obj_to_base_mtx










