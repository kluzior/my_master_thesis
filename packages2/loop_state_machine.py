from packages2.robot_functions import RobotFunctions
from packages2.robot_positions import RobotPositions
from packages2.image_processor import ImageProcessor
from packages2.plane_determination import PlaneDeterminator
from packages2.handeye_eye_in_hand_NEW import HandEyeCalibration
from packages2.common import show_camera
from packages2.neural_network import NN_Classificator

import cv2
import threading
import time

from packages2.robot_poses import RobotPoses



class LoopStateMachine:
    def __init__(self, c, camera_intrinsic_path, handeye_path, mlp_model_path):
        self.state = 'Initialization'
        self.ip = ImageProcessor(camera_intrinsic_path)
        self.camera_mtx, self.dist_params = self.ip.load_camera_params(camera_intrinsic_path)
        self.rf = RobotFunctions(c)
        self.handeye_path = handeye_path
        self.he = HandEyeCalibration(camera_intrinsic_path, c)

        self.robposes = RobotPoses()

        self.pd = PlaneDeterminator(camera_intrinsic_path, handeye_path)

        self.nnc = NN_Classificator(mlp_model_path)

        self.frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.frame_storage = {}
        self.camera_thread = threading.Thread(target=show_camera, args=(self.frame_event, self.frame_storage, self.stop_event))
        self.camera_thread.start()

        self.objects_record = []

        self.bank_counters =  list(0 for _ in range(5))

    def run(self):
        while True:
            if self.state == 'Initialization':
                self.initialization()
            elif self.state == 'GoToWaitPosition_init':
                self.go_to_wait_position_init()
            elif self.state == 'GoToWaitPosition':
                self.go_to_wait_position()
            elif self.state == 'IdentificationContours':
                self.identification_contours()
            elif self.state == 'ChooseObjectToPickUp':
                target_pixel, label = self.choose_object2pick_up()
            elif self.state == 'GoToPickUpObject':
                self.go_to_pick_up_object(target_pixel, label)
            elif self.state == 'PickUpAndMove':
                self.pick_up_and_move(label)
            elif self.state == 'ReturnToWaitPosition':
                self.return_to_wait_position()
            else:
                print("Unknown state!")
                break

    def initialization(self):
        print("Initializing robot...")
        # Initialization logic here
        self.state = 'GoToWaitPosition_init'

    def go_to_wait_position_init(self):
        print("Going to wait position...")
        ret = self.rf.moveJ(RobotPositions.look_at_chessboard)

        self.frame_event.set()  # signal camera thread to capture frame
        while self.frame_event.is_set():
            time.sleep(0.1)  # wait a bit before checking again
        if 'frame' in self.frame_storage:
            _img = self.frame_storage['frame']
            uimg, _ = self.ip.undistort_frame(_img, self.camera_mtx, self.dist_params)
            self.table_roi_points = self.ip.get_roi_points(uimg)

        if self.table_roi_points is not None:
            self.state = 'GoToWaitPosition'

    def go_to_wait_position(self):
        print("Going to wait position...")
        ret = self.rf.moveJ(RobotPositions.look_at_chessboard)
        if ret == 0:
            time.sleep(1)
            self.state = 'IdentificationContours'

    def identification_contours(self):
        print("Identifying contours...")
        self.frame_event.set()  # signal camera thread to capture frame
        while self.frame_event.is_set():
            time.sleep(0.1)  # wait a bit before checking again
        if 'frame' in self.frame_storage:    
            _img = self.frame_storage['frame']

            cv2.imshow("captured picure", _img)
            cv2.waitKey(1000)
            cv2.destroyWindow("captured picure")   

        ### PREPARE IMAGE
            uimg, _ = self.ip.undistort_frame(_img, self.camera_mtx, self.dist_params)
            gray = cv2.cvtColor(uimg, cv2.COLOR_BGR2GRAY)
            buimg = self.ip.apply_binarization(gray)
            final_img, _ = self.ip.ignore_background(buimg, self.table_roi_points)
            
            cv2.imshow("prepared picure", final_img)
            cv2.waitKey(1000)
            cv2.destroyWindow("prepared picure")   
        ###

        ### FIND CONTOURS
            contours, _ = cv2.findContours(final_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]
        ###

            for contour in contours:
            ### CALCULATE CENTER POINT + CHECK IF IT GRIPPABLE
                target_pixel = [0, 0] 
                M = cv2.moments(contour)
                target_pixel[0] = float(M['m10'] / M['m00'])
                target_pixel[1] = float(M['m01'] / M['m00'])
                print(f"cx: {target_pixel[0]}, cy: {target_pixel[1]}")
            ###

                cropped_image, contour_frame = self.ip.crop_contour(final_img, contour)

                contour_prediction, contour_probability = self.nnc.predict_shape(cropped_image)

                self.add_record2object(contour_prediction, contour_probability, target_pixel, contour_frame)

        ### VISUALISE IDENTIFICATION RESULTS
        cv2.imshow("identification results", self.nnc.visualize(uimg, self.objects_record))

        ### ADD SAVING THIS IMAGE TO FILES!!!

        cv2.waitKey(5000)
        cv2.destroyWindow("identification results")

        contour_identified = True
        if contour_identified:
            self.state = 'ChooseObjectToPickUp'

    def choose_object2pick_up(self, strategy="sequence", label=2):
        print("Choose position according to strategy")

        ## strategy - random


        ## strategy - po kolei
        if strategy == "sequence":
            target_pixel, label = self.search_by_labels()

        ## strategy tylko label on input
        if strategy == "only_label":
            target_pixel = self.get_best_pixel_coords(label)

        wait_position_reached = True
        if wait_position_reached:
            self.state = 'GoToPickUpObject'
            return target_pixel, label



    def go_to_pick_up_object(self, target_pixel, label):
        print("Going to pick up object...")
        self.clear_records()
        pose = self.pd.pixel_to_camera_plane(target_pixel)
        print(f"pose: {pose}")
        target_pose, _ = self.he.calculate_point_pose2robot_base(self.pd.rmtx, pose.reshape(-1, 1), self.handeye_path)
        print(f"target_pose: {target_pose}")


        target_pose_waitpos = target_pose.copy()
        target_pose_waitpos["z"] += 0.1
        # Logic to move robot to the contour position
        # Simulate reaching contour position
        self.rf.moveJ_pose(target_pose_waitpos)
        time.sleep(1)
        ret = self.rf.moveL_pose(target_pose)
        time.sleep(1)
        ret = self.rf.moveL_pose(target_pose_waitpos)


        if ret == 0:
            self.state = 'PickUpAndMove'

    def pick_up_and_move(self, label):
        print("Picking up object and moving to position...")
        object_height = 0.008
        bank_pose = self.robposes.banks[label]
        print(f"debug bank pose{bank_pose}")
        bank_pose["z"] += self.bank_counters[label] * object_height

        bank_pose_waitpos = bank_pose.copy()
        bank_pose_waitpos["z"] += 0.1


        self.rf.moveJ_pose(bank_pose_waitpos)
        time.sleep(1)
        self.rf.moveL_pose(bank_pose)
        time.sleep(1)
        ret = self.rf.moveL_pose(bank_pose_waitpos)

        self.bank_counters[label] += 1

        print(f"BANK COUNTER TEST [{label}]: {self.bank_counters[label]}")

        # pick_up_and_move_done = True
        if ret == 0:
            self.state = 'ReturnToWaitPosition'

    def return_to_wait_position(self):
        print("Returning to wait position...")
        # Logic to move robot back to wait position
        # Simulate reaching wait position
        wait_position_reached = True
        if wait_position_reached:
            self.state = 'GoToWaitPosition'






    def add_record2object(self, label, prediction, pixel_coords, contour_frame):
        record = {
            "label": label,
            "prediction": prediction,
            "pixel_coords": pixel_coords,
            "contour_frame": contour_frame
        }
        self.objects_record.append(record)
        print(f"Record added: {record}")

    def clear_records(self):
        self.objects_record.clear()

    def get_best_pixel_coords(self, label):
        filtered_records = [record for record in self.objects_record if record["label"] == label]
        if not filtered_records:
            return None
        best_record = max(filtered_records, key=lambda x: x["prediction"])
        return best_record["pixel_coords"]
    
    def search_by_labels(self):
        for label in range(1, 6):
            filtered_records = [record for record in self.objects_record if record["label"] == label]
            if filtered_records:
                best_record = max(filtered_records, key=lambda x: x["prediction"])
                return best_record["pixel_coords"], label
        return None
    

    def offset_z_axis(self, pose, offset):
        pose["z"] += offset
        return pose
