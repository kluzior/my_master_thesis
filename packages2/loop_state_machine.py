from packages2.robot_functions import RobotFunctions
from packages2.robot_positions import RobotPositions
from packages2.image_processor import ImageProcessor
from packages2.plane_determination import PlaneDeterminator
from packages2.handeye_eye_in_hand import HandEyeCalibration
from packages2.common import show_camera, pose2rvec_tvec, pose_list_to_dict, joints_list_to_dict
from packages2.neural_network import NN_Classificator

import cv2
import threading
import time
import numpy as np

from packages2.robot_poses import RobotPoses
import random
from pathlib import Path
import datetime



class LoopStateMachine:
    def __init__(self, c, camera_intrinsic_path, handeye_path, mlp_model_path, timestamp):
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
        self.camera_thread = threading.Thread(target=show_camera, args=(self.frame_event, self.frame_storage, self.stop_event, "State Machine"))
        self.camera_thread.start()
        self.objects_record = []
        self.bank_counters =  list(0 for _ in range(6))
        self.timestamp = timestamp

        folder_with_time = "images_" + timestamp
        self.directory_with_time = Path("data/results/identification/"+folder_with_time)
        self.directory_with_time.mkdir(parents=True, exist_ok=True)
        self.virtual_plane_offset = 0.010   # mm
        self.objects_height = 0.008         # mm
        # if self.virtual_plane_offset != self.objects_height:
        self.pose_look_at_objects = self.robposes.look_at_chessboard
        self.pose_look_at_objects["z"] += self.objects_height - self.virtual_plane_offset
        print(f"self.pose_look_at_objects: {self.pose_look_at_objects}")

        self.iter_num = 100
        self.iter_count = 0

    def run(self):
        while True:
            if self.state == 'Initialization':
                self.state_initialization()
            elif self.state == 'GoToWaitPosition_init':
                self.state_go_to_wait_position_init()
            elif self.state == 'GoToWaitPosition':
                self.state_go_to_wait_position()
            elif self.state == 'IdentificationContours':
                self.state_identification_contours()
            elif self.state == 'WaitForObjects':
                self.state_wait_for_objects()
            elif self.state == 'ChooseObjectToPickUp':
                target_pixel, label, angle = self.state_choose_object2pick_up(strategy="only_label", label=5)
            elif self.state == 'GoToPickUpObject':
                self.state_go_to_pick_up_object(target_pixel, angle)
            elif self.state == 'PickUpAndMove':
                self.state_pick_up_and_move(label)
            elif self.state == 'ReturnToWaitPosition':
                self.state_return_to_wait_position()
            else:
                print("Unknown state!")
                break

    def state_initialization(self):
        print("Initializing robot...")

        self.state = 'GoToWaitPosition_init'

    def state_go_to_wait_position_init(self):
        print("Going to wait position...")
        ret = self.rf.moveJ_pose(self.pose_look_at_objects)

        self.frame_event.set()  # signal camera thread to capture frame
        while self.frame_event.is_set():
            time.sleep(0.1)  # wait a bit before checking again
        if 'frame' in self.frame_storage:
            _img = self.frame_storage['frame']
            uimg, _ = self.ip.undistort_frame(_img, self.camera_mtx, self.dist_params)
            self.table_roi_points = self.ip.get_roi_points(uimg)

        if self.table_roi_points is not None:
            self.state = 'GoToWaitPosition'

    def state_go_to_wait_position(self):
        print("Going to wait position...")
        ret = self.rf.moveJ_pose(self.pose_look_at_objects)
        if ret == 0:
            time.sleep(1)
            self.state = 'IdentificationContours'

    def state_identification_contours(self):
        contours_identified = False
        print("Identifying contours...")
        self.frame_event.set()  # signal camera thread to capture frame
        self.start_time = datetime.datetime.now()

        while self.frame_event.is_set():
            time.sleep(0.1)  # wait a bit before checking again
        if 'frame' in self.frame_storage:    
            _img = self.frame_storage['frame']
            uimg, _ = self.ip.undistort_frame(_img, self.camera_mtx, self.dist_params)
            gray = cv2.cvtColor(uimg, cv2.COLOR_BGR2GRAY)
            buimg = self.ip.apply_binarization(gray)
            final_img, _ = self.ip.ignore_background(buimg, self.table_roi_points)
            self.end_time_prepare = datetime.datetime.now()

            cv2.imshow("prepared picure", final_img)
            cv2.waitKey(1000)
            cv2.destroyWindow("prepared picure")   

            contours, _ = cv2.findContours(final_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]

            img_to_show_elipse2 = uimg.copy()

            for contour in contours:
                target_pixel = self.calculate_center_point(contour)
                cropped_image, contour_frame = self.ip.crop_contour(final_img, contour)
                contour_prediction, contour_probability = self.nnc.predict_shape(cropped_image)
                # self.add_record2object(contour_prediction, contour_probability, target_pixel, contour_frame)

                # Obliczanie momentów konturu
                M = cv2.moments(contour)

                if M['m00'] != 0:
                    # Środek masy konturu
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Obliczanie kąta orientacji
                    angle = 0.5 * np.arctan2(2 * M['mu11'], (M['mu20'] - M['mu02']))

                    # Rysowanie środka masy
                    cv2.circle(img_to_show_elipse2, (cx, cy), 5, (0, 0, 255), -1)

                    # Rysowanie osi konturu
                    length = 100  # Długość osi
                    x2 = int(cx + length * np.cos(angle))
                    y2 = int(cy + length * np.sin(angle))
                    cv2.line(img_to_show_elipse2, (cx, cy), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img_to_show_elipse2, str(angle), (cx, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)


                self.add_record2object(contour_prediction, contour_probability, target_pixel, contour_frame, angle)

        self.end_time_identification = datetime.datetime.now()

        cv2.imshow("img_to_show_elipse2", img_to_show_elipse2)

        ### VISUALISE IDENTIFICATION RESULTS
        img_identification_result = self.nnc.visualize(uimg, self.objects_record)
        cv2.imshow("identification results", img_identification_result)

        ### ADD SAVING THIS IMAGE TO FILES!!!
        _img_path = f"{self.directory_with_time}/{datetime.datetime.now().strftime("%H-%M-%S")}.jpg"
        _uimg_path = f"{self.directory_with_time}/uimg_{datetime.datetime.now().strftime("%H-%M-%S")}.jpg"
        cv2.imwrite(_img_path, img_identification_result)
        cv2.imwrite(_uimg_path, uimg)
        self.iter_count += 1
        print(f"ITER COUNT: {self.iter_count}")

        cv2.waitKey(3000)
        cv2.destroyWindow("identification results")
        cv2.destroyWindow("img_to_show_elipse2")

        if not self.objects_record:
            self.state = 'WaitForObjects'
        else:
            self.state = 'ChooseObjectToPickUp'

    def state_wait_for_objects(self):
        print("Waiting for objects")
        time.sleep(5)
        self.state = 'IdentificationContours'

    def state_choose_object2pick_up(self, strategy="sequence", label=2):
        print("Choose position according to strategy")
        if strategy == "random":
            target_pixel, label, angle = self.pick_random()
        if strategy == "sequence":
            target_pixel, label, angle = self.search_by_labels()
        if strategy == "only_label":
            target_pixel, angle = self.get_best_pixel_coords(label)
        if target_pixel is not None:
            self.state = 'GoToPickUpObject'
            return target_pixel, label, angle
        else:
            self.state = 'WaitForObjects'
            return None, None, None

    def state_go_to_pick_up_object(self, target_pixel, angle):
        print("Going to pick up object...")
        self.clear_records()
        pose = self.pd.pixel_to_camera_plane(target_pixel)
        robot_pose = self.rf.give_pose()
        rotation_matrix, _, tvec = pose2rvec_tvec(list(robot_pose))

        target_pose, _ = self.he.calculate_point_pose2robot_base_new(self.pd.rmtx, pose.reshape(-1, 1), rotation_matrix, tvec, self.handeye_path)
        target_pose["z"] -= 0.01  # ugięcie mieszków przyssawki
        self.end_time_calc_pose= datetime.datetime.now()
        ret = self.pick_object(target_pose, angle)

        print(f'ALL took {self.end_time_calc_pose - self.start_time} ')
        print(f'PHOTO REQUEST took {self.end_time_prepare - self.start_time} ')
        print(f'IDENTIFICATION took {self.end_time_identification - self.end_time_prepare} ')
        print(f'CALC TARGET POSE took {self.end_time_calc_pose - self.end_time_identification} ')

        if ret == 0:
            self.state = 'GoToWaitPosition'
            # self.state = 'PickUpAndMove'

    def state_pick_up_and_move(self, label):
        print("Picking up object and moving to position...")
        self.rf.moveJ_pose(self.robposes.before_banks)
        object_height = 0.008
        bank_pose = self.robposes.banks[label].copy()
        bank_pose["z"] += (self.bank_counters[label] + 1) * object_height
        print(f"MOVE TO LABEL: {label}")
        ret = self.drop_object(bank_pose)

        self.bank_counters[label] += 1

        if ret == 0:
            self.state = 'ReturnToWaitPosition'

    def state_return_to_wait_position(self):
        print("Returning to wait position...")

        wait_position_reached = True
        if wait_position_reached:
            self.state = 'GoToWaitPosition'


    def add_record2object(self, label, prediction, pixel_coords, contour_frame, angle):
        record = {
            "label": label,
            "prediction": prediction,
            "pixel_coords": pixel_coords,
            "contour_frame": contour_frame,
            "angle": angle
        }
        self.objects_record.append(record)
        print(f"Record added: {record}")

    def clear_records(self):
        self.objects_record.clear()

    def get_best_pixel_coords(self, label, prediction_threshold=0.96):
        filtered_records = [record for record in self.objects_record if record["label"] == label and record["prediction"] >= prediction_threshold]
        if not filtered_records:
            return None, None
        best_record = max(filtered_records, key=lambda x: x["prediction"])
        return best_record["pixel_coords"], best_record["angle"]

    def search_by_labels(self, prediction_threshold=0.96):
        for label in range(1, 6):
            filtered_records = [record for record in self.objects_record if record["label"] == label and record["prediction"] >= prediction_threshold]
            if filtered_records:
                best_record = max(filtered_records, key=lambda x: x["prediction"])
                return best_record["pixel_coords"], label, best_record["angle"]
        return None, None, None

    def pick_random(self, prediction_threshold=0.96):
        filtered_records = [record for record in self.objects_record if record["label"] in range(1, 6) and record["prediction"] >= prediction_threshold]
        if filtered_records:
            random_record = random.choice(filtered_records)
            return random_record["pixel_coords"], random_record["label"], random_record["angle"]
        return None, None, None

    def calculate_center_point(self, contour):
        target_pixel = [0, 0] 
        M = cv2.moments(contour)
        target_pixel[0] = float(M['m10'] / M['m00'])
        target_pixel[1] = float(M['m01'] / M['m00'])
        print(f"cx: {target_pixel[0]}, cy: {target_pixel[1]}")
        return target_pixel
    
    def pick_object(self, pose, angle, z_offset=0.1):
        pick_pose = pose.copy()
        pick_pose["z"] += z_offset
        self.rf.moveJ_pose(pick_pose)
        self.rotate_wrist3(angle)
        time.sleep(0.2)
        # self.rf.moveL_pose(pose)
        self.moveL_onlyZ(-z_offset)
        time.sleep(0.2)
        self.rf.set_gripper()
        time.sleep(1)       # wait longer to give time for gripper to grip
        self.rf.moveL_pose(pick_pose)
        return False

    def drop_object(self, pose, z_offset=0.1):
        drop_pose = pose.copy()
        drop_pose["z"] += z_offset
        self.rf.moveJ_pose(drop_pose)
        time.sleep(0.2)
        self.rf.moveL_pose(pose)
        time.sleep(0.2)
        self.rf.reset_gripper()
        time.sleep(0.2)
        self.rf.moveL_pose(drop_pose)
        return False
    
    def rotate_wrist3(self, angle):
        robot_joints = self.rf.give_joints()
        target_joints = joints_list_to_dict(list(robot_joints))
        target_joints["wrist3"] += angle 
        self.rf.moveJ(target_joints)

    def moveL_onlyZ(self, z):
        act_pose = self.rf.give_pose()
        act_pose = list(act_pose)
        act_pose[2] += z
        pose = pose_list_to_dict(act_pose)
        self.rf.moveL_pose(pose)

