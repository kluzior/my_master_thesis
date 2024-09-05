from packages2.robot_functions import RobotFunctions
from packages2.robot_positions import RobotPositions
from packages2.image_processor import ImageProcessor
from packages2.plane_determination import PlaneDeterminator
from packages2.handeye_eye_in_hand_NEW import HandEyeCalibration
from packages2.common import show_camera, pose2rvec_tvec
from packages2.neural_network import NN_Classificator

import cv2
import threading
import time
import numpy as np

from packages2.robot_poses import RobotPoses
import random



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
        self.camera_thread = threading.Thread(target=show_camera, args=(self.frame_event, self.frame_storage, self.stop_event, "State Machine"))
        self.camera_thread.start()
        self.objects_record = []
        self.bank_counters =  list(0 for _ in range(6))

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
            elif self.state == 'WaitForObjects':
                self.wait_for_objects()
            elif self.state == 'ChooseObjectToPickUp':
                target_pixel, label = self.choose_object2pick_up(strategy="sequence")
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

        self.state = 'GoToWaitPosition_init'

    def go_to_wait_position_init(self):
        print("Going to wait position...")
        ret = self.rf.moveJ_pose(self.robposes.look_at_objects)

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
        ret = self.rf.moveJ_pose(self.robposes.look_at_objects)
        if ret == 0:
            time.sleep(1)
            self.state = 'IdentificationContours'

    def identification_contours(self):
        contours_identified = False
        print("Identifying contours...")
        self.frame_event.set()  # signal camera thread to capture frame
        while self.frame_event.is_set():
            time.sleep(0.1)  # wait a bit before checking again
        if 'frame' in self.frame_storage:    
            _img = self.frame_storage['frame']
            uimg, _ = self.ip.undistort_frame(_img, self.camera_mtx, self.dist_params)
            gray = cv2.cvtColor(uimg, cv2.COLOR_BGR2GRAY)
            buimg = self.ip.apply_binarization(gray)
            final_img, _ = self.ip.ignore_background(buimg, self.table_roi_points)
            
            cv2.imshow("prepared picure", final_img)
            cv2.waitKey(1000)
            cv2.destroyWindow("prepared picure")   

            contours, _ = cv2.findContours(final_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]

            img_to_show_elipse = uimg.copy()
            img_to_show_elipse2 = uimg.copy()

            for contour in contours:
                target_pixel = self.calculate_center_point(contour)
                cropped_image, contour_frame = self.ip.crop_contour(final_img, contour)
                contour_prediction, contour_probability = self.nnc.predict_shape(cropped_image)
                self.add_record2object(contour_prediction, contour_probability, target_pixel, contour_frame)

                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(img_to_show_elipse, ellipse, (0, 255, 0), 2)
                (center, axes, angle) = ellipse
                center = tuple(map(int, center))  # Konwersja do int dla rysowania
                major_axis_length = axes[0] / 2  # Połowa dłuższej osi
                # Obliczanie punktów na końcach głównej osi elipsy
                x_offset = int(major_axis_length * np.cos(np.deg2rad(angle)))
                y_offset = int(major_axis_length * np.sin(np.deg2rad(angle)))
                
                p1 = (center[0] - x_offset, center[1] - y_offset)
                p2 = (center[0] + x_offset, center[1] + y_offset)
                
                # Rysowanie głównej osi konturu
                cv2.line(img_to_show_elipse, p1, p2, (255, 0, 0), 2)


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


        cv2.imshow("img_to_show_elipse", img_to_show_elipse)

        cv2.imshow("img_to_show_elipse2", img_to_show_elipse2)

        ### VISUALISE IDENTIFICATION RESULTS
        img_identification_result = self.nnc.visualize(uimg, self.objects_record)
        cv2.imshow("identification results", img_identification_result)

        ### ADD SAVING THIS IMAGE TO FILES!!!

        cv2.waitKey(3000)
        cv2.destroyWindow("identification results")
        cv2.destroyWindow("img_to_show_elipse")
        cv2.destroyWindow("img_to_show_elipse2")

        if not self.objects_record:
            self.state = 'WaitForObjects'
        else:
            self.state = 'ChooseObjectToPickUp'

    def wait_for_objects(self):
        print("Waiting for objects")
        time.sleep(5)
        self.state = 'IdentificationContours'

    def choose_object2pick_up(self, strategy="sequence", label=2):
        print("Choose position according to strategy")
        if strategy == "random":
            target_pixel, label = self.pick_random()
        if strategy == "sequence":
            target_pixel, label = self.search_by_labels()
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
        robot_pose = self.rf.give_pose()
        rotation_matrix, _, tvec = pose2rvec_tvec(list(robot_pose))

        target_pose, _ = self.he.calculate_point_pose2robot_base_new(self.pd.rmtx, pose.reshape(-1, 1), rotation_matrix, tvec, self.handeye_path)
        target_pose_waitpos = target_pose.copy()
        target_pose_waitpos["z"] += 0.1
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
        bank_pose["z"] += (self.bank_counters[label] + 1) * object_height

        bank_pose_waitpos = bank_pose.copy()
        bank_pose_waitpos["z"] += 0.1

        self.rf.moveJ_pose(bank_pose_waitpos)
        time.sleep(1)
        self.rf.moveL_pose(bank_pose)
        time.sleep(1)
        ret = self.rf.moveL_pose(bank_pose_waitpos)

        self.bank_counters[label] += 1

        if ret == 0:
            self.state = 'ReturnToWaitPosition'

    def return_to_wait_position(self):
        print("Returning to wait position...")

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

    def pick_random(self):
        filtered_records = [record for record in self.objects_record if record["label"] in range(1, 6)]
        if filtered_records:
            random_record = random.choice(filtered_records)
            return random_record["pixel_coords"], random_record["label"]
        return None

    def offset_z_axis(self, pose, offset):
        pose["z"] += offset
        return pose

    def calculate_center_point(self, contour):
        target_pixel = [0, 0] 
        M = cv2.moments(contour)
        target_pixel[0] = float(M['m10'] / M['m00'])
        target_pixel[1] = float(M['m01'] / M['m00'])
        print(f"cx: {target_pixel[0]}, cy: {target_pixel[1]}")
        return target_pixel