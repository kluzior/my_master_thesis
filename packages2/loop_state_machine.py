from packages2.robot_functions import RobotFunctions
from packages2.robot_positions import RobotPositions
from packages2.image_processor import ImageProcessor


import cv2
import threading
import time


class LoopStateMachine:
    def __init__(self, c, camera_intrinsic_path):
        self.state = 'Initialization'
        self.ip = ImageProcessor(camera_intrinsic_path)
        self.camera_mtx, self.dist_params = self.ip.load_camera_params(camera_intrinsic_path)
        self.rf = RobotFunctions(c)


        self.frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.frame_storage = {}
        self.camera_thread = threading.Thread(target=self.show_camera, args=(self.frame_event, self.frame_storage, self.stop_event))
        self.camera_thread.start()



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
            elif self.state == 'GoToPickUpObject':
                self.go_to_pick_up_object()
            elif self.state == 'PickUpAndMove':
                self.pick_up_and_move()
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
            self.table_roi_points = self.ip.get_roi_points(_img)


        if self.table_roi_points is not None:
            self.state = 'GoToWaitPosition'

    def go_to_wait_position(self):
        print("Going to wait position...")
        ret = self.rf.moveJ(RobotPositions.look_at_chessboard)
        if ret == 0:
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

            uimg = self.ip.undistort_frame(_img, self.camera_mtx, self.dist_params)
            gray = cv2.cvtColor(uimg, cv2.COLOR_BGR2GRAY)
            buimg = self.ip.apply_binarization(gray)
            final_img, _ = self.ip.ignore_background(buimg, self.table_roi_points)
            
            cv2.imshow("prepared picure", final_img)
            cv2.waitKey(1000)
            cv2.destroyWindow("prepared picure")   



        # Logic for identifying contours
        # Simulate contour identification
        contour_identified = True
        if contour_identified:
            self.state = 'GoToPickUpObject'

    def go_to_pick_up_object(self):
        print("Going to pick up object...")
        # Logic to move robot to the contour position
        # Simulate reaching contour position
        ret = self.rf.moveJ(RobotPositions.pose_wait_base)

        if ret == 0:
            self.state = 'PickUpAndMove'

    def pick_up_and_move(self):
        print("Picking up object and moving to position...")
        # Logic to pick up the object and move to a new position
        # Simulate picking up and moving
        pick_up_and_move_done = True
        if pick_up_and_move_done:
            self.state = 'ReturnToWaitPosition'

    def return_to_wait_position(self):
        print("Returning to wait position...")
        # Logic to move robot back to wait position
        # Simulate reaching wait position
        wait_position_reached = True
        if wait_position_reached:
            self.state = 'GoToWaitPosition'

