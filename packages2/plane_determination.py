import cv2
import numpy as np
from packages2.image_processor import ImageProcessor
from packages2.handeye_eye_in_hand import HandEyeCalibration

class PlaneDeterminator:
    def __init__(self, camera_params_path, handeye_path):
        self.camera_params_path = camera_params_path
        self.image_processor = ImageProcessor(self.camera_params_path)
        self.camera_matrix, self.dist_params = self.image_processor.load_camera_params(self.camera_params_path)
        
        self.frame = cv2.imread(handeye_path + "/waiting_pos/WAITING_POSE.jpg")
        self.uframe, self.newcameramtx = self.image_processor.undistort_frame(self.frame, self.camera_matrix, self.dist_params)
        self.handeye_path = handeye_path
        self.get_plane_points(self.uframe)

        self.he = HandEyeCalibration(self.camera_params_path)

    def get_plane_points(self, frame, chess_size=(8, 7)):
        rvec, tvec, _ = self.image_processor.calculate_rvec_tvec(frame)
        R, _ = cv2.Rodrigues(rvec)

        # Wektor normalny do płaszczyzny
        self.normal_vector = R[:, 2]  # Trzecia kolumna macierzy R

        # Punkt na płaszczyźnie
        self.point_on_plane = tvec.flatten()

        self.rmtx = R

        return self.normal_vector, self.point_on_plane

    def pixel_to_camera_plane(self, pixel):
        # Konwersja punktu 2D na promień 3D w układzie kamery
        pixel_homog = np.array([pixel[0], pixel[1], 1.0])
        ray_direction = np.linalg.inv(self.camera_matrix) @ pixel_homog

        # Przeskalowanie wektora kierunkowego promienia
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        # Punkt kamery (w układzie kamery jest to zawsze [0, 0, 0])
        camera_origin = np.array([0.0, 0.0, 0.0])

        # Wyznaczenie punktu przecięcia promienia z płaszczyzną
        numerator = np.dot(self.normal_vector, (self.point_on_plane - camera_origin))
        denominator = np.dot(self.normal_vector, ray_direction)

        if denominator == 0:
            raise ValueError("Promień jest równoległy do płaszczyzny i nie przecina jej.")

        t = numerator / denominator
        intersection_point = camera_origin + t * ray_direction

        return intersection_point


    def test_run(self):
        rvec, _, test_pixel = self.image_processor.calculate_rvec_tvec(self.uframe, point_shift=(7,6))
        test_pixel = tuple(test_pixel[0])
        pose = self.pixel_to_camera_plane(test_pixel)
        _, target_pose = self.he.calculate_point_pose2robot_base(self.rmtx, pose.reshape(-1, 1), self.handeye_path)

        return target_pose
