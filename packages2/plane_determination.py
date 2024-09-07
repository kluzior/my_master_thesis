import cv2
import numpy as np
from packages2.image_processor import ImageProcessor
from packages2.handeye_eye_in_hand import HandEyeCalibration
import logging

class PlaneDeterminator:
    def __init__(self, camera_params_path, handeye_path):
        self.camera_params_path = camera_params_path
        self.image_processor = ImageProcessor(self.camera_params_path)
        self.camera_matrix, self.dist_params = self.image_processor.load_camera_params(self.camera_params_path)
        self.frame = cv2.imread(handeye_path + "/waiting_pos/WAITING_POSE.jpg")
        self.uframe, self.newcameramtx = self.image_processor.undistort_frame(self.frame, self.camera_matrix, self.dist_params)
        self.handeye_path = handeye_path
        self.he = HandEyeCalibration(self.camera_params_path)
        self.get_virtual_plane_points(self.uframe)

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'PlaneDeterminator({self}) was initialized.')

    def get_virtual_plane_points(self, frame, chess_size=(8, 7)):
        rvec, tvec, _ = self.image_processor.calculate_rvec_tvec(frame)
        R, _ = cv2.Rodrigues(rvec)
        self.normal_vector_to_vplane = R[:, 2]
        self.point_on_vplane = tvec.flatten()
        self.rmtx = R
        return self.normal_vector_to_vplane, self.point_on_vplane

    def pixel_to_camera_plane(self, pixel):
        pixel_homog = np.array([pixel[0], pixel[1], 1.0])
        ray_direction = np.linalg.inv(self.camera_matrix) @ pixel_homog
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        camera_origin = np.array([0.0, 0.0, 0.0])
        numerator = np.dot(self.normal_vector_to_vplane, (self.point_on_vplane - camera_origin))
        denominator = np.dot(self.normal_vector_to_vplane, ray_direction)
        if denominator == 0:
            raise ValueError("Promień jest równoległy do płaszczyzny i nie przecina jej.")
        t = numerator / denominator
        intersection_point = camera_origin + t * ray_direction
        return intersection_point

