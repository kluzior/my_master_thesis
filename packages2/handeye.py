from packages2.robot import Robot
from packages2.camera import Camera
import cv2 
# import socket
import numpy as np
from packages2.common import RPY_to_rmtx
import logging
from packages2.image_processor import ImageProcessor


class HandEyeCalibration:
    def __init__(self, robot_client):
        self.camera = Camera(1)
        self.robot = Robot(robot_client)
        self.image_processor = ImageProcessor()
        
        self.rvec_tvec_pairs = []  # Lista do przechowywania par (rvec, tvec)
        self.robot_positions = []  # Lista do przechowywania pozycji z robota

        self.camera_mtx = []  
        self.robot_mtx = []    

        self.coordinates_of_chessboard=[(0,0),
                                        (7,0),
                                        (0,6),
                                        (7,6),
                                        (4,3),
                                        (2,5),
                                        (1,3),
                                        (0,4)   
                                        ]               # defined chess corners to calibration

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'HandEyeCalibration({self}) was initialized.')

    def calibrate(self):
        pass 


    def test(self):
        frame = self.camera.get_frame() 
        cv2.imshow('frame', frame) 
        cv2.imwrite('test_frame.png', frame)                        
        cv2.waitKey(1000)

    def run(self):
        chess_image = None
        self._logger.info(f'IF YOU ARE READY PRESS "i" TO SAVE THE CHECKERBOARD IMAGE FOR CALIBRATION')
        while self.camera.cap.isOpened():
            img = self.camera.get_frame()
            k = cv2.waitKey(5) & 0xFF
            if k == 27:  # wait for 'Esc' key to exit
                self._logger.info(f'User pressed Esc key')
                break
            elif k == ord('i'):  # wait for 'i' key to capture chessboard image
                chess_image = img
                self._logger.info(f"Photo of chessboard for calibration saved")
                mtx, distortion = self.image_processor.load_camera_params('CameraParams.npz')
                u_chess_image = self.image_processor.undistort_frame(chess_image, mtx, distortion)
                for index, coords in enumerate(self.coordinates_of_chessboard, start=1):
                    rvec, tvec = self.image_processor.calculate_rvec_tvec(u_chess_image, point_shift=coords)
                    self.image_processor.show_chess_corner(u_chess_image, point=coords)
                    self.rvec_tvec_pairs.append((rvec, tvec))

                    self._logger.info(f'PRESS "s" TO CONFIRM THAT ROBOT IS ON PLACE FOR POINT {index}')
                    while True:
                        k2 = cv2.waitKey(5)
                        if k2 == ord('s'):
                            robot_position = self.robot.give_pose()
                            self.robot_positions.append(robot_position)
                            self._logger.info(f'Received & saved pose for point {index}/{len(self.coordinates_of_chessboard)}: {coords}')
                            break
                self._logger.info("RECEIVED ALL POSES - PRESS ESC")
            cv2.imshow('Camera', img)

        self.camera.cap.release()
        cv2.destroyAllWindows()

        for index, coords in enumerate(self.coordinates_of_chessboard, start=1):
            self._logger.debug(f'aggregated data for point no. {index} ({coords}):')
            self._logger.debug(f'\trobot position: {self.robot_positions[index-1]}')
            self._logger.debug(f'\tcamera rvec: {self.rvec_tvec_pairs[index-1][0]}')
            self._logger.debug(f'\tcamera tvec: {self.rvec_tvec_pairs[index-1][1]}')


    def prepare_camera_mtx(self):
        for index, (rvec, tvec) in enumerate(self.rvec_tvec_pairs):
            self._logger.debug(f'Calculating rotation&translation matrix for camera no. {index}')
            mtx = np.zeros((4, 4))
            
            rmtx = cv2.Rodrigues(rvec)[0]
            mtx[:3, :3] = rmtx
            mtx[:3, 3] = tvec.ravel()
            mtx[3, 3] = 1
            
            self._logger.info(f'Obtained camera matrix for point no. {index}: \n{mtx}')
            self.camera_mtx.append(mtx)

    def prepare_robot_mtx(self):
        for index, pos in enumerate(self.robot_positions):            
            pos = list(pos)
            self._logger.debug(f'Calculating rotation&translation matrix for robot position no. {index}')
            matrix = np.zeros((4, 4))

            roll = pos[3]
            pitch = pos[4]
            yaw = pos[5]
            rotation_matrix = RPY_to_rmtx(roll, pitch, yaw)
            matrix[:3, :3] = rotation_matrix
            for i in range(3):
                matrix[i, 3] = pos[i]
            matrix[3, 3] = 1
            
            self._logger.info(f'Obtained robot matrix for point no. {index}: \n{matrix}')
            self.robot_mtx.append(matrix)




