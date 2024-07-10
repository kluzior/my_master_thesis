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
        self._logger.info(f'')
        print(f'JEZELI JESTEŚ GOTOWY WCISNIJ I FOR IMAGE')
        while self.camera.cap.isOpened():
            img = self.camera.get_frame()
            k = cv2.waitKey(5) & 0xFF
            if k == 27:  # wait for 'Esc' key to exit
                self._logger.info(f'user pressed Esc key')
                break
            elif k == ord('i'):  # wait for 'i' key to capture chessboard image
                chess_image = img
                print(f"zapisano zdjęcie szachownicy")
                cv2.imshow("chess_image", chess_image)
                mtx, distortion = self.image_processor.load_camera_params('CameraParams.npz')
                u_chess_image = self.image_processor.undistort_frame(chess_image, mtx, distortion)
                cv2.imshow("u_chess_image", u_chess_image)
                for index, coords in enumerate(self.coordinates_of_chessboard, start=1):
                    rvec, tvec = self.image_processor.calculate_rvec_tvec(u_chess_image, point_shift=coords)
                    self.image_processor.show_chess_corner(u_chess_image, point=coords)
                    self.rvec_tvec_pairs.append((rvec, tvec))  # Dodaj parę (rvec, tvec) do listy

                    # Wait here until 's' is pressed
                    print(f'Press "s" to confirm robot on place for point {index}')
                    while True:
                        k2 = cv2.waitKey(5)
                        if k2 == ord('s'):
                            print(f"Received pose for point {index}/{len(self.coordinates_of_chessboard)}: {coords}")
                            robot_position = self.robot.give_pose()
                            self.robot_positions.append(robot_position)  # Dodaj pozycję robota do listy
                            break
                print("Received all poses, press Esc")
            cv2.imshow('Camera', img)

        self.camera.cap.release()
        cv2.destroyAllWindows()


    def show_collected_data(self):
        for index, coords in enumerate(self.coordinates_of_chessboard, start=1):
            print(f'Point {index}: {coords}')
            print(f'\tRobot position: {self.robot_positions[index-1]}')
            print(f'\tRvec: {self.rvec_tvec_pairs[index-1][0]}')    
            print(f'\tTvec: {self.rvec_tvec_pairs[index-1][1]}')

    

    def prepare_camera_mtx(self):
        for index, (rvec, tvec) in enumerate(self.rvec_tvec_pairs):
            rmtx = cv2.Rodrigues(rvec)[0]
            mtx = np.zeros((4, 4))
            mtx[:3, :3] = rmtx
            mtx[:3, 3] = tvec.ravel()
            mtx[3, 3] = 1
            self.camera_mtx.append(mtx)
            self._logger.info(f'Obtained camera matrix for point no. {index}: \n{mtx}')

    def prepare_robot_mtx(self):
        for index, pos in enumerate(self.robot_positions):            
            print(f'pos: {pos}')
            pos = list(pos)
            print(f'pos: {pos}')
            self._logger.info(f'Calculating rotation&translation matrix for robot position: {pos}')
            
            roll = pos[3]
            pitch = pos[4]
            yaw = pos[5]
            matrix = np.zeros((4, 4))

            # Get the 3x3 rotation matrix
            rotation_matrix = RPY_to_rmtx(roll, pitch, yaw)

            matrix[:3, :3] = rotation_matrix
            for i in range(3):
                matrix[i, 3] = pos[i]
            matrix[3, 3] = 1
            self._logger.info(f'Obtained robot matrix for point no. {index}: \n{matrix}')
            self.robot_mtx.append(matrix)




