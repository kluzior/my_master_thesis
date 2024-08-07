from packages2.robot import Robot
from packages2.camera import Camera
import cv2 
import numpy as np
import packages2.common as common
import logging
from packages2.image_processor import ImageProcessor
import pickle

class HandEyeCalibration:
    def __init__(self, robot_client):
        """
        Initializes the HandEyeCalibration object.

        Args:
            robot_client: The client object for communicating with robot.

        Attributes:
            camera: Camera object for capturing images
            robot: Robot object for controlling the robot
            image_processor: ImageProcessor object for framework to process images

            rvec_tvec_pairs: A list to store the rotation and translation vectors for objects seen by camera
            robot_positions: A list to store the robot's TCP positions

            camera_mtx: A list to store calculated object-camera matrices 
            robot_mtx: A list to store calculated TCP-robot base matrices 

            tsai_result: The result of Tsai's hand-eye calibration method
            li_result: The result of Li's hand-eye calibration method

            coordinates_of_chessboard: The defined chess corners for calibration which will be used to collect robot positions

            _logger: logger object for logging  messages
        """
        self.camera = Camera(1)
        self.robot = Robot(robot_client)
        self.image_processor = ImageProcessor()
        
        self.rvec_tvec_pairs = []
        self.robot_positions = []

        self.camera_mtx = []  
        self.robot_mtx = []    

        self.tsai_result = None  
        self.li_result = None

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

    def run(self):
        """
        Main function to run hand-eye calibration process.

        This method captures images of a chessboard pattern using a camera and records the corresponding robot TCP positions.
        Then it performs calibration calculations to determine 4x4 transformation matrix between camera and robot base.

        Returns:
            None
        """
        chess_image = None
        self._logger.info(f'IF YOU ARE READY PRESS "i" TO SAVE THE CHECKERBOARD IMAGE FOR CALIBRATION')
        while self.camera.cap.isOpened():
            img = self.camera.get_frame()
            k = cv2.waitKey(5) & 0xFF
            if k == 27:
                self._logger.info(f'User pressed Esc key - exit program')
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
                    self._logger.info(f'PRESS "s" TO CONFIRM THAT ROBOT IS ON PLACE FOR POINT {index}: {coords}')
                    while True:
                        k2 = cv2.waitKey(5)
                        if k2 == ord('s'):  # wait for 's' key to capture robot TCP position
                            robot_position = self.robot.give_pose()
                            self.robot_positions.append(robot_position)
                            self._logger.info(f'Received & saved pose for point {index}/{len(self.coordinates_of_chessboard)}')
                            break
                cv2.destroyWindow("POINT GUIDANCE")
                self._logger.info("RECEIVED ALL POSES - PRESS ESC")
            cv2.imshow('Camera', img)
        self.camera.cap.release()
        cv2.destroyAllWindows()

        for index, coords in enumerate(self.coordinates_of_chessboard, start=1):
            self._logger.debug(f'aggregated data for point no. {index} ({coords}):')
            self._logger.debug(f'\trobot position: {self.robot_positions[index-1]}')
            self._logger.debug(f'\tcamera rvec: {self.rvec_tvec_pairs[index-1][0]}')
            self._logger.debug(f'\tcamera tvec: {self.rvec_tvec_pairs[index-1][1]}')


    def prepare_object_camera_mtx(self):
        """
        Prepare the rotation and translation matrix for object seen by camera.

        This method combine the rotation and translation 4x4 matrix for the object based on the provided rotation and
        translation vectors. It stores result matrix in the class `camera_mtx` list.

        Returns:
            None
        """
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
        """
        Prepares the rotation and translation 4x4 matrix from robot's TCP position.

        This method calculates the rotation and translation (TCP to robot's base) 4x4 matrix from 
        robot position (X, Y, Z, Roll, Pitch, Yaw). It stores result matrix in the class `robot_mtx` list.

        Returns:
            None
        """
        for index, pos in enumerate(self.robot_positions):            
            pos = list(pos)
            self._logger.debug(f'Calculating rotation&translation matrix for robot position no. {index}')
            matrix = np.zeros((4, 4))

            roll = pos[3]
            pitch = pos[4]
            yaw = pos[5]
            rotation_matrix = common.RPY_to_rmtx(roll, pitch, yaw)
            matrix[:3, :3] = rotation_matrix
            for i in range(3):
                matrix[i, 3] = pos[i]
            matrix[3, 3] = 1
            
            self._logger.info(f'Obtained robot matrix for point no. {index}: \n{matrix}')
            self.robot_mtx.append(matrix)


    def calibrate_tsai(self):
        """
        Performs Tsai's hand-eye calibration algorithm.

        This method calculates the the rotation and translation 4x4 matrix beetwen camera and robot base 
        using the Tsai hand-eye calibration algorithm with the camera and robot matrices
        stored in the `camera_mtx` and `robot_mtx` attributes of the class. Then saves the result to 
        a pickle file named 'tsai_results.pkl'.

        Returns:
            None
        """
        A_list = self.camera_mtx.copy()
        B_list = self.robot_mtx.copy()

        N = len(A_list)
        S = None
        RA_I = None
        T = None
        TA = None
        TB = None

        for i in range(N):
            An = A_list[i]
            Bn = B_list[i]
            RA = An[:3,:3]
            tA = An[:3,3].reshape(3,1)
            tA_ = common.skew2(tA)
            RB = Bn[:3,:3]
            tB = Bn[:3,3].reshape(3,1)
            
            uA, wA = common.R_2_angle_axis(RA)
            uB, wB = common.R_2_angle_axis(RB)
            _S = common.skew2(uA + uB)
            _T = uB - uA 
            _RA_I = RA - np.eye(3)
            _TA = tA
            _TB = tB
            
            S = np.append(S, _S, axis=0) if S is not None else _S
            T = np.append(T, _T, axis=0) if T is not None else _T
            
            RA_I = np.append(RA_I,_RA_I,axis=0) if RA_I is not None else _RA_I
            TA = np.append(TA,_TA,axis=0) if TA is not None else _TA
            TB = np.append(TB,_TB,axis=0) if TB is not None else _TB

        ux = common.solveLS(S,T)    
        uX = 2*ux/(np.sqrt(1+common.norm(ux)**2))
        Rx = (1-common.norm(uX)**2/2)*np.eye(3) + 0.5*(uX*uX.T + np.sqrt(4-common.norm(uX)**2)*common.skew2(uX))
        tX = common.get_Translation(Rx,RA_I,TA,TB)

        self.tsai_result = common.to_mtx(Rx, tX)
        self._logger.info(f"Calculated Tsai result: \n{self.tsai_result}")
        
        with open('tsai_results.pkl', 'wb') as f:
            pickle.dump(self.tsai_result, f)

        self._logger.info(f"Tsai result saved to tsai_results.pkl")

    def calibrate_li(self):
        """
        Performs Li's hand-eye calibration algorithm.

        This method calculates the the rotation and translation 4x4 matrix beetwen camera and robot base 
        using the Li hand-eye calibration algorithm with the camera and robot matrices
        stored in the `camera_mtx` and `robot_mtx` attributes of the class. Then saves the result to 
        a pickle file named 'li_results.pkl'.

        Returns:
            None
        """

        A_list = self.camera_mtx.copy()
        B_list = self.robot_mtx.copy()
        
        N = len(A_list)
        I = np.eye(3)
        I9 = np.eye(9)
        S = None
        T = None

        for i in range(N):
            An = A_list[i]
            Bn = B_list[i]
            RA = An[:3,:3]
            tA = An[:3,3].reshape(3,1)
            tA_ = common.skew2(tA)
            RB = Bn[:3,:3]
            tB = Bn[:3,3].reshape(3,1)
            S1 = np.append(I9 - np.kron(RB, RA), np.zeros((9,3)), axis=1)
            S2 = np.append(np.kron(tB.T, tA_), np.dot(tA_, (I - RA)), axis=1)

            _S = np.append(S1, S2, axis=0)
            _T = np.append(np.zeros((9,1)), tA.reshape(3,1), axis=0)
            
            S = np.append(S, _S, axis=0) if S is not None else _S
            T = np.append(T, _T, axis=0) if T is not None else _T

        Rx_tX = common.solveSVD(S)
        Rx, tX = common.get_RX_tX(Rx_tX)
        Rx = common.getRotation(Rx)

        self.li_result = common.to_mtx(Rx, tX)
        self._logger.info(f"Calculated Li result: \n{self.li_result}")

        with open('li_results.pkl', 'wb') as f:
                pickle.dump(self.li_result, f)

        self._logger.info(f"Li result saved to li_results.pkl")
