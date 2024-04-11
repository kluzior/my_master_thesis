from packages.store import save_image
from packages.camera import Camera
import cv2
import logging
import numpy as np

class calibrateCamera(Camera):

    def __init__(self, camera_index):
        super().__init__(camera_index)  # Call parent class constructor

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'calibrateCamera({self}) was initialized.')

        self.max_iter = 30
        self.min_accuracy = 0.001
        self.chess_size = (9,6)
        self.frame_size = (640,480)
        self.square_size = 23


    def get_images(self, write_path=''):
        self.images = []

        cap = cv2.VideoCapture(self.index)
        num = 0
        while cap.isOpened():
            _, img = cap.read()
            k = cv2.waitKey(5)
            if k == 27:                                     # wait for 'Esc' key to save
                self._logger.debug(f'user pressed Esc key')
                break
            elif k == ord('s'):                             # wait for 's' key to save
                self.images.append(img)
                save_image(write_path, num, img)   
                self._logger.debug(f'captured photo no. {num}')
                num += 1
            
            cv2.imshow('Camera', img)

        cap.release()   
        cv2.destroyAllWindows()
        self._logger.info(f'captured {num} photos')
        self._logger.debug(f'exit get_images() program')
        

    def calc_camera_params(self, write_path = ''):
        self.camera_matrix = None
        self.distortion_params = None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, self.min_accuracy)
        print(f'criteria: {criteria}')
        objp = np.zeros((self.chess_size[0] * self.chess_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chess_size[0],0:self.chess_size[1]].T.reshape(-1,2)
        objp = objp * self.square_size

        objpoints = [] 
        imgpoints = [] 
        num = 0

        for image in self.images:
            img = image

            cv2.imshow('img', img)
            cv2.waitKey(1000)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            ret, corners = cv2.findChessboardCorners(gray, self.chess_size, None)
            if ret == True:
                objpoints.append(objp) 
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2) 
                cv2.drawChessboardCorners(img, self.chess_size, corners2, ret)             
                save_image(write_path, num, img)   
                logging.debug(f'photo {num} analyzed!')
                num += 1 

                #cv2.waitKey(1000)
        #cv2.destroyAllWindows()

        ret, self.camera_matrix, self.distortion_params, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.frame_size, None, None)

        # Zapis wyników kalibracji do pliku
        np.savez("./CameraParams", cameraMatrix=self.camera_matrix, dist=self.distortion_params)

        self._logger.info('Camera Calibrated!')
        self._logger.debug(f'Calculated camera matrix: {self.camera_matrix}')
        self._logger.debug(f"Calculated distortion parameters: {self.distortion_params}")
        self._logger.debug(f'Calculated rotation vectors: {rvecs}')
        self._logger.debug(f'Calculated translation vectors: {tvecs}')

        # calculate reprojection error
        mean_error = 0 
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], self.camera_matrix, self.distortion_params)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error 
        error_value = mean_error/len(objpoints)
        self._logger.info(f'total reprojection error: {error_value}')

    def undistort_images(self, write_path=''):
    
        num = 0 
        self.undistorted_images = []
        for img in self.images:
            h, w = img.shape[:2] 

            newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_params, (w,h), 1, (w,h))

            dst = cv2.undistort(img, self.camera_matrix, self.distortion_params, None, newCameraMatrix)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w] 
            self.undistorted_images.append(img)
            save_image(write_path, num, img)   

            num += 1 


    def run(self):

        self._logger.debug(f'start capturing calibration photos!')
        
        self.get_images(write_path = 'captured')
        
        # for img in self.images:
        #     cv2.imshow('img', img)
        #     cv2.waitKey(1000)

        self._logger.debug(f'start calculation calibration parameters!')
        self.calc_camera_params(write_path = 'with_chess')

        self.set_calibrated_intrinsics(self.camera_matrix, self.distortion_params)
        
        self._logger.debug(f'start test undistortion!')  

        self.undistort_images(write_path = 'undistorted')                       

