from packages.store import save_image

import cv2
import logging
import numpy as np

class calibrateCamera:

    def __init__(self) -> None:
        self.max_iter = 30, 
        self.min_accuracy = 0.001
        self.chess_size = (8,6)
        self.frame_size = (640,480)
        self.square_size = 23


    def get_images(self, write_path=''):
        self.images = []

        cap = cv2.VideoCapture(0)
        num = 0
        while cap.isOpened():
            _, img = cap.read()
            k = cv2.waitKey(5)
            if k == 27:                                     # wait for 'Esc' key to save
                logging.getLogger('logger').debug(f'user pressed Esc key')
                break
            elif k == ord('s'):                             # wait for 's' key to save
                self.images.append(img)
                save_image(write_path, num, img)   
                logging.getLogger('logger').debug(f'captured photo no. {num}')
                num += 1
            
            cv2.imshow('Camera', img)

        cap.release()   
        cv2.destroyAllWindows()
        logging.getLogger('logger').info(f'captured {num} photos')
        logging.getLogger('logger').debug(f'exit get_images() program')
        

    def calc_camera_params(self, write_path = ''):
        self.camera_matrix = 0
        self.distortion_params = 0

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, self.max_iter, self.min_accuracy)

        objp = np.zeros((self.chess_size[0] * self.chess_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:self.chess_size[0],0:self.chess_size[1]].T.reshape(-1,2)
        objp = objp * self.square_size

        objpoints = [] 
        imgpoints = [] 
        num = 0

        for image in self.images:
            img = cv2.imread(image) 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            ret, corners = cv2.findChessboardCorners(gray, self.chess_size, None)
            if ret == True:
                objpoints.append(objp) 
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2) 
                cv2.drawChessboardCorners(img, self.chess_size, corners2, ret)             
                save_image(write_path, num, img)   
                logging.getLogger('logger').debug(f'photo {num} analyzed!')
                num += 1 

                #cv2.waitKey(1000)
        #cv2.destroyAllWindows()

        ret, self.cameraMatrix, self.distortion_params, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.frame_size, None, None)

        logging.getLogger('logger').info('Camera Calibrated!')
        logging.getLogger('logger').debug(f'Calculated camera matrix: {self.cameraMatrix}')
        logging.getLogger('logger').debug(f"Calculated distortion parameters: {self.distortion_params}")
        logging.getLogger('logger').debug(f'Calculated rotation vectors: {rvecs}')
        logging.getLogger('logger').debug(f'Calculated translation vectors: {tvecs}')

        # calculate reprojection error
        mean_error = 0 
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], self.cameraMatrix, self.dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error 
        error_value = mean_error/len(objpoints)
        logging.getLogger('logger').info(f'total reprojection error: {error_value}')

    def undistort_images(self, write_path=''):
    
        num = 0 
        self.undistorted_images = []
        for image in self.images:
            img = cv2.imread(image) 
            h, w = img.shape[:2] 

            newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_params, (w,h), 1, (w,h))

            dst = cv2.undistort(img, self.camera_matrix, self.distortion_params, None, newCameraMatrix)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w] 
            self.undistorted_images.append(img)
            save_image(write_path, num, img)   

            num += 1 


    def run(self):

        logging.getLogger('logger').debug(f'start capturing calibration photos!')
        
        self.get_images(write_path = 'captured')
        
        # for img in self.images:
        #     cv2.imshow('img', img)
        #     cv2.waitKey(1000)

        logging.getLogger('logger').debug(f'start calculation calibration parameters!')
        # self.calc_camera_params( write_path = 'with_chess')

        
        logging.getLogger('logger').debug(f'start test undistortion!')                          
        # undistorted_images = undistort_images(
        #                         images = captured_images, 
        #                         camera_matrix = camera_matrix, 
        #                         dist = distortion_params, 
        #                         write_path = 'with_chess'
        #                         )

        #return undistorted_images
