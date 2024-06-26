# from packages.store import save_image
import cv2
import logging
import numpy as np
import os


class calibrateCamera():

    def save_image(self, path, num, img):
        if path != '':
            path = 'C:/Users/kluziak/OneDrive - Nokia/my_master_thesis/test_26_06_2024/images/' + path + '/'
            if not os.path.exists(path):
                os.makedirs(path)
                logging.getLogger('logger').debug(f'new folder was created under the path: {path}')
            if not cv2.imwrite(path + str(num) + '.png', img):
                raise Exception("Could not write image")
        




    def __init__(self):


        self.max_iter = 30
        self.min_accuracy = 0.001
        self.chess_size = (8,7)
        self.frame_size = (640,480)
        self.square_size = 23


    def get_images(self, path=''):
        self.images = []
        # Sprawdź, czy ścieżka jest poprawna
        if not os.path.exists(path):
            print(f"The provided path does not exist: {path}")
            return

        # Przeglądaj pliki w folderze
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img_path = os.path.join(path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    self.images.append(img)
                else:
                    print(f"Failed to load image: {img_path}")
        print(f"Loaded {len(self.images)} images from {path}")
        

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
            img = image.copy()

            cv2.imshow('img', img)
            cv2.waitKey(1000)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            ret, corners = cv2.findChessboardCorners(gray, self.chess_size, None)
            if ret == True:
                objpoints.append(objp) 
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2) 
                cv2.drawChessboardCorners(img, self.chess_size, corners2, ret)             
                self.save_image(write_path, num, img)   
                logging.debug(f'photo {num} analyzed!')
                num += 1 

                #cv2.waitKey(1000)
        #cv2.destroyAllWindows()

        ret, self.camera_matrix, self.distortion_params, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.frame_size, None, None)

        # Zapis wyników kalibracji do pliku
        np.savez("./CameraParams", cameraMatrix=self.camera_matrix, dist=self.distortion_params)

        print('Camera Calibrated!')
        print(f'Calculated camera matrix: {self.camera_matrix}')
        print(f"Calculated distortion parameters: {self.distortion_params}")
        print(f'Calculated rotation vectors: {rvecs}')
        print(f'Calculated translation vectors: {tvecs}')

        # calculate reprojection error
        mean_error = 0 
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], self.camera_matrix, self.distortion_params)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error 
        error_value = mean_error/len(objpoints)
        print(f'total reprojection error: {error_value}')

    def undistort_images(self, write_path=''):
    
        num = 0 
        self.undistorted_images = []
        for image in self.images:
            img = image.copy()
            h, w = img.shape[:2] 

            newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_params, (w,h), 1, (w,h))

            dst = cv2.undistort(img, self.camera_matrix, self.distortion_params, None, newCameraMatrix)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w] 
            self.undistorted_images.append(img)
            self.save_image(write_path, num, img)   

            num += 1 


    def run(self):

        print(f'start capturing calibration photos!')
        
        # self.get_images(write_path = 'captured')
        
        # for img in self.images:
        #     cv2.imshow('img', img)
        #     cv2.waitKey(1000)

        print(f'start calculation calibration parameters!')
        self.calc_camera_params(write_path = 'with_chess')

        # self.set_calibrated_intrinsics(self.camera_matrix, self.distortion_params)
        
        print(f'start test undistortion!')  

        self.undistort_images(write_path = 'undistorted')                       


# Przykład użycia:
camera_calibrator = calibrateCamera()
camera_calibrator.get_images(path='C:/Users/kluziak/OneDrive - Nokia/my_master_thesis/test_26_06_2024/images/chessboard_calib')
camera_calibrator.run()
