import numpy as np
import cv2
import glob
import logging
from packages.store import save_image

def calc_camera_params(images, chess_size, frame_size, square_size, write_path = '', max_iter=30, min_accuracy=0.001):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, min_accuracy)

    objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chess_size[0],0:chess_size[1]].T.reshape(-1,2)

    objp = objp * square_size

    objpoints = [] 
    imgpoints = [] 

    num = 0

    for image in images:
        img = cv2.imread(image) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        ret, corners = cv2.findChessboardCorners(gray, chess_size, None)
        if ret == True:
            objpoints.append(objp) 
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2) 
            cv2.drawChessboardCorners(img, chess_size, corners2, ret)
                      
            #cv2.imshow('img', img)
            save_image(write_path, num, img)   
            
            logging.getLogger('logger').debug(f'photo {num} analyzed!')
            num += 1 

            #cv2.waitKey(1000)
    #cv2.destroyAllWindows()

    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    logging.getLogger('logger').info('Camera Calibrated!')
    logging.getLogger('logger').debug(f'Calculated camera matrix: {cameraMatrix}')
    logging.getLogger('logger').debug(f"Calculated distortion parameters: {dist}")
    logging.getLogger('logger').debug(f'Calculated rotation vectors: {rvecs}')
    logging.getLogger('logger').debug(f'Calculated translation vectors: {tvecs}')

        # czy zapisywać parametry / może if?
    #np.savez("./CameraParams", cameraMatrix=cameraMatrix, dist=dist)


    # calculate reprojection error
    mean_error = 0 
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error 
    error_value = mean_error/len(objpoints)

    logging.getLogger('logger').info(f'total reprojection error: {error_value}')

    return cameraMatrix, dist





