import numpy as np
import cv2
import glob
import logging

def calc_camera_params(images, ch_size, f_size, square_size, write_path = '', max_iter=30, min_accuracy=0.001):

    #chessboard_size = (8,6) 
    #frame_size = (640,480) 
    #size_of_chessboard_squares_mm = 23 

    chessboard_size = ch_size
    frame_size = f_size 
    size_of_chessboard_squares_mm = square_size

    #criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, min_accuracy)

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0],0:chessboard_size[1]].T.reshape(-1,2)

    objp = objp * size_of_chessboard_squares_mm

    objpoints = [] 
    imgpoints = [] 

    num = 0

    for image in images:
        img = cv2.imread(image) 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret == True:
            objpoints.append(objp) 
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2) 
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)

            #if ścieżka -> wpisz do pliku
            if write_path != '':
                if not cv2.imwrite(write_path + str(num) + '.png', img):
                    raise Exception("Could not write image")     
                       
            #cv2.imshow('img', img)
            #cv2.imwrite('./image_banks/01_with_chess/img' + str(num) + '.png', img)
            logging.getLogger('logger').debug(f'photo {num} analyzed!')
            num += 1 

            #cv2.waitKey(1000)
    #cv2.destroyAllWindows()

    ret, cameraMatrix, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, frame_size, None, None)

    #print("Camera Calibrated: ", ret)
    #print("\nCamera Matrix: \n", cameraMatrix)
    #print("\nDistortion Parameters: \n", dist)
    #print("\nRotation Vectors: \n", rvecs)
    #print("\nTranslation Vectors: \n", tvecs)

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
    #print( "total error: {}".format(error_value) )
    logging.getLogger('logger').info(f'total reprojection error: {error_value}')

    return cameraMatrix, dist


def undistort_images(images, camera_matrix, dist):
   
    num = 0 
    undistorted = []
    for image in images:
        img = cv2.imread(image) 
        h, w = img.shape[:2] 

        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w,h), 1, (w,h))


        dst = cv2.undistort(img, camera_matrix, dist, None, newCameraMatrix)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w] 
        undistorted.append(img)
        #cv2.imwrite('./image_banks/02_undistorted/img' + str(num) + '.png', img)
        num += 1 

    return undistorted


