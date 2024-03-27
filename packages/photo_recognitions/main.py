from .get_images import get_images
from .calc_camera_params import calc_camera_params
from .undistort_images import undistort_images

import cv2
import logging

def calibrate_camera():

    logging.getLogger('logger').debug(f'start capturing calibration photos!')
    captured_images = get_images(write_path = 'captured')
    
    for img in captured_images:
        cv2.imshow('img', img)
        cv2.waitKey(1000)

    logging.getLogger('logger').debug(f'start calculation calibration parameters!')
    # camera_matrix, distortion_params = calc_camera_params(  
    #                                     images = captured_images, 
    #                                     chess_size = (8,6), 
    #                                     frame_size = (640,480), 
    #                                     square_size = 23, 
    #                                     write_path = 'with_chess'
    #                                     )
    
    logging.getLogger('logger').debug(f'start test undistortion!')                          
    # undistorted_images = undistort_images(
    #                         images = captured_images, 
    #                         camera_matrix = camera_matrix, 
    #                         dist = distortion_params, 
    #                         write_path = 'with_chess'
    #                         )

    #return undistorted_images
