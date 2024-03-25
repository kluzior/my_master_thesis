if __name__ != "__main__":
    from .get_images import get_images
    from .calc_camera_params import calc_camera_params
    from .undistort_images import undistort_images

import cv2

def calibrate_camera():
    from .get_images import get_images

    captured_images = get_images(write_path = 'captured')
    
    for img in captured_images:
        cv2.imshow('img', img)
        cv2.waitKey(1000)

    # camera_matrix, distortion_params = calc_camera_params(  
    #                                     images = captured_images, 
    #                                     chess_size = (8,6), 
    #                                     frame_size = (640,480), 
    #                                     square_size = 23, 
    #                                     write_path = 'with_chess'
    #                                     )
                                    
    # undistorted_images = undistort_images(
    #                         images = captured_images, 
    #                         camera_matrix = camera_matrix, 
    #                         dist = distortion_params, 
    #                         write_path = 'with_chess'
    #                         )

    #return undistorted_images


if __name__ == "__main__":
    from get_images import get_images
    from camera_calibration import calc_camera_params, undistort_images

    calibrate_camera()