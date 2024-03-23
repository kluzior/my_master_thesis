if __name__ != "__main__":
    from photo_recognitions.functions.get_images import get_images
    from photo_recognitions.functions.camera_calibration import calc_camera_params, undistort_images

import cv2

def calibrate_camera():

    captured_images = get_images()
    for img in captured_images:
        cv2.imshow('img', img)
        cv2.waitKey(1000)

    #camera_matrix, distortion_params = calc_camera_params(captured_images, (8,6), (640,480), 23)
    
    #undistorted_images = undistort_images(captured_images, camera_matrix, distortion_params)



    #return undistorted_images


if __name__ == "__main__":
    from functions.get_images import get_images
    from functions.camera_calibration import calc_camera_params, undistort_images

    calibrate_camera()