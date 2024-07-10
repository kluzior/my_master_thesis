import numpy as np
import cv2 as cv
import glob

from packages.boxes import drawBoxes

# load intrinsic parameters from camera_calibration.py
with np.load('CameraParams.npz') as file:
    mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]
print("\nLoaded camera Matrix: \n", mtx)
print("\nLoaded distortion Parameters: \n", distortion)

# define chessboard params
chessboardSize = (8,7)
size_of_chessboard_squares_mm = 23

# prepare chessboard corners in real world 
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
objp2 = objp.copy()
objp2[:, 0] -= 7
objp2[:, 1] -= 6
print(f" objp2: {objp2}")
objp *= size_of_chessboard_squares_mm
objp2 *= size_of_chessboard_squares_mm

# prepare box coordinates in real world 
axisBoxes = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0], [0,0,-1], [0,1,-1], [1,1,-1], [1,0,-1]])
axisBoxes *= size_of_chessboard_squares_mm

# load undistorted images
images = glob.glob('./images/images_26_06_24/chessboard_calib/0.png')
num = 0
for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find the chessboard corners on image 
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    print(f"len corner: {len(corners)}")
    for corner in corners:
            print(f"corner: {corner}")

    if ret == True:
        # compute rotation & translation vectors from chessboard corners in real world & image
        _, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, distortion)
        print(f"len objp: {len(objp)}")
        print(f"len corners: {len(corners)}")
        print(f"rvecs: {rvecs}")
        print(f"tvecs: {tvecs}")
        _, rvecs2, tvecs2 = cv.solvePnP(objp2, corners, mtx, distortion)
        print(f"len objp2: {len(objp2)}")
        print(f"len corners: {len(corners)}")
        print(f"rvecs2: {rvecs2}")
        print(f"tvecs2: {tvecs2}")

        # compute box coordinates on image 
        imgpts, jac = cv.projectPoints(axisBoxes, rvecs2, tvecs2, mtx, distortion)

        # put boxes on image and save to file
        img = drawBoxes(img, imgpts)
        cv.imshow('img', img)
        cv.imwrite('./images/images_26_06_24/pose_comp_calib/img' + str(num) + '.png', img)
        num += 1
        cv.waitKey(0)

# cv.destroyAllWindows()