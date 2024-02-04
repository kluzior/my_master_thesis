import numpy as np
import cv2 as cv
import glob
import webcolors

def drawBoxes(img, corners, imgpts):

    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, webcolors.name_to_rgb("green"), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), webcolors.name_to_rgb("blue"), 3)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, webcolors.name_to_rgb("red"), 3)

    return img

################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

chessboardSize = (8,6)
frameSize = (640,480)

with np.load('CameraParams.npz') as file:
    mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]

print("\nLoaded camera Matrix: \n", mtx)
print("\nLoaded distortion Parameters: \n", distortion)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 23
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
print("\naxis: \n", axis)
axisBoxes = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3], [0,3,-3], [3,3,-3], [3,0,-3]])
axisBoxes = axisBoxes *10
print("\naxisBoxes: \n", axisBoxes)

images = glob.glob('my_master_thesis/image_banks/02_undistorted/*.png')
print(len(images))

num = 0

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        corners2_pnp = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

        ret_pnp, rvecs_pnp, tvecs_pnp = cv.solvePnP(objp, corners, mtx, distortion)

        imgpts, jac = cv.projectPoints(axisBoxes, rvecs_pnp, tvecs_pnp, mtx, distortion)

        img = drawBoxes(img, corners2_pnp, imgpts)
        cv.imshow('img', img)
        cv.imwrite('my_master_thesis/image_banks/03_posed/img' + str(num) + '.png', img)
        num += 1

        cv.waitKey(1000)


cv.destroyAllWindows()