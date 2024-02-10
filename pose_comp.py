import numpy as np
import cv2 as cv
import glob
import webcolors

# define function to draw box
def drawBoxes(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor
    img = cv.drawContours(image         = img, 
                          contours      = [imgpts[:4]], 
                          contourIdx    = -1, 
                          color         = webcolors.name_to_rgb("green"), 
                          thickness     = -1)

    # draw pillars
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img       = img, 
                      pt1       = tuple(imgpts[i]), 
                      pt2       = tuple(imgpts[j]), 
                      color     = webcolors.name_to_rgb("blue"), 
                      thickness = 3)

    # draw top floor
    img = cv.drawContours(image         = img, 
                          contours      = [imgpts[4:]], 
                          contourIdx    = -1, 
                          color         = webcolors.name_to_rgb("red"), 
                          thickness     = 3)
    return img

# load intrinsic parameters from camera_calibration.py
with np.load('CameraParams.npz') as file:
    mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]
print("\nLoaded camera Matrix: \n", mtx)
print("\nLoaded distortion Parameters: \n", distortion)

# define chessboard params
chessboardSize = (8,6)
size_of_chessboard_squares_mm = 23

# prepare chessboard corners in real world 
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)
objp *= size_of_chessboard_squares_mm

# prepare box coordinates in real world 
axisBoxes = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0], [0,0,-1], [0,1,-1], [1,1,-1], [1,0,-1]])
axisBoxes *= size_of_chessboard_squares_mm

# load undistorted images
images = glob.glob('./image_banks/02_undistorted/*.png')

num = 0
for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # find the chessboard corners on image 
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    if ret == True:
        # compute rotation & translation vectors from chessboard corners in real world & image
        _, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, distortion)

        # compute box coordinates on image 
        imgpts, jac = cv.projectPoints(axisBoxes, rvecs, tvecs, mtx, distortion)

        # put boxes on image and save to file
        img = drawBoxes(img, imgpts)
        cv.imshow('img', img)
        cv.imwrite('./image_banks/03_posed/img' + str(num) + '.png', img)
        num += 1
        cv.waitKey(1000)

cv.destroyAllWindows()