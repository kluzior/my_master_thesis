import webcolors
import cv2
import numpy as np

# define function to draw box
def drawBoxes(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor
    img = cv2.drawContours(image         = img, 
                          contours      = [imgpts[:4]], 
                          contourIdx    = -1, 
                          color         = webcolors.name_to_rgb("green"), 
                          thickness     = -1)

    # draw pillars
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img       = img, 
                      pt1       = tuple(imgpts[i]), 
                      pt2       = tuple(imgpts[j]), 
                      color     = webcolors.name_to_rgb("blue"), 
                      thickness = 3)

    # draw top floor
    img = cv2.drawContours(image         = img, 
                          contours      = [imgpts[4:]], 
                          contourIdx    = -1, 
                          color         = webcolors.name_to_rgb("red"), 
                          thickness     = 3)
    return img