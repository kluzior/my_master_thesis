import cv2
from object_detector import *
import numpy as np
import pandas as pd
import math


# Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

detector = HomogeneousBgDetector()

# Load Image
img = cv2.imread("C:/Users/kluziak/Downloads/measure_object_size/6.png")
cv2.imshow('img', img)
cv2.waitKey(1000)

# Get Aruco marker
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# Draw polygon around the marker
int_corners = np.intp(corners)
cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

cv2.imshow('img2', img)
cv2.waitKey(1000)

# Aruco Perimeter
aruco_perimeter = cv2.arcLength(corners[0], True)
print(aruco_perimeter)
# Pixel to cm ratio
pixel_cm_ratio = aruco_perimeter / 8

contours = detector.detect_objects(img)

cv2.polylines(img, contours, True, (255, 0, 0), 3)

cv2.imshow('img3', img)
cv2.waitKey(1000)

j = 0
# Loop over the contours
for contour in contours:

    # Get the image moments for the contour
    M = cv2.moments(contour)
    
    # Calculate Hu Moments
    hu_M = cv2.HuMoments(M)
   
    # d1 = cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I1,0) - używa hu momemts do porównania dwóch obiektów
    

    # Log scale hu moments 
    for i in range(0,7):
        hu_M[i] = -1* math.copysign(1.0,  hu_M[i]) * math.log10(abs(hu_M[i]))

        df = pd.DataFrame(hu_M,columns=['Hu-moments of original Image']) 
        df

    print(f'{j}: {hu_M}')

    # Calculate the centroid
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Draw a circle to indicate the contour
    cv2.circle(img,(cx,cy), 10, (0,0,255), -1)

    obw = (cv2.arcLength(contour, True)/pixel_cm_ratio)
    area = (cv2.contourArea(contour)/(pixel_cm_ratio**2))

    cv2.putText(img, "Obwod: {} cm".format(round(obw, 1)), (int(cx - 100), int(cy - 30)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.putText(img, "Pole: {} cm2".format(round(area, 1)), (int(cx - 100), int(cy + 30)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    j += 1
# Display the results
cv2.imshow('img4', img)
cv2.imwrite('contour_test/test_photo.png', img)
cv2.waitKey(1000)
