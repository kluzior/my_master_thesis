import cv2
# from object_detector import *
import numpy as np
import pandas as pd
import math
import glob
import time



def detect_objects(frame, mask_name='', min_area=8000):
    # 1) Grayscale Conversion: Convert Image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2) Binarization: Create a Mask with adaptive threshold
    # mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    _, mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    cv2.imshow('mask', mask); cv2.waitKey(500)
    
    if mask_name != '':
        cv2.imwrite(mask_name + time.strftime("%Y%m%d-%H%M%S") + ".png", mask)
 
    # 3) Contour Detection: Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    objects_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            objects_contours.append(contour)
    # 4) Draw Contours: not used here

    return objects_contours


#PLAN:

# 1) Get reference
ref = cv2.imread("./images/captured_9_04/reference/trojkat.png")
cv2.imshow('ref', ref); cv2.waitKey(500)

contour_reference = detect_objects(ref, 'contour_test/reference_mask')
cv2.polylines(ref, contour_reference[0], True, (0, 0, 255), 3)
cv2.imshow('ref', ref); cv2.waitKey(500)
cv2.imwrite("contour_test/reference" + time.strftime("%Y%m%d-%H%M%S") + ".png", ref)
M = cv2.moments(contour_reference[0])
hu_reference = cv2.HuMoments(M)


# 2) Detect objects & choose most relevant
images = glob.glob('./images/captured_9_04/filtered/*.png')
num = 0
for i in images:
    img = cv2.imread(i)

    cv2.imshow('i', img); cv2.waitKey(500)
 
    contours = detect_objects(img)

    j = 0
    # Loop over the contours
    for contour in contours:

        M = cv2.moments(contour)
        
        # Calculate Hu Moments
        hu_M = cv2.HuMoments(M)
    
        retval = cv2.matchShapes(contour, contour_reference[0], cv2.CONTOURS_MATCH_I1, 0)
        print(f'retval: {retval}')
        if  retval < 0.01:
            cv2.polylines(img, contour, True, (0, 0, 255), 3)
        else:
            cv2.polylines(img, contour, True, (255, 0, 0), 3)

        # Calculate the centroid
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.putText(img, "ret: {}".format(round(retval, 5)), (int(cx - 100), int(cy - 30)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        j += 1

    # Display the results
    cv2.imshow('img4', img)
    cv2.imwrite(f'contour_test/test_photo_{num}.png', img)
    cv2.waitKey(1000)
    num += 1