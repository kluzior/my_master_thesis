import cv2
# from object_detector import *
import numpy as np
import pandas as pd
import math
import glob

def detect_objects(frame, min_area=8000):
    # Convert Image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create a Mask with adaptive threshold
    # mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    _, mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    cv2.imshow('mask', mask); cv2.waitKey(500)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    objects_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    return objects_contours

def detect_aruco(frame, aruco_edge_cm):
    aruco_corners, _, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

    # Draw polygon around the marker
    cv2.polylines(frame, np.intp(aruco_corners), True, (0, 255, 0), 5)

    # cv2.imshow('img2', frame); cv2.waitKey(500)

    # Aruco Perimeter - pixel to cm ratio
    pixel_cm_ratio = cv2.arcLength(aruco_corners[0], True) / (4*aruco_edge_cm)

    return pixel_cm_ratio

def get_hu_reference():
    ref = cv2.imread("./images/captured_9_04/reference/kwadrat.png")
    cv2.imshow('ref', ref); cv2.waitKey(1000)

    contour_reference = detect_objects(ref)
    # print(f'contour_reference: {contour_reference}')
    # cv2.imshow('contour_reference', contour_reference); cv2.waitKey(1000)
    cv2.polylines(ref, contour_reference[0], True, (0, 0, 255), 3)
    cv2.imshow('ref', ref); cv2.waitKey(1000)

    M = cv2.moments(contour_reference[0])
    hu_reference = cv2.HuMoments(M)
    return contour_reference[0], hu_reference

def undistort_frame(frame, mtx, distortion):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distortion, (w,h), 1, (w,h))
    undst = cv2.undistort(img, mtx, distortion, None, newcameramtx)
    # x, y, w, h = roi                  # disable roi cut to not lose object detection
    # undst = undst[y:y+h, x:x+w]
    return undst

cont_ref, hu_ref = get_hu_reference()


# Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

# read camera intrinsic params
with np.load('CameraParams.npz') as file:
    mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]
print("\nLoaded camera Matrix: \n", mtx)
print("\nLoaded distortion Parameters: \n", distortion)

# Load Images
images = glob.glob('./images/captured_9_04/filtered/*.png')
num = 0
for i in images:
    img = cv2.imread(i)

    cv2.imshow('i', img); cv2.waitKey(1000)

    # undistort image
    uimg = undistort_frame(img, mtx, distortion)
    
    # Get Aruco marker
    pixel_cm_ratio = detect_aruco(uimg, 2)


    contours = detect_objects(uimg)

    # cv2.polylines(img, contours, True, (255, 0, 0), 3)

    cv2.imshow('img3', uimg); cv2.waitKey(1000)

    # hu ref
    # cont_ref, hu_ref = get_hu_reference()

    j = 0
    # Loop over the contours
    for contour in contours:
        # print(f'contour: {contour}')
        # Get the image moments for the contour
        # epsilon = 0.1*cv2.arcLength(contour,True)
        # print(f'epsilon: {epsilon}')
        # contour = cv2.approxPolyDP(contour,epsilon,True)

        M = cv2.moments(contour)
        
        # Calculate Hu Moments
        hu_M = cv2.HuMoments(M)
    
        retval = cv2.matchShapes(contour,cont_ref,cv2.CONTOURS_MATCH_I2,0)
        print(f'retval: {retval}')
        if  retval < 0.001:
            cv2.polylines(uimg, contour, True, (0, 0, 255), 3)
            print("zgoda!")
        else:
            cv2.polylines(uimg, contour, True, (255, 0, 0), 3)
            pass

        cv2.imshow('in_for', uimg); cv2.waitKey(1000)


        # d1 = cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I1,0) - używa hu momemts do porównania dwóch obiektów


        # Log scale hu moments 
        for i in range(0,7):
            hu_M[i] = -1* math.copysign(1.0,  hu_M[i]) * math.log10(abs(hu_M[i]))


        # print(f'{j}: {hu_M}')

        # Calculate the centroid
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Draw a circle to indicate the contour
        cv2.circle(img,(cx,cy), 10, (0,0,255), -1)

        obw = (cv2.arcLength(contour, True)/pixel_cm_ratio)
        area = (cv2.contourArea(contour)/(pixel_cm_ratio**2))

        cv2.putText(uimg, "Obwod: {} cm".format(round(obw, 1)), (int(cx - 100), int(cy - 30)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        cv2.putText(uimg, "Pole: {} cm2".format(round(area, 1)), (int(cx - 100), int(cy + 30)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
        j += 1

    # Display the results
    cv2.imshow('img4', uimg)
    cv2.imwrite(f'contour_test/test_photo_{num}.png', uimg)
    cv2.waitKey(1000)
    num += 1