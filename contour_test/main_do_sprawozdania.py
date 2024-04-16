import cv2
import numpy as np
import math
import glob
from statistics import mean
import time

def undistort_frame(frame, mtx, distortion):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distortion, (w,h), 1, (w,h))
    undst = cv2.undistort(frame, mtx, distortion, None, newcameramtx)
    # x, y, w, h = roi                  # disable roi cut to not lose object detection
    # undst = undst[y:y+h, x:x+w]
    return undst

def detect_objects(frame, ref_name, min_area=8000, save_mask=False, binarization='standard'):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if binarization == 'standard':          # 1) standard binarization
        _, mask = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

    if binarization == 'adaptive':          # 2) binarization with adaptive threshold
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)

    if binarization == 'canny':             # 3) test of canny edge detection
        blur = cv2.GaussianBlur(gray.copy(),(5,5),0); mask = cv2.Canny(blur, 50, 150)

    if binarization == 'otsu':
        blur = cv2.GaussianBlur(gray,(5,5),0); _, mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('mask', mask); cv2.waitKey(100)
    if save_mask:
        cv2.imwrite(f'contour_test/img/detection/mask_' + ref_name + '.png', mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    objects_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    return objects_contours


def detect_aruco(frame, aruco_edge_cm):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

    aruco_corners, _, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=cv2.aruco.DetectorParameters())

    cv2.polylines(frame, np.intp(aruco_corners), True, (0, 255, 0), 5)      # mark detected aruco

    pixel_cm_ratio = cv2.arcLength(aruco_corners[0], True) / (4*aruco_edge_cm)

    return pixel_cm_ratio


def get_hu_reference(ref_name):
    ref = cv2.imread("./images/ALL_CONTOURS/reference/" + ref_name + ".png")

    contour_reference = detect_objects(ref, ref_name, save_mask=True)
    M = cv2.moments(contour_reference[0])
    hu_reference = cv2.HuMoments(M)

    return contour_reference[0], hu_reference


def run_contour_detection(reference_name, binarization_type):
    # read previous camera intrinsic params
    with np.load('CameraParams.npz') as file:
        mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]

    ### get reference's contour ###
    cont_ref, hu_ref = get_hu_reference(reference_name) 

    images = glob.glob('./images/ALL_CONTOURS/*.png')
    num = 0
    for i in images:
        img = cv2.imread(i)

        ### undistort image ###
        uimg = undistort_frame(img, mtx, distortion)

        ### find contour ###
        contours = detect_objects(uimg, reference_name, binarization=binarization_type)

        ### get Aruco marker ###
        pixel_cm_ratio = detect_aruco(uimg, 2.2)

        img_sizes = uimg.copy()
        img_detection = uimg.copy()
        hu_moments_list = []
        for contour in contours:
            ###################### part of sizing objects ######################
            M = cv2.moments(contour)
            hu_M = cv2.HuMoments(M)
            for i in range(6):          # convert hu moments to log scale 
                hu_M[i] = -1* math.copysign(1.0,  hu_M[i]) * math.log10(abs(hu_M[i]))

            hu_moments_list.append(hu_M)

            # calculate centroid of contour
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            cv2.circle(img_sizes,(cx,cy), 10, (0,0,255), -1)

            obw  = (cv2.arcLength(contour, True)/pixel_cm_ratio)
            area = (cv2.contourArea(contour)/(pixel_cm_ratio**2))

            cv2.putText(img_sizes, "Obwod: {} cm".format(round(obw, 1)), (int(cx - 100), int(cy - 40)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 2)
            cv2.putText(img_sizes, "Pole: {} cm2".format(round(area, 1)), (int(cx - 100), int(cy - 20)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 2)

            
            ###################### part of shape matching objects ######################
            retval = []
            retval1 = cv2.matchShapes(contour, cont_ref, cv2.CONTOURS_MATCH_I1, 0);             retval.append(retval1)
            retval2 = cv2.matchShapes(contour, cont_ref, cv2.CONTOURS_MATCH_I2, 0);             retval.append(retval2)
            retval3 = cv2.matchShapes(contour, cont_ref, cv2.CONTOURS_MATCH_I3, 0);             retval.append(retval3)

            if  mean(retval) < 0.004:
                cv2.polylines(img_detection, contour, True, (0, 0, 255), 3)
            else:
                cv2.polylines(img_detection, contour, True, (255, 0, 0), 3)

            cv2.putText(img_detection, "I1: {}".format(round(retval1, 5)), (int(cx - 100), int(cy - 60)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 2)
            cv2.putText(img_detection, "I2: {}".format(round(retval2, 5)), (int(cx - 100), int(cy - 40)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 2)
            cv2.putText(img_detection, "I3: {}".format(round(retval3, 5)), (int(cx - 100), int(cy - 20)), cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 2)
            cv2.putText(img_detection, "shape: {}".format(reference_name), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
            cv2.putText(img_detection, "binarization: {}".format(binarization_type), (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)


        # save the results
        cv2.imwrite(f'contour_test/img/detection/{num}_{reference_name}_{binarization_type}.png', img_detection)
        if binarization_type == 'standard':
            cv2.imwrite(f'contour_test/img/sizing/{num}.png', img_sizes)
        cv2.imshow('in_for', img_detection); cv2.waitKey(100)
        num += 1

if __name__ == "__main__":

    reference_names = ['kwadrat', 'szesciokat', 'wielokat', 'trojkat', 'okrag']
    binarization_types = ['standard', 'adaptive', 'otsu', 'canny']

    for reference_name in reference_names:
        for binarization_type in binarization_types:
            run_contour_detection(reference_name, binarization_type)