import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('C:/Users/kluziak/OneDrive - Nokia/my_master_thesis/feature_matching_test/0.png',cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('C:/Users/kluziak/OneDrive - Nokia/my_master_thesis/feature_matching_test/ref3.png',cv.IMREAD_GRAYSCALE) # trainImage
_, mask1 = cv.threshold(img1, 230, 255, cv.THRESH_BINARY)
_, mask2 = cv.threshold(img2, 230, 255, cv.THRESH_BINARY)
cv.imshow('img2', img2); cv.waitKey(1000)
cv.imshow('test1', mask1); cv.waitKey(100)
cv.imshow('test2', mask2); cv.waitKey(100)


img = mask2 #cv.imread('blox.jpg', cv.IMREAD_GRAYSCALE) # `<opencv_root>/samples/data/blox.jpg`

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()

# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

cv.imshow('fast_true.png', img2); cv.waitKey(100)

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)

print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

img3 = cv.drawKeypoints(img, kp, None, color=(255,0,0))

cv.imshow('fast_false.png', img3); cv.waitKey(1000)
