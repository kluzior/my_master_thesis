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

img1 = mask1 
img2 = mask2

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50) # or pass empty dictionary

flann = cv.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
    singlePointColor = (255,0,0),
    matchesMask = matchesMask,
    flags = cv.DrawMatchesFlags_DEFAULT)

img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()