import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread('C:/Users/kluziak/OneDrive - Nokia/my_master_thesis/feature_matching_test/0.png',cv.IMREAD_GRAYSCALE) # queryImage
img2 = cv.imread('C:/Users/kluziak/OneDrive - Nokia/my_master_thesis/feature_matching_test/ref1.png',cv.IMREAD_GRAYSCALE) # trainImage
_, mask1 = cv.threshold(img1, 230, 255, cv.THRESH_BINARY)
_, mask2 = cv.threshold(img2, 230, 255, cv.THRESH_BINARY)
cv.imshow('img2', img2); cv.waitKey(1000)
cv.imshow('test1', mask1); cv.waitKey(100)
cv.imshow('test2', mask2); cv.waitKey(100)



img = mask2
gray= img #cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp, des = sift.detectAndCompute(gray,None)
print(f'sift: {sift}')
print(f'kp: {kp}')
print(f'len_kp: {len(kp)}')
print(f'des: {des}')

img=cv.drawKeypoints(gray,kp,img)

cv.imshow('sift_keypoints', img); cv.waitKey(1000)
img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('sift_keypoints2', img); cv.waitKey(1000)

# cv.imwrite('sift_keypoints.jpg',img)