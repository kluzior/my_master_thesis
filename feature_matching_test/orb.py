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


img = mask2 #cv.imread('simple.jpg', cv.IMREAD_GRAYSCALE)
image = img.copy()
# Find contours
contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# Find the bounding rectangle for the largest contour (assuming largest white region is the object)
max_contour = max(contours, key=cv.contourArea)
x, y, w, h = cv.boundingRect(max_contour)

# Add a 5-pixel margin around the bounding rectangle
margin = 100
x = max(0, x - margin)
y = max(0, y - margin)
w = min(image.shape[1] - x, w + 2 * margin)
h = min(image.shape[0] - y, h + 2 * margin)

# Crop the image using the expanded bounding rectangle
cropped_image = img[y:y+h, x:x+w]


# Initiate ORB detector
orb = cv.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)
kp2 = orb.detect(cropped_image,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
print(f'ORB | len_kp: {len(kp)}')

kp2, des2 = orb.compute(cropped_image, kp2)
print(f'ORB | len_kp2: {len(kp2)}')

# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv.imshow('aaaa', img2); cv.waitKey(100)

# draw only keypoints location,not size and orientation
img22 = cv.drawKeypoints(cropped_image, kp2, None, color=(0,255,0), flags=0)
cv.imshow('bbbb', img22); cv.waitKey(100)

plt.imshow(img2), plt.show()