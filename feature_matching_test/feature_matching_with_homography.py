import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def my_sift(ref_img, img):
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(ref_img, None)
    kp2, des2 = sift.detectAndCompute(img, None)
    img222 = cv.drawKeypoints(ref_img, kp1, None, color=(0,255,0), flags=0)
    cv.imshow('SIFT | img_ref', img222); cv.waitKey(1000)
    print(f'SIFT | len_kp1: {len(kp1)}')
    print(f'SIFT | len_kp2: {len(kp2)}')

    return kp1, kp2, des1, des2

def my_orb(ref_img, img):
    # Initiate ORB detector
    orb = cv.ORB_create()

    # find the keypoints with ORB
    kp1 = orb.detect(ref_img,None)
    kp2 = orb.detect(img,None)

    # compute the descriptors with ORB
    kp1, des1 = orb.compute(ref_img, kp1)
    kp2, des2 = orb.compute(img, kp2)
    print(f'ORB | len_kp1: {len(kp1)}')
    print(f'ORB | len_kp2: {len(kp2)}')
    return kp1, kp2, des1, des2


MIN_MATCH_COUNT = 5

img01 = cv.imread('C:/Users/kluziak/OneDrive - Nokia/my_master_thesis/feature_matching_test/ref3.png',cv.IMREAD_GRAYSCALE) # trainImage
img02 = cv.imread('C:/Users/kluziak/OneDrive - Nokia/my_master_thesis/feature_matching_test/0.png',cv.IMREAD_GRAYSCALE) # queryImage
_, mask1 = cv.threshold(img01, 230, 255, cv.THRESH_BINARY)
_, mask2 = cv.threshold(img02, 230, 255, cv.THRESH_BINARY)
cv.imshow('img2', img02); cv.waitKey(1000)
cv.imshow('test1', mask1); cv.waitKey(100)
cv.imshow('test2', mask2); cv.waitKey(100)

img1 = mask1
img2 = mask2

# Load the binarized image
image = img1.copy()

# Find contours
contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# print(f'contour: {contours}')
# Find the bounding rectangle for the largest contour (assuming largest white region is the object)
max_contour = max(contours, key=cv.contourArea)
print(f'max_contour: {max_contour}')

x, y, w, h = cv.boundingRect(max_contour)
print(f'x: {x} y: {y} w: {w} h: {h}')

# Add a 5-pixel margin around the bounding rectangle
margin = 100
x = max(0, x - margin)
y = max(0, y - margin)
w = min(image.shape[1] - x, w + 2 * margin)
h = min(image.shape[0] - y, h + 2 * margin)

# Crop the image using the bounding rectangle
cropped_image = image[y:y+h, x:x+w]

# Save or display the cropped image
cv.imwrite('cropped_image.png', cropped_image)
cv.imshow('Cropped Image', cropped_image)
cv.waitKey(100)

# find the keypoints and descriptors with SIFT
kp1, kp2, des1, des2 = my_sift(cropped_image, img2) #sift.detectAndCompute(cropped_image,None)
kp11, kp22, des11, des22 = my_orb(cropped_image, img2) #sift.detectAndCompute(cropped_image,None)

print(f'SIFT | des1: {des1}')
print(f'ORB | des11: {des11}')
print(f'SIFT | des1: {len(des1[0])}')
print(f'SIFT | des11: {len(des11[0])}')

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    # h,w = img1.shape
    h,w = cropped_image.shape

    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)

    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    print( "Matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )

else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

img3 = cv.drawMatches(cropped_image,kp1,img2,kp2,good,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()