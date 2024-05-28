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
    img222 = cv.drawKeypoints(ref_img, kp1, None, color=(0,255,0), flags=0)
    cv.imshow('ORB | img_ref', img222); cv.waitKey(1000)
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
cropped_image = image[y:y+h, x:x+w]


# Save or display the cropped image
cv.imwrite('cropped_image.png', cropped_image)
cv.imshow('Cropped Image', cropped_image)
cv.waitKey(100)

# Find the keypoints and descriptors with SIFT and ORB
kp1_sift, kp2_sift, des1_sift, des2_sift = my_sift(cropped_image, img2)
kp1_orb, kp2_orb, des1_orb, des2_orb = my_orb(cropped_image, img2)

print(f'SIFT | des1: {des1_sift}')
print(f'ORB | des1: {des1_orb}')
print(f'SIFT | des1 length: {len(des1_sift[0])}')
print(f'ORB | des1 length: {len(des1_orb[0])}')

# FLANN based Matcher for SIFT descriptors
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matches_sift = flann.knnMatch(des1_sift, des2_sift, k=2)

# store all the good matches as per Lowe's ratio test for SIFT
good_sift = []
for m, n in matches_sift:
    if m.distance < 0.7 * n.distance:
        good_sift.append(m)

if len(good_sift) > MIN_MATCH_COUNT:
    src_pts_sift = np.float32([kp1_sift[m.queryIdx].pt for m in good_sift]).reshape(-1, 1, 2)
    dst_pts_sift = np.float32([kp2_sift[m.trainIdx].pt for m in good_sift]).reshape(-1, 1, 2)

    M_sift, mask_sift = cv.findHomography(src_pts_sift, dst_pts_sift, cv.RANSAC, 5.0)
    matchesMask_sift = mask_sift.ravel().tolist()

    h, w = cropped_image.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M_sift)

    img2_sift = cv.polylines(img2.copy(), [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    print("SIFT | Matches found - {}/{}".format(len(good_sift), MIN_MATCH_COUNT))
else:
    print("SIFT | Not enough matches are found - {}/{}".format(len(good_sift), MIN_MATCH_COUNT))
    matchesMask_sift = None

# BFMatcher for ORB descriptors
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches_orb = bf.match(des1_orb, des2_orb)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)

# store all the good matches as per Lowe's ratio test for ORB
good_orb = matches_orb[:MIN_MATCH_COUNT] if len(matches_orb) > MIN_MATCH_COUNT else matches_orb

if len(good_orb) > MIN_MATCH_COUNT:
    src_pts_orb = np.float32([kp1_orb[m.queryIdx].pt for m in good_orb]).reshape(-1, 1, 2)
    dst_pts_orb = np.float32([kp2_orb[m.trainIdx].pt for m in good_orb]).reshape(-1, 1, 2)

    M_orb, mask_orb = cv.findHomography(src_pts_orb, dst_pts_orb, cv.RANSAC, 5.0)
    matchesMask_orb = mask_orb.ravel().tolist()

    h, w = cropped_image.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M_orb)

    img2_orb = cv.polylines(img2.copy(), [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    print("ORB | Matches found - {}/{}".format(len(good_orb), MIN_MATCH_COUNT))
else:
    print("ORB | Not enough matches are found - {}/{}".format(len(good_orb), MIN_MATCH_COUNT))
    matchesMask_orb = None

# Draw matches
draw_params_sift = dict(matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=matchesMask_sift,  # draw only inliers
                        flags=2)

draw_params_orb = dict(matchColor=(255, 0, 0),  # draw matches in blue color
                        singlePointColor=None,
                        matchesMask=matchesMask_orb,  # draw only inliers
                        flags=2)

img3_sift = cv.drawMatches(cropped_image, kp1_sift, img2, kp2_sift, good_sift, None, **draw_params_sift)
img3_orb = cv.drawMatches(cropped_image, kp1_orb, img2, kp2_orb, good_orb, None, **draw_params_orb)

plt.subplot(121), plt.imshow(img3_sift, 'gray'), plt.title('SIFT Matches')
plt.subplot(122), plt.imshow(img3_orb, 'gray'), plt.title('ORB Matches')
plt.show()
