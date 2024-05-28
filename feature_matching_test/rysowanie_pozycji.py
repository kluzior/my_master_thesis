import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def crop_image(image, margin=5):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    cropped_images = []
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)

        # Create a mask for the current contour
        mask = np.zeros_like(image)
        cv.drawContours(mask, [contour], 0, (255), thickness=cv.FILLED)

        # Apply the mask to the original image
        masked_image = cv.bitwise_and(image, mask)

        cropped_images.append(masked_image[y:y+h, x:x+w])

    return cropped_images


MIN_MATCH_COUNT = 10

img1 = cv.imread('C:/Users/kluziak/OneDrive - Nokia/my_master_thesis/feature_matching_test/ref3.png',cv.IMREAD_GRAYSCALE) # trainImage
img2 = cv.imread('C:/Users/kluziak/OneDrive - Nokia/my_master_thesis/feature_matching_test/3.png',cv.IMREAD_GRAYSCALE) # queryImage
_, mask1 = cv.threshold(img1, 230, 255, cv.THRESH_BINARY)
_, mask2 = cv.threshold(img2, 230, 255, cv.THRESH_BINARY)
cv.imshow('img2', img2); cv.waitKey(1000)
cv.imshow('test1', mask1); cv.waitKey(100)
cv.imshow('test2', mask2); cv.waitKey(100)

img1 = crop_image(mask1)[0]
# img2 = crop_image(mask2)[0]
img2 = mask2

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 100)

flann = cv.FlannBasedMatcher(index_params, search_params)

matches_flann = flann.knnMatch(des1, des2,k=2)

# store all the good matches as per Lowe's ratio test.
good_sift = []
for m, n in matches_flann:
    if m.distance < 0.75 * n.distance:
        good_sift.append(m)
print(f'len(good_sift): {len(good_sift)}')


if len(good_sift)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_sift ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_sift ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    print( "FLANN | Enough matches are found - {}/{}".format(len(good_sift), MIN_MATCH_COUNT) )

else:
    print( "FLANN | Not enough matches are found - {}/{}".format(len(good_sift), MIN_MATCH_COUNT) )
    matchesMask = None


draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    # matchesMask = matchesMask, # draw only inliers
                    flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good_sift,None,**draw_params)

plt.imshow(img3, 'gray'),plt.show()