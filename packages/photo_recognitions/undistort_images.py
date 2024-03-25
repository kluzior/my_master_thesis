import cv2
from packages.store import save_image

def undistort_images(images, camera_matrix, dist, write_path=''):
   
    num = 0 
    undistorted = []
    for image in images:
        img = cv2.imread(image) 
        h, w = img.shape[:2] 

        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist, (w,h), 1, (w,h))

        dst = cv2.undistort(img, camera_matrix, dist, None, newCameraMatrix)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w] 
        undistorted.append(img)
        save_image(write_path, num, img)   

        num += 1 

    return undistorted