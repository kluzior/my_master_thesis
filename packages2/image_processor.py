import cv2
import numpy as np

class ImageProcessor:

    def load_camera_params(self, path):
        with np.load(path) as file:
            mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]
        return mtx, distortion

    def undistort_frame(self, frame, mtx, distortion):
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distortion, (w, h), 1, (w, h))
        undst = cv2.undistort(frame, mtx, distortion, None, newcameramtx)
        x, y, w, h = roi
        undst = undst[y:y+h, x:x+w]
        return undst

    def apply_binarization(self, gray, binarization='standard'):
        if binarization == 'standard' or binarization == 'raw':
            _, mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
        elif binarization == 'adaptive':
            mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
        elif binarization == 'canny':
            blur = cv2.GaussianBlur(gray.copy(), (5, 5), 0)
            mask = cv2.Canny(blur, 50, 150)
        elif binarization == 'otsu':
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError("Unknown binarization method")
        return mask

    def crop_image(self, image, margin=1, image_raw=None):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]
        cropped_images = []
        cropped_raw_images = []
        coordinates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(image.shape[1] - x, w + 2 * margin)
            h = min(image.shape[0] - y, h + 2 * margin)
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            cropped_images.append(masked_image[y:y+h, x:x+w])
            coordinates.append((x, y, w, h))
        if image_raw is not None:
            for contour in contours:
                cropped_raw_images.append(image_raw[y:y+h, x:x+w])
        return cropped_images, coordinates, cropped_raw_images
    
    def ignore_background(self, image, vertices):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [vertices], (255,255,255))
        ignored_image = cv2.bitwise_and(image, image, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"ignore_background | contours: {contours}")
        return ignored_image, contours



