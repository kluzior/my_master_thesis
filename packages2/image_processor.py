import cv2
import numpy as np
import logging


class ImageProcessor:
    def __init__(self, camera_params_path = None):
        self.camera_params_path = camera_params_path

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'ImageProcessor({self}) was initialized.')

    def load_camera_params(self, path):
        with np.load(path) as file:
            mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]
        return mtx, distortion

    def undistort_frame(self, frame, mtx, distortion):
        """
        Undistorts a frame using camera calibration parameters.

        Args:
            frame (numpy.ndarray): Input frame to be undistorted.
            mtx (numpy.ndarray): Camera matrix.
            distortion (numpy.ndarray): Distortion coefficients.

        Returns:
            tuple: A tuple containing the undistorted frame and the new camera matrix.
        """
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distortion, (w, h), 1, (w, h))
        undst = cv2.undistort(frame, mtx, distortion, None, newcameramtx)
        x, y, w, h = roi
        undst = undst[y:y+h, x:x+w]
        return undst, newcameramtx

    def apply_binarization(self, gray, binarization='standard'):
        """
        Apply binarization to a grayscale image.

        Parameters:
        - gray: numpy.ndarray
            Grayscale image to be binarized.
        - binarization: str, optional
            Binarization method to be used. Default is 'standard'.
            Available options: 'standard', 'adaptive', 'canny', 'otsu'.

        Returns:
        - mask: numpy.ndarray
            Binarized image.

        Raises:
        - ValueError: If an unknown binarization method is provided.
        """
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
    
    def crop_contour(self, image, contour, margin=1, image_raw=None):
        x, y, w, h = cv2.boundingRect(contour)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        cropped_image = masked_image[y:y+h, x:x+w]
        contour_frame = (x, y, w, h)
        return cropped_image, contour_frame
    
    def ignore_background(self, image, vertices):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [vertices], (255,255,255))
        ignored_image = cv2.bitwise_and(image, image, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return ignored_image, contours

    def calculate_rvec_tvec(self, frame, point_shift=(0,0), chess_size=(8,7), side_of_chess_mm=40):
        """
        Calculates the chessboard's rotation (rvecs) and translation vector (tvecs) using the solvePnP method.

        Args:
            frame: The input frame, must be undistorted!
            point_shift: A tuple representing the shift of the chessboard corners in the x and y directions
            chess_size: A tuple representing the number of inner corners per chessboard row and column
            side_of_chess_mm: The size of each square on the chessboard in millimeters

        Returns:
            rvecs: The rotation vector
            tvecs: The translation vector
        """
        mtx, _ = self.load_camera_params(self.camera_params_path)
        # prepare chessboard corners in real world 
        objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chess_size[0],0:chess_size[1]].T.reshape(-1,2)
        objp[:, 0] -= point_shift[0]
        objp[:, 1] -= point_shift[1]
        side_of_chess_m = side_of_chess_mm / 1000
        objp *= side_of_chess_m
        
        index = np.where((objp == [0.0, 0.0, 0.0]).all(axis=1))[0]
        assert index.size == 1, "Index should have only one position"
        idx = index[0]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chess_size, None)
        if ret == True:
            _, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, None)    # None dist_param because image is already undistorted!
        return rvecs, tvecs, corners[idx]

    def get_roi_points(self, frame):                
        print("Please select ROI point on picture!")
        while True:
            def click_event(event, x, y, flags, params):
                roi_points, img = params['roi_points'], params['img']
                if event == cv2.EVENT_LBUTTONDOWN:
                    roi_points.append([x, y])
                    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
                    cv2.imshow("image", img)
            img = frame.copy()
            if img is None:
                print("Error loading image")
                return None
            roi_points = []
            cv2.imshow('image', img)
            cv2.setMouseCallback('image', click_event, {'roi_points': roi_points, 'img': img})
            while True:
                if cv2.waitKey(20) == 27:  # Esc key
                    break
            if len(roi_points) > 0:
                overlay = img.copy()
                pts = np.array([roi_points], np.int32)
                cv2.fillPoly(overlay, [pts], (255, 0, 0, 70))
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
                cv2.imshow("ROI", img)
                cv2.waitKey(1)
            user_input = input("Accept ROI? (y/n): ")
            if user_input.lower() == 'y':
                cv2.destroyWindow("ROI")
                cv2.destroyWindow('image')
                roi_array = np.array(roi_points)
                self._logger.info("ROI points saved as:\n", roi_array)
                return roi_array
            else:
                print("ROI selection cancelled. Trying again...")
                cv2.destroyWindow('image')
