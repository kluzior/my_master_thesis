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
        return ignored_image, contours
    

    def calculate_rvec_tvec(self, frame, point_shift=(0,0), chess_size=(8,7), side_of_chess_mm=23):
        mtx, distortion = self.load_camera_params('CameraParams.npz')
        # prepare chessboard corners in real world 
        objp = np.zeros((chess_size[0] * chess_size[1], 3), np.float32)
        objp[:,:2] = np.mgrid[0:chess_size[0],0:chess_size[1]].T.reshape(-1,2)
        objp[:, 0] -= point_shift[0]
        objp[:, 1] -= point_shift[1]
        objp *= side_of_chess_mm

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chess_size, None)
        print(f"TEST1, len corners: {len(corners)}")
        if ret == True:
            _, rvecs, tvecs = cv2.solvePnP(objp, corners, mtx, distortion)
            print(f"calculated rvec: {rvecs}")
            print(f"calculated tvec: {tvecs}")
        return rvecs, tvecs



    # def show_chess_corner(self, frame, chess_size=(8,7), point=(0,0)):
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     ret, corners2 = cv2.findChessboardCorners(gray, chess_size, None)    
    #     print(f"TEST2, len corners: {len(corners2)}")
    #     corners_mtx = np.array(corners2).reshape(chess_size[0], chess_size[1])
        
    #     image_to_show = frame.copy()
        
    #     center = corners_mtx(point[0], point[1])
    #     center = tuple(center.astype(int))
    #     # Rysowanie okręgu
    #     radius = 20  # Promień okręgu
    #     color = (0, 255, 0)  # Kolor okręgu (zielony)
    #     thickness = 2  # Grubość linii okręgu

    #     cv2.circle(image_to_show, center, radius, color, thickness)

    #     # Wyświetlanie obrazu
    #     cv2.imshow('Image with Circle', image_to_show)

    def show_chess_corner(self, frame, chess_size=(8,7), point=(0,0)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners2 = cv2.findChessboardCorners(gray, chess_size, None)
        if ret:
            print(f"TEST2, len corners: {len(corners2)}")
            # Reshape corners2 to a 3D array: (chess_size[0], chess_size[1], 2)
            corners_mtx = np.array(corners2).reshape(chess_size[1], chess_size[0], 2)
            
            image_to_show = frame.copy()
            
            # Access the specific point using the 'point' variable
            center = corners_mtx[point[1], point[0]]
            center = tuple(center.astype(int))
            # Rysowanie okręgu
            radius = 10  # Promień okręgu
            color = (0, 0, 255)  # Kolor okręgu 
            thickness = 5  # Grubość linii okręgu

            cv2.circle(image_to_show, center, radius, color, thickness)

            # Wyświetlanie obrazu
            cv2.imshow('Image with Circle', image_to_show)
            # cv2.waitKey(0)  # Czeka na naciśnięcie klawisza przed zamknięciem okna
            # cv2.destroyAllWindows()  # Zamyka wszystkie okna
        else:
            print("Nie znaleziono narożników szachownicy.")
