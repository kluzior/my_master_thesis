import cv2


class Camera:
    def __init__(self, mtx, distortion, index=0):
        self.cap = cv2.VideoCapture(index)
        if self.cap.isOpened():
            print("Camera initialized")
        else:
            print("Failed to initialize camera")
        self.mtx = mtx 
        self.distortion = distortion

    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return self.undistort_frame(frame)
        return None  # Return None if not opened or read fails

    def get_frame_old(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()

        return frame

    def undistort_frame(self, frame):
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.distortion, (w, h), 1, (w, h))
        undst = cv2.undistort(frame, self.mtx, self.distortion, None, newcameramtx)
        x, y, w, h = roi
        undst = undst[y:y+h, x:x+w]
        return undst