import cv2

class Camera:
    def __init__(self, index=0): 
        self.cap = cv2.VideoCapture(index)
        print("Camera initialized")

    def take_picture(self):
        print("Picture taken") 


    def get_frame(self): 
        if self.cap.isOpened():
            _, frame = self.cap.read()

        return frame

    def show_frame(self): 
        if self.cap.isOpened():
            _, frame = self.cap.read()
            cv2.imshow('frame', frame)


