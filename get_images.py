import cv2

cap = cv2.VideoCapture(0)

num = 0

while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        if not cv2.imwrite('my_master_thesis/image_banks/00_captured_raw/img' + str(num) + '.png', img):
            raise Exception("Could not write image")
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()   

cv2.destroyAllWindows()