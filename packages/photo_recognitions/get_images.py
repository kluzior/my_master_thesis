import cv2
import logging
import os
from packages.store import save_image
#from packages.boxes import drawBoxes

def get_images(write_path=''):

    cap = cv2.VideoCapture(0)
    images = []
    num = 0

    while cap.isOpened():

        _, img = cap.read()
        k = cv2.waitKey(5)

        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            save_image(write_path, num, img)   
            images.append(img)
            logging.getLogger('logger').debug(f'captured photo no {num}')
            num += 1
        
        cv2.imshow('Img',img)

    cap.release()   
    cv2.destroyAllWindows()
    logging.getLogger('logger').info(f'captured {num} photos')

    return images

if __name__ == "__main__":
    #from ...save_image import save_image

    get_images()