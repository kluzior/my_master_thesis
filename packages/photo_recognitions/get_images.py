import cv2
import logging

from packages.store import save_image

def get_images(write_path=''):

    cap = cv2.VideoCapture(0)
    images = []
    num = 0

    while cap.isOpened():

        _, img = cap.read()
        k = cv2.waitKey(5)

        if k == 27:                                     # wait for 'Esc' key to save
            logging.getLogger('logger').debug(f'user pressed Esc key')
            break
        elif k == ord('s'):                             # wait for 's' key to save
            images.append(img)
            save_image(write_path, num, img)   
            logging.getLogger('logger').debug(f'captured photo no. {num}')
            num += 1
        
        cv2.imshow('Camera',img)

    cap.release()   
    cv2.destroyAllWindows()
    logging.getLogger('logger').info(f'captured {num} photos')
    logging.getLogger('logger').debug(f'exit get_images() program')
    
    return images