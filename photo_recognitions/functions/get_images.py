import cv2
import logging


def get_images(write_path=''):

    cap = cv2.VideoCapture(0)
    images = []
    num = 0

    while cap.isOpened():

        succes, img = cap.read()
        k = cv2.waitKey(5)

        if k == 27:
            break
        elif k == ord('s'): # wait for 's' key to save and exit
            if write_path != '':
                if not cv2.imwrite(write_path + str(num) + '.png', img):
                    raise Exception("Could not write image")
                
            images.append(img)
            logging.getLogger('logger').debug(f'captured photo no {num}')
            num += 1
        
        cv2.imshow('Img',img)

    # Release and destroy all windows before termination
    cap.release()   

    cv2.destroyAllWindows()
    logging.getLogger('logger').info(f'captured {num} photos')

    return images


if __name__ == "__main__":
    get_images()