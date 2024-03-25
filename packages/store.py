import cv2
import logging
import os
import numpy as np


def save_image(path, num, img):
    if path != '':
        path = 'images/' + path + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        if not cv2.imwrite(path + str(num) + '.png', img):
            raise Exception("Could not write image")
        
    

