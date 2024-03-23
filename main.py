import threading
from logger_configuration import configure_logger
from photo_recognitions.main import calibrate_camera

import logging

is_camera_calibrated = 0
 

# test - to delete
def print_cube(num):
    for i in range(100):
        print("Cube: {}" .format(num * num * num))
        print(f'cube {i}')
 
# test - to delete
def print_square(num):
    for i in range(100):
        print("Square: {}" .format(num * num))
        print(f'square {i}')

 
if __name__ =="__main__":
    configure_logger()

    #t_calibration = threading.Thread(target=calibrate_camera)
    calibrate_camera()
    
    # test threading
    t1 = threading.Thread(target=print_square, args=(10,))
    t2 = threading.Thread(target=print_cube, args=(10,))
 
    #t_calibration.start()
    t1.start()
    t2.start()
 
    #t_calibration.join()
    t1.join()
    t2.join()
 
    logging.getLogger('logger').debug("test")


    print("Done!")



