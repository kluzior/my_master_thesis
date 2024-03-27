from queue import Queue 
from threading import Thread 

from packages.logger_configuration import configure_logger
from packages.photo_recognitions.main import calibrate_camera
from packages.communication.server_with_queues import start_server

import logging

is_camera_calibrated = 0
 

# test - to delete
def print_cube(num):
    for i in range(10):
        print("Cube: {}" .format(num * num * num))
        print(f'cube {i}')
 
# test - to delete
def print_square(num):
    for i in range(10):
        print("Square: {}" .format(num * num))
        print(f'square {i}')

 
if __name__ =="__main__":
    # Queues for communication with the robot
    queue_from_robot = Queue(maxsize = 10)
    queue_to_robot = Queue()
    
    
    configure_logger()

    #start_server()
    t_server = Thread(target=start_server, args=(queue_from_robot, queue_to_robot))
    t_server.start()
    
    t_calibration = Thread(target=calibrate_camera, args=())
    t_calibration.start()

    # test threading
    t1 = Thread(target=print_square, args=(10,))
    t2 = Thread(target=print_cube, args=(10,))
 
    t1.start()
    t2.start()
 
    t_calibration.join()
    t1.join()
    t2.join()
    t_server.join()

    logging.getLogger('logger').debug("test")


    print("Done!")



