from queue import Queue 
from threading import Thread 
import time

from packages.logger_configuration import configure_logger
from packages.server_with_queues_class import Server


from packages.calibration import calibrateCamera

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


def dummy_queue_print(queue_from_robot, queue_to_robot):
    while True:
        if not queue_from_robot.empty():
            data = queue_from_robot.get()
            print(f'data from queue: {data}')
            print(f'queue size (inside): {queue_from_robot.qsize()}')
            time.sleep(1)        

 
if __name__ =="__main__":
    # Queues for communication with the robot
    queue_from_robot = Queue(maxsize = 10)
    queue_to_robot = Queue()
    
    
    configure_logger()

    #start_server()
    # t_server = Thread(target=start_server, args=(queue_from_robot, queue_to_robot))
    # t_server.start()

    serverObj = Server()
    t_server_class = Thread(target=serverObj.start_server, args=(queue_from_robot, queue_to_robot))
    t_server_class.start()  
    
    t_dummy_queue = Thread(target=dummy_queue_print, args=(queue_from_robot, queue_to_robot))
    t_dummy_queue.start()

    obj = calibrateCamera()
    t_calibration = Thread(target=obj.run(), args=())
    t_calibration.start()
    

    # test threading
    # t1 = Thread(target=print_square, args=(10,))
    # t2 = Thread(target=print_cube, args=(10,))
 
    # t1.start()
    # t2.start()


    t_dummy_queue.join()
    t_calibration.join()
    # t1.join()
    # t2.join()
    # t_server.join()
    t_server_class.join()
    serverObj.robot_thread.join()

    logging.getLogger('logger').debug("test")


    print("Done!")



