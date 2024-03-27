import time

def dummy_queue_print(queue_from_robot, queue_to_robot):
    while True:
        if not queue_from_robot.empty():
            data = queue_from_robot.get()
            print(f'data from queue: {data}')
            print(f'queue size (inside): {queue_from_robot.qsize()}')
            time.sleep(1)
