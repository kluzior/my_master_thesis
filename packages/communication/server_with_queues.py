import socket
import threading
#from queue import Queue
import logging

# Server configuration 
HOST = "192.168.100.38"  
PORT = 30002  

def handle_robot(sock, robot_address, queue_from_robot, queue_to_robot):
    """
    Function to handle communication with the connected robot.

    Args:
        sock (socket.socket): The socket object for the robot connection.
        robot_address (tuple): A tuple containing the robot's IP address and port.
    """
    logging.getLogger('logger').info(f'connected to robot at {robot_address}')

    while True:
        try:
            # Receive data from the robot
            data = sock.recv(1024).decode()
            if not data:  # Handle empty data or connection closure
                logging.getLogger('logger').warning(f'robot {robot_address} disconnected')
                start_server(queue_from_robot, queue_to_robot)
                break
            queue_from_robot.put(data)  # Add received data to the queue

            # Check if data needs to be sent to the robot
            if not queue_to_robot.empty():
                data = queue_to_robot.get()
                sock.sendall(data.encode())

        except (socket.error, ConnectionResetError):
            logging.getLogger('logger').error(f'error communicating with robot {robot_address}')
            break

        except Exception as e:  # Catch unexpected errors
            logging.getLogger('logger').error(f'unexpected error: {e}')
            break


def start_server(queue_from_robot, queue_to_robot):
    """
    Function to start the server and listen for connections.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen()

        #print(f"Server listening on {HOST}:{PORT}")
        logging.getLogger('logger').info(f'server listening on {HOST}:{PORT}')

        while True:
        # Accept connection from the robot
            conn, robot_address = server_sock.accept()
            # Create a separate thread for each robot connection
            robot_thread = threading.Thread(target=handle_robot, args=(conn, robot_address, queue_from_robot, queue_to_robot))
            robot_thread.start()
            print(f'queue size (inside): {queue_from_robot.qsize()}')
