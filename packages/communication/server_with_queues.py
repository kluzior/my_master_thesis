import socket
import threading
from queue import Queue
import time

# Server configuration (replace with your desired values)
HOST = "192.168.100.38"  # Replace with your server's IP address
PORT = 30002  # Choose a free port

# Queues for communication with the robot
# data_from_robot = Queue(maxsize = 10)
# data_to_robot = Queue()


def handle_robot(sock, robot_address, queue_from_robot, queue_to_robot):
    """
    Function to handle communication with the connected robot.

    Args:
        sock (socket.socket): The socket object for the robot connection.
        robot_address (tuple): A tuple containing the robot's IP address and port.
    """
    print(f"Connected to robot at {robot_address}")

    while True:
        try:
            # Receive data from the robot
            data = sock.recv(1024).decode()
            if not data:  # Handle empty data or connection closure
                print(f"Robot {robot_address} disconnected")
                start_server(queue_from_robot, queue_to_robot)
                break
            queue_from_robot.put(data)  # Add received data to the queue

            # Check if data needs to be sent to the robot
            if not queue_to_robot.empty():
                data = queue_to_robot.get()
                sock.sendall(data.encode())

        except (socket.error, ConnectionResetError):
            print(f"Error communicating with robot {robot_address}")
            break

        except Exception as e:  # Catch unexpected errors
            print(f"Unexpected error: {e}")
            break


def start_server(queue_from_robot, queue_to_robot):
    """
    Function to start the server and listen for connections.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen()

        print(f"Server listening on {HOST}:{PORT}")

        while True:
        # Accept connection from the robot
            conn, robot_address = server_sock.accept()


            print(f'conn = {conn} robot address = {robot_address}')
            # Create a separate thread for each robot connection
            robot_thread = threading.Thread(target=handle_robot, args=(conn, robot_address, queue_from_robot, queue_to_robot))
            robot_thread.start()
            #print("i'm in while true")
            print(f'queue size (inside): {queue_from_robot.qsize()}')
            
           
# if __name__ == "__main__":

    # Start the server
    # start_server()

    # # Main program loop (replace with your program logic)
    # while True:
    #     # Get data from the robot via data_from_robot queue
    #     # ... your program logic here ...

    #     # Send data to the robot via data_to_robot queue
    #     # ... your program logic here ...

    #     # Handle Ctrl+C interruption (use `keyboard` library for a cleaner approach)
    #     try:
    #         if not data_from_robot.empty():
    #             print(f'queue size: {data_from_robot.qsize()}')
    #             data = data_from_robot.get()
    #             print(f'DATA = {data}')
    #         #print('dummy')
    #         #time.sleep(0.05)
    #         # ... your program logic here ...
                

    #     except KeyboardInterrupt:
    #         print("Server shutting down...")
    #         break

    # Close any remaining connections (optional)
