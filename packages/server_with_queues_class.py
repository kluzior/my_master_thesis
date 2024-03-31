import socket
import threading
import logging

class Server:
    def __init__(self, host = "192.168.100.38", port = 30002):
        self.HOST = host  
        self.PORT = port 
        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'Server({self}) was initialized.')

    def handle_robot(self, queue_from_robot, queue_to_robot):
        """
        Function to handle communication with the connected robot.

        """
        self._logger.info(f'connected to robot at {self.robot_address}')

        while True:
            try:
                # Receive data from the robot
                data = self.sock.recv(1024).decode()
                if not data:  # Handle empty data or connection closure
                    self._logger.warning(f'robot {self.robot_address} disconnected')
                    self.start_server(queue_from_robot, queue_to_robot)
                    break
                queue_from_robot.put(data)  # Add received data to the queue

                # Check if data needs to be sent to the robot
                if not queue_to_robot.empty():
                    data = queue_to_robot.get()
                    self.sock.sendall(data.encode())

            except (socket.error, ConnectionResetError):
                self._logger.error(f'error communicating with robot {self.robot_address}')
                break

            except Exception as e:  # Catch unexpected errors
                self._logger.error(f'unexpected error: {e}')
                break

    def start_server(self, queue_from_robot, queue_to_robot):
        """
        Function to start the server and listen for connections.

        """
        self.sock = 0 
        self.robot_address = 0

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
            server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_sock.bind((self.HOST, self.PORT))
            server_sock.listen()
            self._logger.info(f'server listening on {self.HOST}:{self.PORT}')

            while True:
            # Accept connection from the robot
                self.sock, self.robot_address = server_sock.accept()
                # Create a separate thread for each robot connection
                self.robot_thread = threading.Thread(target=self.handle_robot, args=(queue_from_robot, queue_to_robot))
                self.robot_thread.start()
                print(f'queue size (inside): {queue_from_robot.qsize()}')








    # how to make robot thread stop?

    # def run(self, queue_from_robot, queue_to_robot):

    #     # accept connection thread 
    #     self.acc_thread = threading.Thread(target=self.start_server, args=())
    #     self.acc_thread.start()
        
        
    #     self.robot_thread = threading.Thread(target=self.handle_robot, args=(self.conn, self.robot_address, queue_from_robot, queue_to_robot))
    #     self.robot_thread.start()