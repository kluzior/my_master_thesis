import socket
import re
import threading
import time

class ServerThread(threading.Thread):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = False
        self.client_socket = None
        self.client_address = None

    def run(self):
        print(f"Serwer nasłuchuje na {self.host}:{self.port}")
        self.running = True
        while self.running:
            try:
                self.client_socket, self.client_address = self.server_socket.accept()
                print(f"Połączono z {self.client_address[0]}:{self.client_address[1]} (IP:dynamic_port)")
                while self.running and self.client_socket:
                    pass
            except (ConnectionError, ConnectionAbortedError):
                print("Błąd połączenia z klientem")
                self.client_socket = None
            finally:
                if self.client_socket:
                    self.client_socket.close()

    def stop(self):
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        self.server_socket.close()
        print("Serwer zatrzymany")

    def is_client_connected(self):
        return self.client_socket is not None

    def send_pose_request(self):
        if self.is_client_connected():
            self.client_socket.sendall(b"give_pose")
            data = self.client_socket.recv(1024)
            if data:
                x, y, z, rx, ry, rz = self.concat_tcp_pose(data.decode('utf-8'))
                print(f"Otrzymano pozycję: \n   x={x}, y={y}, z={z}, rx={rx}, ry={ry}, rz={rz}")
                return (x, y, z, rx, ry, rz)
        return None

    def concat_tcp_pose(self, received_data):
        matches = re.findall(r'-?\d+\.\d+e?-?\d*', received_data)
        if len(matches) == 6:
            return map(float, matches)
        else:
            print("Error: Expected 6 values, but found a different number.")
            return (0, 0, 0, 0, 0, 0)

    def send_move_command(self, x, y, z):
        if self.is_client_connected():
            cmd = f"({x}, {y}, {z})"
            print(f"Wysyłam komendę ruchu: {cmd}")
            self.client_socket.sendall(cmd.encode('utf-8'))
            move_response = self.client_socket.recv(1024)
            print(f"Otrzymano odpowiedź na komendę ruchu: \n     {move_response.decode('utf-8')}")
            if move_response.decode('utf-8') == "moveL OK":
                return True
            
        return False
