import socket
import threading
import time

class ClientThread(threading.Thread):
    def __init__(self, host, port):
        super().__init__()
        self.HOST = host
        self.PORT = port
        self.s = None
        self.running = False
        self.connected = False

    def run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.s.connect((self.HOST, self.PORT))
            self.connected = True
        except ConnectionError:
            self.connected = False

        self.running = True
        while self.running:
            time.sleep(1)  # Keep the thread alive to maintain the connection

    def send(self, cmd):
        if self.is_connected():
            self.s.send(cmd.encode('utf-8'))

    def receive(self):
        if self.is_connected():
            data = self.s.recv(1024)
            return data.decode('utf-8')
        return None

    def stop(self):
        self.running = False
        if self.s:
            self.s.close()

    def is_connected(self):
        return self.connected
