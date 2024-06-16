import socket
import threading
import time

class ClientThread(threading.Thread):
    def __init__(self, host, port):
        threading.Thread.__init__(self)
        self.HOST = host
        self.PORT = port
        self.s = None
        self.running = False

    def run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.HOST, self.PORT))
        self.running = True
        while self.running:
            time.sleep(1)  # Keep the thread alive to maintain the connection

    def send(self, cmd):
        if self.s:
            self.s.send(cmd.encode('utf-8'))

    def receive(self):
        if self.s:
            data = self.s.recv(1024)
            return data.decode('utf-8')
        return None

    def stop(self):
        self.running = False
        if self.s:
            self.s.close()
