import socket
import threading

class ClientThread(threading.Thread):
    def __init__(self, host, port):
        super().__init__()
        self.host = host
        self.port = port
        self.s = None
        self.running = False
        self.connected = False

    def run(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.s.connect((self.host, self.port))
            self.connected = True
        except ConnectionError:
            self.connected = False
        self.running = True
        while self.running:
            pass

    def send(self, cmd):
        if self.is_connected():
            self.s.send(cmd.encode('utf-8'))
            print(f"ClientThread | sent: \n {cmd.encode('utf-8')}")

    def receive(self):
        if self.is_connected():
            data = self.s.recv(1024)
            # print(f"ClientThread | received: \n {data.decode('utf-8')}")
            return data.decode('utf-8')
        return None

    def stop(self):
        self.running = False
        if self.s:
            self.s.close()

    def is_connected(self):
        return self.connected
