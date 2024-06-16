import socket
import time

class TestClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):
        self.client_socket.connect((self.host, self.port))
        print(f"Połączono z serwerem na {self.host}:{self.port}")

    def receive_and_respond(self):
        try:
            while True:
                data = self.client_socket.recv(1024)
                if not data:
                    break
                command = data.decode('utf-8')
                print(f"Otrzymano od serwera: {command}")
                if command == "give_pose":
                    pose = "p[-0.187702, 0.27884, -0.237913, -3.11466, -0.410051, -3.8586e-06]"
                    self.client_socket.sendall(pose.encode('utf-8'))
                elif command.startswith("("):
                    print(f"Komenda ruchu: {command}")
                    time.sleep(5)
                    self.client_socket.sendall(b"moveL OK")
                else:
                    print(f"Nieznana komenda: {command}")
        except KeyboardInterrupt:
            print("\nPrzerwanie przez użytkownika. Zamykanie klienta...")

    def stop(self):
        self.client_socket.close()
        print("Połączenie z serwerem zamknięte.")

if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 10000

    client = TestClient(HOST, PORT)
    try:
        client.start()
        client.receive_and_respond()
    finally:
        client.stop()
