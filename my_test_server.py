#my_test_server.py

import socket
import time

HOST = "192.168.100.30" 
PORT = 34567

print("Start listening...")
 
while True:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))        # Bind to the port 
    s.listen(5)                 # Now wait for client connection.
    c, addr = s.accept()        # Establish connection with client.

    try:
        msg = c.recv(1024)
        print(msg)
        time.sleep(1)
        if msg == "asking_for_data":
            print("The count is:")
    except socket.error as socketerror:
        print("Socket error")
c.close()
s.close()