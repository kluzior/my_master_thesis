# read_raw_data.py

import socket
import time

HOST = "192.168.100.29"
PORT = 30003

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

data = s.recv(1024)
s.close()

print('Received raw data: ', repr(data))