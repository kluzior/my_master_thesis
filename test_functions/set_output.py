# set_output.py

import socket
import time

HOST = "192.168.100.39"
PORT = 30001

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

cmd = "set_digital_out(2, True)" + "\n"
s.send(cmd.encode('utf-8'))

time.sleep(2)

cmd = "set_digital_out(2, False)" + "\n"
s.send(cmd.encode('utf-8'))

#data = s.recv(1024)
s.close()

#print('Received', repr(data))