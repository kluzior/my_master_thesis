import socket
import time
import math
import sys
sys.path.append("my_func")
import my_functions
HOST = "192.168.100.29"
PORT = 30001

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.connect((HOST, PORT))

pose_send = {
    "base": 90,
    "shoulder": 180,
    "elbow": 56,
    "wrist1": 78,
    "wrist2": 94,
    "wrist3": 61
}


function = "movej"

cmd = my_functions.send_pos_concatenate(pose_send)

print(cmd)

#cmd = """movej([-0.540537125683036, -2.2018732555807086, -1.0986348160112505, -2.6437150406384227, 3.352864759694935, -1.2294883935868013], a=1.3962634015954636, v=1.0471975511965976)""" + "\n"
s.send(cmd.encode('utf-8'))

time.sleep(10)


cmd = """movej([0,1.57,-1.57,3.14,-1.57,1.57], a=1.4, v=1.05, t=0, r=0)""" + "\n"
#cmd = my_functions.send_pos_concatenate()

s.send(cmd.encode('utf-8'))

data = s.recv(1024)
s.close()

print('Received', repr(data))