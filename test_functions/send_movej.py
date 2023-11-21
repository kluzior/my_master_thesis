# send_movej.py

import socket
import time
import math
HOST = "192.168.100.29"
PORT = 30002

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

pose_send_j = {
    "base": math.radians(30),
    "shoulder": math.radians(-120),
    "elbow": math.radians(-35),
    "wrist1": math.radians(0),
    "wrist2": math.radians(45),
    "wrist3": math.radians(60.5)
}
acceleration = 1
velocity = 1
cmd = f"movej([{pose_send_j.get('base')}," +\
        f"{pose_send_j.get('shoulder')}," +\
        f"{pose_send_j.get('elbow')}," +\
        f"{pose_send_j.get('wrist1')}," +\
        f"{pose_send_j.get('wrist2')}," +\
        f"{pose_send_j.get('wrist3')}" +\
        f"], a = {acceleration}, v = {velocity})" + "\n"
s.send(cmd.encode('utf-8'))

time.sleep(5)

pose_send_j["base"] += math.radians(50)
pose_send_j['shoulder'] -= math.radians(45)
velocity = 0.2
cmd = f"movej([{pose_send_j.get('base')}," +\
        f"{pose_send_j.get('shoulder')}," +\
        f"{pose_send_j.get('elbow')}," +\
        f"{pose_send_j.get('wrist1')}," +\
        f"{pose_send_j.get('wrist2')}," +\
        f"{pose_send_j.get('wrist3')}" +\
        f"], a = {acceleration}, v = {velocity})" + "\n"

s.send(cmd.encode('utf-8'))

data = s.recv(1024)
s.close()

print('Received', repr(data))