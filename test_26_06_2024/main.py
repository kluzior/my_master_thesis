import time
from cmd_generator import CmdGenerator
from robot_positions import RobotPositions
from robot_functions import RobotFunctions

#my_test_server.py
import socket
import time
HOST = "192.168.0.1"
PORT = 10000
print("Start listening...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(5)
c, addr = s.accept()
Robot = RobotFunctions(c)

try:
    msg = c.recv(1024)
    print(msg)
    if msg == b"Hello server, robot here":
        print("Robot requested for data!")
        time.sleep(0.5)
        
        # Robot.moveJ(RobotPositions.home)
        # time.sleep(2)


        Robot.moveJ(RobotPositions.pose_wait_base)
        # time.sleep(2)
        # if ret == 0:

        Robot.moveJ(RobotPositions.pose_1)

        Robot.moveL_onlyZ(-0.17)
        
        Robot.set_gripper()
        Robot.moveL_onlyZ(0.17)


        Robot.moveJ(RobotPositions.pose_2)
        time.sleep(1)

        Robot.moveL_onlyZ(-0.31)
        Robot.reset_gripper()
        Robot.moveJ(RobotPositions.pose_2)
        Robot.moveL_onlyZ(-0.32)
        Robot.set_gripper()
        Robot.moveL_onlyZ(0.32)
        Robot.moveJ(RobotPositions.pose_1)
        Robot.moveL_onlyZ(-0.16)
        Robot.reset_gripper()
        Robot.moveJ(RobotPositions.pose_1)
        Robot.moveJ(RobotPositions.pose_wait_base)

        time.sleep(2)

except socket.error as socketerror:
    print("Socket error: ", socketerror)
    c.close()
    s.close()
    print("socket closed")




# if __name__ == "__main__":
