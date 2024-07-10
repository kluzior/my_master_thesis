import re
import time
from packages2.cmd_generator import CmdGenerator
import logging

class Robot:
    def __init__(self, client):
        self.client = client
        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'Robot({self}) was initialized.')

    def give_pose(self):
        self.client.send(CmdGenerator.basic("give_pose"))
        msg = self.client.recv(1024)
        pose = self.concat_tcp_pose(msg)
        return pose

    def set_gripper(self):
        self.client.send(CmdGenerator.basic("set_gripper"))
        msg = self.client.recv(1024)
        if msg == b"gripper_on":
            time.sleep(1)           # give time for gripper to grip
            return 0
        else: 
            self._logger.error(f'set_gripper | wrong message : {msg}')
            return 1

    def reset_gripper(self):
        self.client.send(CmdGenerator.basic("reset_gripper"))
        msg = self.client.recv(1024)
        if msg == b"gripper_off":
            time.sleep(1)           # give time for gripper to release
            return 0
        else: 
            self._logger.error(f'reset_gripper | wrong message : {msg}')
            return 1

    def moveL_onlyZ(self, z_trans):
        robot_pose = self.give_pose()
        self.client.send(CmdGenerator.basic("MoveL"))
        msg = self.client.recv(1024)
        if msg == b"MoveL_wait_pos":
            x, y, z, rx, ry, rz = robot_pose
            cmd = f"({x}, {y}, {z + z_trans})"
            self.client.send(CmdGenerator.basic(cmd))
            msg = self.client.recv(1024)
            if msg == b"MoveL_done":
                return 0
            else:
                self._logger.error(f'moveL done | wrong message : {msg}')
            return 1
        else: 
            self._logger.error(f'moveL | wrong message : {msg}')
            return 1

    def moveJ(self, pose):
        self.client.send(CmdGenerator.basic("MoveJ"))
        msg = self.client.recv(1024)
        if msg == b"MoveJ_wait_pos":
            self.client.send(CmdGenerator.pose_convert_to_tcp_frame(pose))
            msg = self.client.recv(1024)
            if msg == b"MoveJ_done":
                return 0
            else:
                self._logger.error(f'moveJ done | wrong message : {msg}')
            return 1        

        else:
            self._logger.error(f'moveJ | wrong message : {msg}')
            return 1            

    def concat_tcp_pose(self, received_data):
        received_data = received_data.decode('utf-8')
        self._logger.debug(f'received_data: {received_data}')
        matches = re.findall(r'-?\d+\.\d+e?-?\d*', received_data)
        if len(matches) == 6:
            return map(float, matches)
        else:
            self._logger.error("Error: Expected 6 values, but found a different number.")
            return (0, 0, 0, 0, 0, 0)        

