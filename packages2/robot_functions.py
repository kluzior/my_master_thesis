import re
import time
from packages2.cmd_generator import CmdGenerator
from packages2.common import pose_list_to_dict, joints_list_to_dict

class RobotFunctions():
    def __init__(self, client):
        self.client = client

    def give_pose(self):
        self.client.send(CmdGenerator.basic("give_pose"))
        msg = self.client.recv(1024)
        pose = self.concat_tcp_pose(msg)
        return pose
    
    def give_joints(self):
        self.client.send(CmdGenerator.basic("give_joints"))
        msg = self.client.recv(1024)
        joints = self.concat_joints_pose(msg)
        return joints

    def set_gripper(self):
        self.client.send(CmdGenerator.basic("set_gripper"))
        msg = self.client.recv(1024)
        if msg == b"gripper_on":
            time.sleep(1)           # give time for gripper to grip
            return 0
        else: 
            print(f'set_gripper | wrong message : {msg}')
            return 1

    def reset_gripper(self):
        self.client.send(CmdGenerator.basic("reset_gripper"))
        msg = self.client.recv(1024)
        if msg == b"gripper_off":
            time.sleep(1)           # give time for gripper to release
            return 0
        else: 
            print(f'reset_gripper | wrong message : {msg}')
            return 1

    def moveL_pose(self, pose):
        self.client.send(CmdGenerator.basic("MoveL_pose"))
        msg = self.client.recv(1024)
        if msg == b"MoveL_pose_wait_pos":
            print(f'CmdGenerator.pose_convert_to_tcp_frame(pose): {CmdGenerator.pose_convert_to_tcp_frame(pose)}')
            self.client.send(CmdGenerator.pose_convert_to_tcp_frame(pose))
            msg = self.client.recv(1024)
            if msg == b"MoveL_pose_done":
                return 0
            else:
                print(f'moveL_pose done | wrong message : {msg}')
            return 1        
        else:
            print(f'moveL_pose | wrong message : {msg}')
            return 1

    def moveJ(self, joints):
        self.client.send(CmdGenerator.basic("MoveJ"))
        msg = self.client.recv(1024)
        if msg == b"MoveJ_wait_pos":
            print(f'CmdGenerator.joints_convert_to_tcp_frame(pose): {CmdGenerator.joints_convert_to_tcp_frame(joints)}')
            self.client.send(CmdGenerator.joints_convert_to_tcp_frame(joints))
            msg = self.client.recv(1024)
            if msg == b"MoveJ_done":
                return 0
            else:
                print(f'moveJ done | wrong message : {msg}')
            return 1        
        else:
            print(f'moveJ | wrong message : {msg}')
            return 1            

    def moveJ_pose(self, pose):
        self.client.send(CmdGenerator.basic("MoveJ_pose"))
        msg = self.client.recv(1024)
        if msg == b"MoveJ_pose_wait_pos":
            print(f'CmdGenerator.pose_convert_to_tcp_frame(pose): {CmdGenerator.pose_convert_to_tcp_frame(pose)}')
            self.client.send(CmdGenerator.pose_convert_to_tcp_frame(pose))
            msg = self.client.recv(1024)
            if msg == b"MoveJ_pose_done":
                return 0
            else:
                print(f'moveJ_pose done | wrong message : {msg}')
            return 1        
        else:
            print(f'moveJ_pose | wrong message : {msg}')
            return 1

    def concat_tcp_pose(self, received_data):
        received_data = received_data.decode('utf-8')
        print(f'received_data: {received_data}')
        matches = re.findall(r'-?\d+\.\d+e?-?\d*', received_data)
        if len(matches) == 6:
            return map(float, matches)
        else:
            print("Error: Expected 6 values, but found a different number.")
            return (0, 0, 0, 0, 0, 0)
        
    def concat_joints_pose(self, received_data):
        received_data = received_data.decode('utf-8')
        print(f'received_data: {received_data}')
        matches = re.findall(r'-?\d+\.\d+e?-?\d*', received_data)
        if len(matches) == 6:
            return map(float, matches)
        else:
            print("Error: Expected 6 values, but found a different number.")
            return (0, 0, 0, 0, 0, 0)
        

    def rotate_wrist3(self, angle):
        robot_joints = self.give_joints()
        target_joints = joints_list_to_dict(list(robot_joints))
        target_joints["wrist3"] += angle 
        self.moveJ(target_joints)

    def moveL_onlyZ(self, z):
        act_pose = self.give_pose()
        act_pose = list(act_pose)
        act_pose[2] += z
        pose = pose_list_to_dict(act_pose)
        self.moveL_pose(pose)


    def pick_object(self, pose, angle, z_offset=0.1):
        pick_pose = pose.copy()
        pick_pose["z"] += z_offset
        self.moveJ_pose(pick_pose)
        self.rotate_wrist3(angle)
        time.sleep(0.2)
        self.moveL_onlyZ(-z_offset)
        time.sleep(0.2)
        self.set_gripper()
        time.sleep(0.2)
        self.moveL_pose(pick_pose)
        return False

    def drop_object(self, pose, z_offset=0.1):
        drop_pose = pose.copy()
        drop_pose["z"] += z_offset
        self.moveJ_pose(drop_pose)
        time.sleep(0.2)
        self.moveL_pose(pose)
        time.sleep(0.2)
        self.reset_gripper()
        time.sleep(0.2)
        self.moveL_pose(drop_pose)
        return False