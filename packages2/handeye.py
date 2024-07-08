from robot import Robot
from camera import Camera




class HandEyeCalibration:
    def __init__(self, robot_client):
        self.camera = Camera()
        self.robot = Robot(robot_client)

    def calibrate(self):
        pass 