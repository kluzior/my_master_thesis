import math

class RobotPoses:
    vacuum_gripper = 9

    bank1 = {
        "x": 0.120,
        "y": 0.200,
        "z": 0.02,
        "Rx": 2.86,
        "Ry": -1.21,
        "Rz": 0
    } 
    bank2 = {
        "x": 0.240,
        "y": 0.045,
        "z": 0.02,
        "Rx": 2.13,
        "Ry": -2.26,
        "Rz": 0
    } 
    bank3 = {
        "x": -0.180,
        "y": -0.110,
        "z": 0.02,
        "Rx": -1.88,
        "Ry": -2.51,
        "Rz": 0
    }     
    bank4 = {
        "x": 0.000,
        "y": -0.250,
        "z": 0.02,
        "Rx": -0.12,
        "Ry": -3.09,
        "Rz": 0
    }     
    bank5 = {               ## TO BE CHANGED
        "x": 0.120,
        "y": 0.200,
        "z": 0.02,
        "Rx": 2.86,
        "Ry": -1.21,
        "Rz": 0
    } 

    banks = [bank1, bank2, bank3, bank4, bank5]