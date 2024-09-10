import math

class RobotPoses:
    vacuum_gripper = 9

    look_at_chessboard = {
        "x": -0.190,
        "y": 0.110,
        "z": 0.635,
        "Rx": -2.8651,
        "Ry": -1.2757,
        "Rz": 0
    } 

    # look_at_objects = {
    #     "x": -0.190,
    #     "y": 0.110,
    #     "z": 0.638,
    #     "Rx": -2.8651,
    #     "Ry": -1.2757,
    #     "Rz": 0
    # } 

    before_banks = {
        "x": 0.000,
        "y": 0.250,
        "z": 0.135,
        "Rx": 3.09,
        "Ry": -0.40,
        "Rz": 0
    } 


    bank1 = {
        "x": 0.070,
        "y": 0.220,
        "z": 0.02,
        "Rx": 3.14,
        "Ry": 0,
        "Rz": 0
    } 
    bank2 = {
        "x": 0.150,
        "y": 0.150,
        "z": 0.02,
        "Rx": 1.40,
        "Ry": -2.77,
        "Rz": 0
    } 
    bank3 = {
        "x": 0.210,
        "y": 0.065,
        "z": 0.02,
        "Rx": 1.24,
        "Ry": -2.88,
        "Rz": 0
    }     
    bank4 = {
        "x": 0.280,
        "y": 0.000,
        "z": 0.02,
        "Rx": 1.18,
        "Ry": -2.90,
        "Rz": 0
    }     
    bank5 = {      
        "x": 0.180,
        "y": -0.080,
        "z": 0.02,
        "Rx": 1.09,
        "Ry": -2.83,
        "Rz": 0
    } 

    banks = [None, bank1, bank2, bank3, bank4, bank5]