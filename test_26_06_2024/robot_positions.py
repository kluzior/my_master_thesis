import math

class RobotPositions:
    vacuum_gripper = 9

    home = {
        "base": math.radians(0),
        "shoulder": math.radians(-90),
        "elbow": math.radians(0),
        "wrist1": math.radians(-90),
        "wrist2": math.radians(0),
        "wrist3": math.radians(0)                   # home position
    }


    pose_wait_base = {
        "base": math.radians(-45),
        "shoulder": math.radians(-112),
        "elbow": math.radians(100),
        "wrist1": math.radians(-80),
        "wrist2": math.radians(-90),
        "wrist3": math.radians(0)                   # base position
    }

    pose_1 = {
        "base": math.radians(-75),
        "shoulder": math.radians(-110),
        "elbow": math.radians(130),
        "wrist1": math.radians(-110),
        "wrist2": math.radians(-90),
        "wrist3": math.radians(0)                   # position for MOVEJ to pickup
    }
    pose_2 = {
        "base": math.radians(-35),
        "shoulder": math.radians(-113),
        "elbow": math.radians(110),
        "wrist1": math.radians(-85),
        "wrist2": math.radians(-90),
        "wrist3": math.radians(0)                   # position for MOVEJ to pickup
    }