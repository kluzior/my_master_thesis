from packages2.robot_poses import RobotPoses
from packages2.common import start_communication
from packages2.robot_functions import RobotFunctions

c,s = start_communication()

rf = RobotFunctions(c)

# rf.moveJ_pose(RobotPoses.before_banks)

rf.moveJ_pose(RobotPoses.look_at_objects)