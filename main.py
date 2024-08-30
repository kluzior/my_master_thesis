"""
MAIN PROGRAM TO CONTROL UR COBOT WITH VISUAL SYSTEM

"""

from packages2.start_communication import start_communication
from packages2.robot_functions import RobotFunctions
from packages2.camera_calibrator import CameraCalibrator









# start communication
c, s = start_communication()

# initialization
robot_functions = RobotFunctions(c)
camera_calibrator = CameraCalibrator(c)

# camera calibration
camera_calib_result_path = camera_calibrator.run()
print(f"camera_calib_result_path: {camera_calib_result_path}")




# end communication
c.close()
s.close()
