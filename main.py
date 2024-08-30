"""
MAIN PROGRAM TO CONTROL UR COBOT WITH VISUAL SYSTEM

"""

from packages2.start_communication import start_communication
from packages2.robot_functions import RobotFunctions
from packages2.camera_calibrator import CameraCalibrator
from packages2.handeye_eye_in_hand_NEW import HandEyeCalibration

from packages2.logger_configuration import configure_logger








# start communication
c, s = start_communication()

# initialization
robot_functions = RobotFunctions(c)
configure_logger()

# camera calibration
camera_calibrator = CameraCalibrator(c)
camera_intrinsic_path = camera_calibrator.run()
print(f"camera_calib_result_path: {camera_intrinsic_path}")



# hand-eye calibration
# handeye_calibrator = HandEyeCalibration(camera_calib_result_path, c)
# handeye_calibrator.run()

# end communication
c.close()
s.close()
