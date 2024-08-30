"""
MAIN PROGRAM TO CONTROL UR COBOT WITH VISUAL SYSTEM

program działa z 2024_08_30_SK_comm
"""

from packages2.start_communication import start_communication
from packages2.robot_functions import RobotFunctions
from packages2.camera_calibrator import CameraCalibrator
from packages2.handeye_eye_in_hand_NEW import HandEyeCalibration

from packages2.logger_configuration import configure_logger

c = None
# CAMERA CALIBRATION FILES (uncomment when you want to not collect new data)
cc_files = "data/results/for_camera_calib/images_30-08_15-25"

# HAND_EYE CALIBRATION FILES (uncomment when you want to not collect new data)
he_files = "data/results/for_hand_eye_calib/images_30-08_15-26"   # wszystkie pomiary



# start communication
# c, s = start_communication()

# initialization
robot_functions = RobotFunctions(c)
configure_logger()

# camera calibration
camera_calibrator = CameraCalibrator(c)
# camera_intrinsic_path = camera_calibrator.run()
camera_intrinsic_path = camera_calibrator.run(cc_files)
print(f"camera_intrinsic_path: {camera_intrinsic_path}")



# hand-eye calibration
handeye_calibrator = HandEyeCalibration(camera_intrinsic_path, c)
# handeye_calibrator.run()
handeye_calibrator.run(he_files)

# test run to few points on chess board
handeye_calibrator.send_robot_to_test_poses(he_files, handeye_type='tsai')


# end communication
# c.close()
# s.close()
