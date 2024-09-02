"""
MAIN PROGRAM TO CONTROL UR COBOT WITH VISUAL SYSTEM

program działa z 2024_08_30_SK_comm
"""

from packages2.start_communication import start_communication
from packages2.robot_functions import RobotFunctions
from packages2.camera_calibrator import CameraCalibrator
from packages2.handeye_eye_in_hand_NEW import HandEyeCalibration
from packages2.plane_determination import PlaneDeterminator
from packages2.logger_configuration import configure_logger

OFFLINE_SIMULATION_ONLY = False
COLLECT_CAMERA_CALIB_DATA = False
COLLECT_HAND_EYE_CALIB_DATA = False




# start communication
if OFFLINE_SIMULATION_ONLY:
    c = None
else:
    c, s = start_communication()

# initialization
robot_functions = RobotFunctions(c)
configure_logger()

# camera calibration
camera_calibrator = CameraCalibrator(c)
if COLLECT_CAMERA_CALIB_DATA:
    camera_intrinsic_path = camera_calibrator.run()
else:
    cc_files = "data/results/for_camera_calib/images_28-08_10-27"
    camera_intrinsic_path = camera_calibrator.run(cc_files)
print(f"camera_intrinsic_path: {camera_intrinsic_path}")


# hand-eye calibration
handeye_calibrator = HandEyeCalibration(camera_intrinsic_path, c)
if COLLECT_HAND_EYE_CALIB_DATA:
    handeye_path = handeye_calibrator.run()
else:
    he_files = "data/results/for_hand_eye_calib/images_02-09_10-26"
    handeye_path = handeye_calibrator.run(he_files)


# test run to few points on chess board
if OFFLINE_SIMULATION_ONLY:
    pass
else:
    # handeye_calibrator.send_robot_to_test_poses(handeye_path, handeye_type='tsai')
    pass

# plane determination
pd = PlaneDeterminator(camera_intrinsic_path, handeye_path)

test_pose = pd.test_run()
print(f"test_pose: {test_pose}")


robot_functions.moveJ_pose(test_pose)







# end communication
# c.close()
# s.close()
