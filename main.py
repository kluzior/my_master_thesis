"""
[PL]
FUNKCJA GŁÓWNA DO STEROWANIA COBOTA UR Z SYSTEMEM WIZYJNYM

Za pomocą funkcji można: 
    * przeprowadzić kalibrację kamery - camera_calibrator.run()
    * przeprowadzić kalibrację hand-eye - handeye_calibrator.run()
    * przetestować wyznaczone przekształcenia na punktach szachownicy - handeye_calibrator.send_robot_to_test_poses()
    * uruchomić maszynę stanów pętli nieskończonej - loop.run()


[ENG]
MAIN PROGRAM TO CONTROL UR COBOT WITH VISUAL SYSTEM

program działa z 2024_09_06_SK_comm
"""

from packages2.common import start_communication
from packages2.robot_functions import RobotFunctions
from packages2.camera_calibrator import CameraCalibrator
from packages2.handeye_eye_in_hand import HandEyeCalibration
from packages2.logger_configuration import configure_logger
from packages2.loop_state_machine import LoopStateMachine
import datetime

OFFLINE_SIMULATION_ONLY = False
COLLECT_CAMERA_CALIB_DATA = False
COLLECT_HAND_EYE_CALIB_DATA = False

timestamp = datetime.datetime.now().strftime("%d-%m_%H-%M")


# start communication
if OFFLINE_SIMULATION_ONLY:
    c = None
else:
    c, s = start_communication()

# initialization
robot_functions = RobotFunctions(c)
# configure_logger(timestamp)

# camera calibration
camera_calibrator = CameraCalibrator(c, timestamp)
if COLLECT_CAMERA_CALIB_DATA:
    camera_intrinsic_path = camera_calibrator.run()
else:
    cc_files = "data/results/for_camera_calib/images_07-09_16-28"
    camera_intrinsic_path = camera_calibrator.run(cc_files)
print(f"camera_intrinsic_path: {camera_intrinsic_path}")


# hand-eye calibration
handeye_calibrator = HandEyeCalibration(camera_intrinsic_path, c, timestamp)
if COLLECT_HAND_EYE_CALIB_DATA:
    handeye_path = handeye_calibrator.run()
    handeye_calibrator.send_robot_to_test_poses(handeye_path, handeye_type='tsai')
else:
    he_files = "data/results/for_hand_eye_calib/images_07-09_16-28"
    handeye_path = handeye_calibrator.run(he_files)


# LOOP
mlp_model_path = 'data/results/models/mlp_model_accuracy_0_9400_lbfgs_logistic_adaptive_batch32_07-09_16-27_1.joblib'

loop = LoopStateMachine(c, camera_intrinsic_path, handeye_path, mlp_model_path, timestamp)

loop.run()






# end communication
# c.close()
# s.close()
