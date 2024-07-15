from packages2.handeye import HandEyeCalibration
import socket
from packages2.logger_configuration import configure_logger




HOST = "192.168.0.1"
PORT = 10000
print("Start listening...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(5)
c, addr = s.accept()
configure_logger("debug_handeye")
handeye = HandEyeCalibration(c)
handeye.run()
handeye.prepare_robot_mtx()
handeye.prepare_object_camera_mtx()
handeye.calibrate_tsai()
handeye.calibrate_li()
print("Done!") 