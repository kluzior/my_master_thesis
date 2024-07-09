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
a, positions  = handeye.run()
print(f"{len(a)}")
pos_list = [list(pos) for pos in positions]
print(pos_list)
handeye.prepare_data(a, pos_list)
print("Done!") 