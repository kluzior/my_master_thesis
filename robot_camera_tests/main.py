from common import generate_input_data
import tsai, li


# Generate matrices for testing
camera_list = generate_input_data(10)
robot_list = generate_input_data(10)

Rx_tsai, tX_tsai = tsai.calibrate(camera_list, robot_list)
Rx_li, tX_li = li.calibrate(camera_list, robot_list)

print("\n[Tsai] \nCalibrated Rotation Matrix (Rx):\n", Rx_tsai)
print("\n[Li] \nCalibrated Rotation Matrix (Rx):\n", Rx_li)

print("\n[Tsai] \nCalibrated Translation Matrix (tX):\n", tX_tsai)
print("\n[Li] \nCalibrated Translation Matrix (tX):\n", tX_li)
