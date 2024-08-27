from packages2.handeye_eye_in_hand_NEW import HandEyeCalibration


camera_params_path = "CameraParams_inna_kamera_17-07_16-34.npz"

he = HandEyeCalibration(camera_params_path)

files_path =  "images/for_hand_eye_calib/images_27-08_13-57"


# he.run_get_data()

# he.run_calibration(files_path)


point1 = he.calculate_point_to_robot_base(files_path, point_shiftt=(0,0))
point2 = he.calculate_point_to_robot_base(files_path, point_shiftt=(1,0))
point3 = he.calculate_point_to_robot_base(files_path, point_shiftt=(0,1))
point4 = he.calculate_point_to_robot_base(files_path, point_shiftt=(1,1))

print(f"point1 (0,0):\n {point1}")
print(f"point2 (1,0):\n {point2}")
print(f"point3 (0,1):\n {point3}")
print(f"point4 (1,1):\n {point4}")