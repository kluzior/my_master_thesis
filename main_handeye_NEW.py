from packages2.handeye_eye_in_hand_NEW import HandEyeCalibration


camera_params_path = "CameraParams_inna_kamera_17-07_16-34.npz"

he = HandEyeCalibration(camera_params_path)

files_path =  "images/for_hand_eye_calib/images_27-08_13-57"


# he.run_get_data()




he.run_calibration(files_path)


he.calculate_point_to_robot_base(files_path)