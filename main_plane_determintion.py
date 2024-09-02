from packages2.plane_determination import PlaneDeterminator
from packages2.image_processor import ImageProcessor
from packages2.handeye_eye_in_hand_NEW import HandEyeCalibration
import cv2
from packages2.start_communication import start_communication
from packages2.robot_functions import RobotFunctions
from packages2.common import rvec_tvec2pose
camera_params_path = "data/results/for_camera_calib/images_28-08_10-27/CameraParams.npz"

c, s = start_communication()
robot_functions = RobotFunctions(c)


# pd = PlaneDeterminator(camera_params_path)
# ip = ImageProcessor(camera_params_path)

# frame = cv2.imread("z_captured_image.jpg")
# tvecs_list, pixels2D_list = pd.get_plane_points(uframe)
# plane = pd.determine_plane(tvecs_list)

# pixel = (100, 150)  # Example pixel coordinates
# point_3d = pd.pixel_to_plane(pixel, plane)
# print(f"3D point on the plane: {point_3d}")

# Przygotowanie danych
camera_params_path = "data/results/for_camera_calib/images_28-08_10-27/CameraParams.npz"
handeye_path = "data/results/for_hand_eye_calib/images_02-09_10-26"
pd = PlaneDeterminator(camera_params_path, handeye_path)
ip = ImageProcessor(camera_params_path)
he = HandEyeCalibration(camera_params_path, c)
frame = cv2.imread("z_captured_image.jpg")
mtx, dist = ip.load_camera_params(camera_params_path)
uframe = ip.undistort_frame(frame, mtx, dist)

normal_vector, point_on_plane = pd.get_plane_points(uframe)


rvec, _, pixel = ip.calculate_rvec_tvec(uframe, point_shift=(0,0))
pixel_tuple = tuple(pixel[0])

print(f"pixel {pixel}")
print(f"pixel_tuple {pixel_tuple}")
intersection_point = pd.pixel_to_camera_plane(pixel_tuple)

print(f"Współrzędne 3D punktu przecięcia dla piksela {pixel_tuple}: {intersection_point}")

pose = rvec_tvec2pose(rvec, intersection_point)
print(f"pose: {pose}")

rmtx, _ = cv2.Rodrigues(rvec)
target_pose = he.calculate_point_pose2robot_base(rmtx, intersection_point.reshape(-1, 1), handeye_path)


robot_functions.moveJ_pose(target_pose)

