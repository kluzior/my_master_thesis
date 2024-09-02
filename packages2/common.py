import math
import numpy as np
from numpy.linalg import inv, svd, norm, pinv
from scipy.spatial.transform import Rotation as Rot
import cv2


def show_camera(frame_event, frame_storage, stop_event):
    cap = cv2.VideoCapture(1)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Live camera view", frame)
        if frame_event.is_set():
            frame_storage['frame'] = frame
            frame_event.clear()
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def pose2rvec_tvec(pose):
    tvec = np.array(pose[:3])
    rvec = np.array(pose[3:])
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    return rotation_matrix, rvec, tvec.reshape(-1, 1)

def rvec_tvec2pose(rvec, tvec):
    x, y, z = tvec.flatten()
    Rx, Ry, Rz = rvec.flatten()
    pose = np.array([x, y, z, Rx, Ry, Rz])
    return pose