import numpy as np
import cv2
import socket

def start_communication(host = "192.168.0.1", port = 10000):
    print(f"Start listening on {host}:{port}...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(5)
    c, addr = s.accept()
    if c is not None:
        print("client connected!")
    try:
        msg = c.recv(1024)
        print(msg)
        if msg == b"Hello server, robot here":
            print("Robot requested for data!")
            print("its correct client, connection established")
    except:
        pass
    return c, s

def show_camera(frame_event, frame_storage, stop_event, name = None):
    cap = cv2.VideoCapture(1)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow(f"Live camera view | {name}", frame)
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

def pose_list_to_dict(coords):
    if len(coords) != 6:
        raise ValueError("Input list must have exactly 6 elements.")
    keys = ["x", "y", "z", "Rx", "Ry", "Rz"]
    return {key: value for key, value in zip(keys, coords)}

def joints_list_to_dict(coords):
    if len(coords) != 6:
        raise ValueError("Input list must have exactly 6 elements.")
    keys = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
    return {key: value for key, value in zip(keys, coords)}

def calculate_center_point(contour):
    target_pixel = [0, 0] 
    M = cv2.moments(contour)
    target_pixel[0] = float(M['m10'] / M['m00'])
    target_pixel[1] = float(M['m01'] / M['m00'])
    angle = 0.5 * np.arctan2(2 * M['mu11'], (M['mu20'] - M['mu02']))
    return target_pixel, angle