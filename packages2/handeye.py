from packages2.robot import Robot
from packages2.camera import Camera
import cv2 
# import socket
import numpy as np
from packages2.common import RPY_to_rmtx
import logging


class HandEyeCalibration:
    def __init__(self, robot_client):
        self.camera = Camera(1)
        self.robot = Robot(robot_client)

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'HandEyeCalibration({self}) was initialized.')

    def calibrate(self):
        pass 


    def test(self):
        frame = self.camera.get_frame() 
        cv2.imshow('frame', frame) 
        cv2.imwrite('test_frame.png', frame)                        
        cv2.waitKey(1000)

    def run(self):
        images = []
        robot_positions = []
        while self.camera.cap.isOpened():
            img = self.camera.get_frame()
            k = cv2.waitKey(5)
            if k == 27:                                     # wait for 'Esc' key to save
                self._logger.info(f'user pressed Esc key')
                break
            elif k == ord('s'):                             # wait for 's' key to save
                position = self.robot.give_pose()
                robot_positions.append(position)
                images.append(img)

                # save_image(write_path, num, img)   
                # self._logger.debug(f'captured photo no. {num}')
            
            cv2.imshow('Camera', img)

        self.camera.cap.release()   
        cv2.destroyAllWindows()
        self._logger.info(f'captured {len(images)} photos')
        # self._logger.debug(f'exit get_images() program')

        return images, robot_positions

    def prepare_data(self, images, robot_positions):
        chess_datas = []
        robot_datas = []
        image_no = 0
        for image in images:
            self._logger.info(f'Calculating rotation&translation matrix for image no: {image_no}')
            image_no += 1



            pass

        for pos in robot_positions:
            self._logger.info(f'Calculating rotation&translation matrix for robot position: {pos}')
            
            roll = pos[3]
            pitch = pos[4]
            yaw = pos[5]
            matrix = np.zeros((4, 4))

            # Get the 3x3 rotation matrix
            rotation_matrix = RPY_to_rmtx(roll, pitch, yaw)

            matrix[:3, :3] = rotation_matrix
            for i in range(3):
                matrix[i, 3] = pos[i]
            matrix[3, 3] = 1
            self._logger.info(f'Obtained matrix: \n{matrix}')
            robot_datas.append(matrix)
        return chess_datas, robot_datas





# if __name__ == "__main__":
#     HOST = "192.168.0.1"
#     PORT = 10000
#     print("Start listening...")
#     s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#     s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
#     s.bind((HOST, PORT))
#     s.listen(5)
#     c, addr = s.accept()

    
#     configure_logger("debug_handeye")
#     handeye = HandEyeCalibration(c)
#     a, positions  = handeye.run()
#     print(f"{len(a)}")
#     pos_list = [list(pos) for pos in positions]
#     print(pos_list)
#     handeye.prepare_data(a, pos_list)
#     print("Done!") 