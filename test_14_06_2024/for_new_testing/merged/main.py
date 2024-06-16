import time
from client_thread import ClientThread
from server_thread import ServerThread
from cmd_generator import CmdGenerator
from robot_positions import RobotPositions

def run_server():
    HOST = "127.0.0.1"
    PORT = 10000
    server_thread = ServerThread(HOST, PORT)
    server_thread.start()
    return server_thread

def run_client():
    HOST = "127.0.0.1"
    PORT = 30002
    client_thread = ClientThread(HOST, PORT)
    client_thread.start()
    return client_thread

if __name__ == "__main__":
    server = run_server()
    client = run_client()

    try:
        while True:
            if server.is_client_connected() and client.is_connected():
                
                # MOVEJ TO WAITING POSE
                client.send(CmdGenerator.movej_gen(RobotPositions.pose_wait_base))
                time.sleep(5)
                
                # MOVEJ TO POSE OVER OBJECT 
                client.send(CmdGenerator.movej_gen(RobotPositions.pose_1))   
                time.sleep(5)

                #MOVEL TO OBJECT & GRIP
                pose = server.send_pose_request()
                if pose:
                    x, y, z, rx, ry, rz = pose
                    ret = server.send_move_command(x, y, z - 0.17)
                if ret:
                    client.send(CmdGenerator.set_output(RobotPositions.vacuum_gripper))
                
                #MOVEL UP WITH OBJECT
                pickup_pose = server.send_pose_request()
                if pickup_pose:
                    x, y, z, rx, ry, rz = pickup_pose
                    ret = server.send_move_command(x, y, z + 0.3)
                if ret:                
                    time.sleep(1)  
                
                # PUT DOWN OBJECT & LEAVE
                if pickup_pose:
                    x, y, z, rx, ry, rz = pickup_pose
                    ret = server.send_move_command(x, y, z)
                if ret:
                    client.send(CmdGenerator.reset_output(RobotPositions.vacuum_gripper))
                
                #MOVEL UP WITHOUT OBJECT
                pose = server.send_pose_request()
                if pose:
                    x, y, z, rx, ry, rz = pose
                    ret = server.send_move_command(x, y, z + 0.17)
                if ret:                
                    time.sleep(1)
                
                # MOVEJ TO WAITING POSE
                client.send(CmdGenerator.movej_gen(RobotPositions.pose_wait_base))
                time.sleep(5)

    except KeyboardInterrupt:
        server.stop()
        client.stop()
        server.join()
        client.join()
