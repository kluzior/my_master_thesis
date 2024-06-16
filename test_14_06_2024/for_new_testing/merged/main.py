import time
from client_thread import ClientThread
from server_thread import RobotServer

def run_server():
    HOST = "127.0.0.1"
    PORT = 10000
    server = RobotServer(HOST, PORT)
    server.start()
    return server

def run_client():
    host = "127.0.0.1"
    port = 30002  # Ensure this matches the server's port

    client_thread = ClientThread(host, port)
    client_thread.start()
    return client_thread

if __name__ == "__main__":
    server = run_server()
    client = run_client()

    try:
        while True:
            # Interact with the server
            pose = server.send_pose_request()
            if pose:
                x, y, z, rx, ry, rz = pose
                server.send_move_command(x, y, z - 0.17)

            time.sleep(2)  # Give some time before sending the next command

            # Interact with the client
            client.send("give_pose")
            time.sleep(1)
            response = client.receive()
            if response:
                print(f"Client received: {response}")

            time.sleep(2)  # Give some time before the next interaction
    except KeyboardInterrupt:
        server.stop()
        client.stop()
        server.join()
        client.join()
