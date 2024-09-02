import socket

def start_communication(host = "192.168.0.1",   port = 10000):
    print("Start listening...")
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