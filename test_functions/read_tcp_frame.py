# Echo client program
import socket
import time as t
import struct

HOST = "192.168.100.29"
PORT = 30003

print("Start listening...")

while True:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(10)
        s.connect((HOST, PORT))
        t.sleep(1.00)
        messageSize = s.recv(4)
        time = s.recv(8)
        qTarget = s.recv(48)
        qdTarget = s.recv(48)
        qddTarget = s.recv(48)
        iTarget = s.recv(48)
        mTarget = s.recv(48) 
        qActual = s.recv(48)
        qdActual = s.recv(48)
        iActual = s.recv(48)
        iControl = s.recv(48)
        toolVectorActual = s.recv(48)
        xToolActual = toolVectorActual[0:7]
        yToolActual = toolVectorActual[8:15]
        zToolActual = toolVectorActual[16:23]
        RxToolActual = toolVectorActual[24:31]
        RyToolActual = toolVectorActual[32:39]
        RzToolActual = toolVectorActual[40:47]

        print("toolVectorActual [bytes]: ", toolVectorActual)
        
        print("xToolActual: ", int.from_bytes(xToolActual, "big"))
        print("yToolActual: ", int.from_bytes(yToolActual, "big"))
        print("zToolActual: ", int.from_bytes(zToolActual, "big"))
        print("RxToolActual: ", int.from_bytes(RxToolActual, "big"))
        print("RyToolActual: ", int.from_bytes(RyToolActual, "big"))
        print("RzToolActual: ", int.from_bytes(RzToolActual, "big"))

        s.close()
    except socket.error as socketerror:
        print("Error: ", socketerror)
    print("Program finish")
