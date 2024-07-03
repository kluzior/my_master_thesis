import cv2

capture = cv2.VideoCapture('http://192.168.100.24:8080/video')

while True:
    _, frame = capture.read()

    scale_percent = 60 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)


    frame_show = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow("test", frame_show)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()


