import cv2

# Open the camera at index 1
cap = cv2.VideoCapture(1)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If the frame is captured successfully
    if ret:
        # Display the resulting frame
        cv2.imshow('USB Camera Feed', frame)
    else:
        print("Error: Could not read frame.")
        break

    # Press 'q' on the keyboard to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
