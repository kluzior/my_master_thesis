from packages2.image_processor import ImageProcessor
from packages2.neural_network import NN_Classification
from packages2.camera import Camera
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage import io, color, transform
import joblib
import os
import cv2
import glob
from threading import Thread, Event
from queue import Queue




def process_frame(frame):
    # Example processing: Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Add more processing steps here
    return gray_frame




def camera_loop(camera_index, frame_queue, ready_queue, my_camera, stop_event):
    while not stop_event.is_set():
        ready_queue.get()
        frame = my_camera.get_frame()
        if frame is None:
            print("Failed to capture frame")
            break
        frame_queue.put(frame)

def processing_loop(frame_queue, ready_queue, stop_event):
    while not stop_event.is_set():
        frame = frame_queue.get()
        cv2.imshow("Raw Frame", frame)
        processed_frame = process_frame(frame)
        cv2.imshow("Processed Frame", processed_frame)
        ready_queue.put(True)
        if cv2.waitKey(5) == 27:  # Esc key
            stop_event.set()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    frame_queue = Queue()
    ready_queue = Queue()
    stop_event = Event()
    my_camera = Camera(1)

    ready_queue.put(True)

    camera_thread = Thread(target=camera_loop, args=(1, frame_queue, ready_queue, my_camera, stop_event))
    processing_thread = Thread(target=processing_loop, args=(frame_queue, ready_queue, stop_event))

    camera_thread.start()
    processing_thread.start()

    camera_thread.join()
    processing_thread.join()
    my_camera.cap.release()