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
from packages2.neural_network import NN_Classification




def process_frame(frame, classifier, image_processor, mlp_model):
    # Example processing: Convert to grayscale
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # objects, coordinates, roi_contours = classifier.prepare_image_from_path(image)
    shape_colors = {
        'None': (0, 0, 0),          #  Black for unknown shape
        'Label1': (255, 0, 0),      # Blue for Label1
        'Label2': (0, 255, 0),      # Green for Label2
        'Label3': (0, 0, 255),      # Red for Label3
        'Label4': (255, 255, 0),    # Cyan for Label4
        'Label5': (255, 255, 255)   # White for Label5
    }
    shape_labels = ['None', 'Label1', 'Label2', 'Label3', 'Label4', 'Label5']

    img_to_show = frame.copy()
    # mtx, distortion = image_processor.load_camera_params('CameraParams.npz')
    # img_to_show = image_processor.undistort_frame(img_to_show, mtx, distortion)


    buimg, objects, coordinates, roi_contours = classifier.prepare_frame(frame)


    # Draw ROI contours 
    cv2.drawContours(img_to_show, roi_contours, -1, (255, 0, 255), 2)
    cv2.putText(img_to_show, "ROI", (roi_contours[0][0][0][0], roi_contours[0][0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)

    # buimg = image_processor.apply_binarization(gray_frame, 'standard')
    for idx, object in enumerate(objects):
        pred = classifier.predict_shape(object, mlp_model)
        shape_label = shape_labels[pred[0]]
        shape_color = shape_colors[shape_label]

        x, y, w, h = coordinates[idx]
        cv2.rectangle(img_to_show, (x, y), (x + w, y + h), shape_color, 2)
        cv2.putText(img_to_show, shape_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, shape_color, 2)

    # Add more processing steps here
    return buimg, img_to_show




def camera_loop(camera_index, frame_queue, ready_queue, my_camera, stop_event):
    while not stop_event.is_set():
        ready_queue.get()
        frame = my_camera.get_frame()
        if frame is None:
            print("Failed to capture frame")
            break
        frame_queue.put(frame)

def processing_loop(frame_queue, ready_queue, stop_event, table_roi_points, mlp_model):
    my_classifier = NN_Classification(table_roi_points)
    my_image_framework = ImageProcessor()
    while not stop_event.is_set():
        frame = frame_queue.get()
        cv2.imshow("Raw Frame", frame)
        processed_frame, image_to_show = process_frame(frame, my_classifier, my_image_framework, mlp_model)
        cv2.imshow("Processed Frame", processed_frame)
        cv2.imshow("Annotated Frame", image_to_show)
        ready_queue.put(True)
        if cv2.waitKey(5) == 27:  # Esc key
            stop_event.set()

    cv2.destroyAllWindows()

if __name__ == "__main__":
### PARAMS&PATHS ###
    model_file_path = 'models/mlp_model_accuracy_1_0000_lbfgs_logistic_constant_batch32_09-07_03-31.joblib'

### INIT ###
    # declaration queues, events etc.
    frame_queue = Queue()
    ready_queue = Queue()
    stop_event = Event()
    
    # declaration objects of my classes
    my_image_framework = ImageProcessor()
    mtx, distortion = my_image_framework.load_camera_params('CameraParams.npz')


    my_camera = Camera(mtx, distortion, 1)
    # get ROI point for processing
    # table_roi_points = my_image_framework.get_roi_points(my_image_framework.undistort_frame(my_camera.get_frame(), mtx, distortion))
    table_roi_points = my_image_framework.get_roi_points(my_camera.get_frame())

    # import MLP model
    mlp_model = joblib.load(model_file_path)




### PROGRAM ###
    ready_queue.put(True)

    camera_thread = Thread(target=camera_loop, args=(1, frame_queue, ready_queue, my_camera, stop_event))
    processing_thread = Thread(target=processing_loop, args=(frame_queue, ready_queue, stop_event, table_roi_points, mlp_model))

    camera_thread.start()
    processing_thread.start()

    camera_thread.join()
    processing_thread.join()
    my_camera.cap.release()