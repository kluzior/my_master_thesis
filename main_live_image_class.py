from packages2.image_processor import ImageProcessor
from packages2.neural_network import NN_Classification
from packages2.camera import Camera
import joblib
import cv2
from threading import Thread, Event
from queue import Queue

class Program ():
    def __init__(self, model_file_path, camera_calib_result_path):
        self.model_file_path = model_file_path
        self.camera_calib_result_path = camera_calib_result_path
        self.frame_queue = Queue()
        self.ready_queue = Queue()
        self.stop_event = Event()
        self.my_image_framework = ImageProcessor()
        self.mtx, self.distortion = self.my_image_framework.load_camera_params(self.camera_calib_result_path)
        self.my_camera = Camera(self.mtx, self.distortion, 1)
        self.table_roi_points = self.my_image_framework.get_roi_points(self.my_camera.get_frame())
        self.mlp_model = joblib.load(self.model_file_path)

    def process_frame(self, frame, classifier, image_processor, mlp_model):
        img_to_show = frame.copy()

        buimg, objects, coordinates, roi_contours = classifier.prepare_frame(frame)

        # Draw ROI contours 
        cv2.drawContours(img_to_show, roi_contours, -1, (255, 0, 255), 2)
        cv2.putText(img_to_show, "ROI", (roi_contours[0][0][0][0], roi_contours[0][0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)

        for idx, object in enumerate(objects):
            pred = classifier.predict_shape(object, mlp_model)
            shape_label = classifier.shape_labels[pred[0]]
            shape_color = classifier.shape_colors[shape_label]

            x, y, w, h = coordinates[idx]
            cv2.rectangle(img_to_show, (x, y), (x + w, y + h), shape_color, 2)
            cv2.putText(img_to_show, shape_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, shape_color, 2)

        return buimg, img_to_show
    def camera_loop(self, camera_index, frame_queue, ready_queue, my_camera, stop_event):
        while not stop_event.is_set():
            ready_queue.get()
            frame = my_camera.get_frame()
            if frame is None:
                print("Failed to capture frame")
                break
            frame_queue.put(frame)

    def processing_loop(self, frame_queue, ready_queue, stop_event, table_roi_points, mlp_model):
        my_classifier = NN_Classification(table_roi_points)
        my_image_framework = ImageProcessor()
        while not stop_event.is_set():
            frame = frame_queue.get()
            cv2.imshow("Raw Frame", frame)
            processed_frame, image_to_show = self.process_frame(frame, my_classifier, my_image_framework, mlp_model)
            cv2.imshow("Processed Frame", processed_frame)
            cv2.imshow("Annotated Frame", image_to_show)
            ready_queue.put(True)
            if cv2.waitKey(5) == 27:  # Esc key
                stop_event.set()

        cv2.destroyAllWindows()

    def run(self):
        self.ready_queue.put(True)
        camera_thread = Thread(target=self.camera_loop, args=(1, self.frame_queue, self.ready_queue, self.my_camera, self.stop_event))
        processing_thread = Thread(target=self.processing_loop, args=(self.frame_queue, self.ready_queue, self.stop_event, self.table_roi_points, self.mlp_model))

        camera_thread.start()
        processing_thread.start()

        camera_thread.join()
        processing_thread.join()
        self.my_camera.cap.release()

if __name__ == "__main__":
    model_file_path = 'models/mlp_model_accuracy_1_0000_lbfgs_logistic_constant_batch32_09-07_03-31.joblib'
    camera_calib_result_path = 'data/results/for_camera_calib/images_05-09_14-51/CameraParams.npz'
    program = Program(model_file_path, camera_calib_result_path)
    program.run()



