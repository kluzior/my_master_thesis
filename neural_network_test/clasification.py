from packages2.image_processor import ImageProcessor
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage import io, color, transform
import joblib
import os
import cv2
import glob


class NN_Classification:
    def __init__(self):
        self.image_processor = ImageProcessor()

    def prepare_image(self, image_path):
        mtx, distortion = self.image_processor.load_camera_params('CameraParams.npz')
        # read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read {image_path}")
            return
        # undistort
        uimg = self.image_processor.undistort_frame(img, mtx, distortion)
        
        # Ignore aruco
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        aruco_corners, _, _ = cv2.aruco.detectMarkers(uimg, aruco_dict, parameters=cv2.aruco.DetectorParameters())
        print(f'aruco_corners: {aruco_corners}')    
        # binarize
        buimg = self.image_processor.apply_binarization(uimg, 'standard')
        # find contour + crop
        objects, coordinates, _ = self.image_processor.crop_image(buimg, aruco_corners=aruco_corners)
        # save
        return objects, coordinates

    def predict_shape(img, model, image_size=(28, 28)):
        # img = io.imread(image_path, as_gray=True)
        img = transform.resize(img, image_size)
        img = img.flatten().reshape(1, -1)
        prediction = model.predict(img)
        return prediction


if __name__ == "__main__":

    image_processor = ImageProcessor()
    classifier = NN_Classification()
    model_file_path = 'models/MODEL.joblib'
    shape_colors = {
        'None': (255, 255, 255),  # White for unknown shape
        'Label1': (255, 0, 0),      # Blue for Label1
        'Label2': (0, 255, 0),      # Green for Label2
        'Label3': (0, 0, 255),      # Red for Label3
        'Label4': (255, 255, 0),    # Cyan for Label4
        'Label5': (255, 0, 255)     # Magenta for Label5
    }
    mlp_model = joblib.load(model_file_path)
    # pred = mlp_model.predict(X_test)
    # print(mlp_model.coefs_)  # Współczynniki
    # print(mlp_model.intercepts_)  # Przechwytywanie

    shape_labels = ['None', 'Label1', 'Label2', 'Label3', 'Label4', 'Label5']

    mtx, distortion = image_processor.load_camera_params('CameraParams.npz')

    # images = glob.glob('neural_network_test/new/images_2606/*.png')
    images = glob.glob('images/images_2606/objects/*.png')

    print(f"images: {images}")
    i = 0
    for image in images:
        objects, coordinates = classifier.prepare_image(image)
        img_to_show = cv2.imread(image)
        img_to_show = image_processor.undistort_frame(img_to_show, mtx, distortion)
        for idx, object in enumerate(objects):
            pred = classifier.predict_shape(object, mlp_model)
            shape_label = shape_labels[pred[0]]
            shape_color = shape_colors[shape_label]

            # Draw rectangle around object with color based on predicted shape
            x, y, w, h = coordinates[idx]
            cv2.rectangle(img_to_show, (x, y), (x + w, y + h), shape_color, 2)

            # Annotate with predicted shape label using same color
            cv2.putText(img_to_show, shape_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, shape_color, 2)

        # Show or save the image with annotations
        cv2.imshow("Annotated Image", img_to_show)
        cv2.waitKey(1000)  # Wait for any key press to close the image window
        # cv2.destroyAllWindows()

        cv2.imwrite(f'images/images_2606/class_results/{i}.png', img_to_show)
        i+=1