from packages2.image_processor import ImageProcessor
from packages2.neural_network import NN_Classification
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage import io, color, transform
import joblib
import os
import cv2
import glob

if __name__ == "__main__":
    table_roi_points = np.array([[42, 55], [15, 350], [610, 375], [600, 65]])

    image_processor = ImageProcessor()
    classifier = NN_Classification(table_roi_points)
    # model_file_path = 'models/MODEL.joblib'
    model_file_path = 'models/MODEL_WITH_GRID_SEARCH.joblib'
    shape_colors = {
        'None': (0, 0, 0),          #  Black for unknown shape
        'Label1': (255, 0, 0),      # Blue for Label1
        'Label2': (0, 255, 0),      # Green for Label2
        'Label3': (0, 0, 255),      # Red for Label3
        'Label4': (255, 255, 0),    # Cyan for Label4
        'Label5': (255, 255, 255)   # White for Label5
    }
    mlp_model = joblib.load(model_file_path)

    shape_labels = ['None', 'Label1', 'Label2', 'Label3', 'Label4', 'Label5']

    mtx, distortion = image_processor.load_camera_params('CameraParams.npz')

    images = glob.glob('images/images_26_06_24/objects/*.png')

    print(f"images: {images}")
    i = 0
    for image in images:
        objects, coordinates, roi_contours = classifier.prepare_image(image)
        img_to_show = cv2.imread(image)
        # Draw ROI contours 
        cv2.drawContours(img_to_show, roi_contours, -1, (255, 0, 255), 2)
        cv2.putText(img_to_show, "ROI", (roi_contours[0][0][0][0], roi_contours[0][0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 2)

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
        cv2.waitKey(1000)

        cv2.imwrite(f'images/images_2606/class_results/{i}.png', img_to_show)
        i+=1