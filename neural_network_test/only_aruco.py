import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage import io, color, transform
import joblib
import os
import cv2
import glob

def load_camera_params(path):
    with np.load(path) as file:
        mtx, distortion = [file[i] for i in ('cameraMatrix', 'dist')]
    return mtx, distortion

def undistort_frame(frame, mtx, distortion):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, distortion, (w, h), 1, (w, h))
    undst = cv2.undistort(frame, mtx, distortion, None, newcameramtx)
    x, y, w, h = roi
    undst = undst[y:y+h, x:x+w]
    return undst


def apply_binarization(gray, binarization='standard'):
    if binarization == 'standard' or binarization == 'raw':
        _, mask = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
    elif binarization == 'adaptive':
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    elif binarization == 'canny':
        blur = cv2.GaussianBlur(gray.copy(), (5, 5), 0)
        mask = cv2.Canny(blur, 50, 150)
    elif binarization == 'otsu':
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError("Unknown binarization method")
    return mask

def is_aruco_inside_bbox(x, y, w, h, aruco_corners):
    for corner in aruco_corners:
        print('corner: {corner}')
        i = 0
        for point in corner[0]:
            print(f'point{i}: {point}')
            px, py = point
            print(f'x: {x}')
            print(f'px: {px}')
            print(f'x + w: {x + w}')
            print(f'y: {y}')
            print(f'py: {py}')
            print(f'y + h: {y + h}')
            if x <= int(px) <= x + w and y <= int(py) <= y + h:
                print("TRUE")
                return True
            i +=1
    return False

def crop_image(image, margin=1, image_raw=None, aruco_corners=None):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 600]

    cropped_images = []
    cropped_raw_images = []
    coordinates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        print("NEW CONTOUR")
        # Check if any aruco marker is inside the bounding box
        if aruco_corners is not None:
            if is_aruco_inside_bbox(x, y, w, h, aruco_corners):
                continue

        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        cropped_images.append(masked_image[y:y+h, x:x+w])
        coordinates.append((x, y, w, h))

        if image_raw is not None:
            cropped_raw_images.append(image_raw[y:y+h, x:x+w])

    return cropped_images, coordinates, cropped_raw_images


def prepare_image(image_path):
    # get camera params
    mtx, distortion = load_camera_params('CameraParams.npz')
    # read image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to read {image_path}")
        return
    # undistort
    uimg = undistort_frame(img, mtx, distortion)
    
    # Ignore aruco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    aruco_corners, _, _ = cv2.aruco.detectMarkers(uimg, aruco_dict, parameters=cv2.aruco.DetectorParameters())
    print(f'aruco_corners: {aruco_corners}')    
    # binarize
    buimg = apply_binarization(uimg, 'standard')
    # find contour + crop
    objects, coordinates, _ = crop_image(buimg, aruco_corners=aruco_corners)
    # save
    return objects, coordinates

def detect_aruco(frame, aruco_edge_cm):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)

    aruco_corners, _, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=cv2.aruco.DetectorParameters())

    cv2.polylines(frame, np.intp(aruco_corners), True, (0, 255, 0), 5)      # mark detected aruco

    pixel_cm_ratio = cv2.arcLength(aruco_corners[0], True) / (4*aruco_edge_cm)

    return pixel_cm_ratio

def predict_shape(img, model, image_size=(28, 28)):
    # img = io.imread(image_path, as_gray=True)
    img = transform.resize(img, image_size)
    img = img.flatten().reshape(1, -1)
    prediction = model.predict(img)
    return prediction

model_file_path = 'mlp_model.joblib'
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
print(mlp_model.coefs_)  # Współczynniki
print(mlp_model.intercepts_)  # Przechwytywanie

shape_labels = ['None', 'Label1', 'Label2', 'Label3', 'Label4', 'Label5']

mtx, distortion = load_camera_params('CameraParams.npz')

images = glob.glob('neural_network_test/new/images/*.png')
print(images)

for image in images:
    objects, coordinates = prepare_image(image)
    img_to_show = cv2.imread(image)
    img_to_show = undistort_frame(img_to_show, mtx, distortion)

    for idx, object in enumerate(objects):
        pred = predict_shape(object, mlp_model)
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
    cv2.destroyAllWindows()