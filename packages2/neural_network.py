from packages2.image_processor import ImageProcessor
import cv2
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from skimage import io, color, transform
import joblib
import datetime


class NN_Preparing:

    def __init__(self):
        self.image_processor = ImageProcessor()

    def prepare_image(self, image_path, save_path, roi_points=None):
        mtx, distortion = self.image_processor.load_camera_params('CameraParams.npz')
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read {image_path}")
            return
        # ignore background
        iimg, _ = self.image_processor.ignore_background(img, roi_points)
        # undistort
        uimg = self.image_processor.undistort_frame(iimg, mtx, distortion)
        # binarize
        buimg = self.image_processor.apply_binarization(uimg, 'standard')
        # find contour + crop
        objects, _, _ = self.image_processor.crop_image(buimg, margin=2)
        # save
        if objects:
            cv2.imwrite(save_path, objects[0])
        else:
            print(f"No objects found in {image_path}")

    def process_images_in_folder(self, reference_folder, results_folder, roi_points=None):
        for root, dirs, files in os.walk(reference_folder):
            print(f'root: {root}')
            print(f'dirs: {dirs}')
            print(f'files: {files}')
            # Tworzenie analogicznej struktury folderów wewnątrz folderu wynikowego
            relative_path = os.path.relpath(root, reference_folder)
            save_dir = os.path.join(results_folder, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    save_path = os.path.join(save_dir, file)
                    self.prepare_image(file_path, save_path, roi_points)
                    print(f"Processed and saved {file_path} to {save_path}")

    def run(self, reference_folder, results_folder, roi_points=None):
        self.process_images_in_folder(reference_folder, results_folder, roi_points)


class NN_Training:

    def __init__(self):
        pass

    def load_images_from_folder(self, folder, label, image_size=(28, 28)):
        images = []
        labels = []
        for filename in os.listdir(folder):
            img = io.imread(os.path.join(folder, filename), as_gray=True)
            if img is not None:
                img = transform.resize(img, image_size)
                img = img.flatten()
                images.append(img)
                labels.append(label)
        return images, labels

    def predict_shape(image_path, model, image_size=(28, 28)):
        img = io.imread(image_path, as_gray=True)
        img = transform.resize(img, image_size)
        img = img.flatten().reshape(1, -1)
        prediction = model.predict(img)
        return prediction

    def run(self, results_folder, model_file_path='mlp_model'):
        if model_file_path == 'mlp_model':
            model_file_path = f"{model_file_path}_{datetime.datetime.now().strftime("%d-%m_%H-%M")}"
        images_1, labels_1 = self.load_images_from_folder(results_folder + 'object1', label=1)
        images_2, labels_2 = self.load_images_from_folder(results_folder + 'object2', label=2)
        images_3, labels_3 = self.load_images_from_folder(results_folder + 'object3', label=3)
        images_4, labels_4 = self.load_images_from_folder(results_folder + 'object4', label=4)
        images_5, labels_5 = self.load_images_from_folder(results_folder + 'object5', label=5)
        X = np.array(images_1 + images_2 + images_3 + images_4 + images_5)
        y = np.array(labels_1 + labels_2 + labels_3 + labels_4 + labels_5)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, solver='adam', random_state=42)
        mlp.fit(X_train, y_train)

        y_pred = mlp.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        model_file_path = f"models/{model_file_path}.joblib"
        joblib.dump(mlp, model_file_path)
        print("Model saved at:", os.path.abspath(model_file_path))

    def run_with_grid_search(self, results_folder, model_file_path='mlp_model'):
        if model_file_path == 'mlp_model':
            model_file_path = f"{model_file_path}_{datetime.datetime.now().strftime("%d-%m_%H-%M")}"        
        images_1, labels_1 = self.load_images_from_folder(results_folder + 'object1', label=1)
        images_2, labels_2 = self.load_images_from_folder(results_folder + 'object2', label=2)
        images_3, labels_3 = self.load_images_from_folder(results_folder + 'object3', label=3)
        images_4, labels_4 = self.load_images_from_folder(results_folder + 'object4', label=4)
        images_5, labels_5 = self.load_images_from_folder(results_folder + 'object5', label=5)

        X = np.array(images_1 + images_2 + images_3 + images_4 + images_5)
        y = np.array(labels_1 + labels_2 + labels_3 + labels_4 + labels_5)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlp = MLPClassifier(max_iter=3000, random_state=42)
        parameter_space = {
            'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100)],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'solver': ['sgd', 'adam', 'lbfgs'],
        }
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, verbose=2)
        clf.fit(X_train, y_train)

        print('Best parameters found:\n', clf.best_params_)

        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        joblib.dump(clf, model_file_path)
        print("Model saved at:", os.path.abspath(model_file_path))


class NN_Classification:
    def __init__(self, roi_points):
        self.image_processor = ImageProcessor()
        self.roi_points = roi_points

    def prepare_image(self, image_path):
        mtx, distortion = self.image_processor.load_camera_params('CameraParams.npz')
        # read image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Failed to read {image_path}")
            return
        # ignore background                
        iimg, roi_contours = self.image_processor.ignore_background(img, self.roi_points)
        # undistort
        uimg = self.image_processor.undistort_frame(iimg, mtx, distortion)
        # binarize
        buimg = self.image_processor.apply_binarization(uimg, 'standard')
        # find contour + crop
        objects, coordinates, _ = self.image_processor.crop_image(buimg)
        # save
        return objects, coordinates, roi_contours

    def predict_shape(self, img, model, image_size=(28, 28)):
        img = transform.resize(img, image_size)
        img = img.flatten().reshape(1, -1)
        prediction = model.predict(img)
        return prediction