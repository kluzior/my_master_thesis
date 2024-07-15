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
import logging
import inspect


class NN_Preparing:
    def __init__(self):
        self.image_processor = ImageProcessor()

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'NN_Preparing({self}) was initialized.')

    def prepare_image(self, image_path, save_path, roi_points=None):
        """
        Preprocesses an image by performing various operations such as loading camera parameters,
        ignoring the background, undistorting the image, applying binarization, finding contours,
        cropping the image, and saving the resulting object.

        Args:
            image_path (str): The path to the input image.
            save_path (str): The path to save the resulting object.
            roi_points (list of tuples, optional): The region of interest points for ignoring the background.

        Returns:
            None
        """
        mtx, distortion = self.image_processor.load_camera_params('CameraParams.npz')
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._logger.error(f'Failed to read {image_path}')
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
            self._logger.error(f'No objects found in {image_path}')

    def process_images_in_folder(self, reference_folder, results_folder, roi_points=None):
        """
        Process images in a given folder and save the processed images to a results folder.

        Args:
            reference_folder (str): The path to the folder containing the reference images.
            results_folder (str): The path to the folder where the processed images will be saved.
            roi_points (list, optional): List of region of interest (ROI) points. Defaults to None.

        Returns:
            None
        """
        for root, dirs, files in os.walk(reference_folder):
            self._logger.info(f'Photo processing at:\n'
                            f'\troot: \t{root}\n'
                            f'\tdirs: \t{dirs}\n'
                            f'\tfiles: \t{files}')
            # Tworzenie analogicznej struktury folderów wewnątrz folderu wynikowego
            relative_path = os.path.relpath(root, reference_folder)
            save_dir = os.path.join(results_folder, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    save_path = os.path.join(save_dir, file)
                    self.prepare_image(file_path, save_path, roi_points)
                    self._logger.debug(f'Processed and saved {file_path} to {save_path}')

    def run(self, reference_folder, results_folder, roi_points=None):
        self.process_images_in_folder(reference_folder, results_folder, roi_points)

class NN_Training:
    def __init__(self):
        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'NN_Training({self}) was initialized.')

    def load_images_from_folder(self, folder, label, image_size=(28, 28)):
        """
        Load images from a specified folder and resize them to a given image size.

        Args:
            folder (str): The path to the folder containing the images.
            label (str): The label to assign to the loaded images.
            image_size (tuple, optional): The desired size of the images. Defaults to (28, 28).

        Returns:
            tuple: A tuple containing two lists - the loaded images and their corresponding labels.
        """
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
        """
        Predicts the shape of an image using a trained model.

        Parameters:
        - image_path (str): The path to the image file.
        - model: The trained model used for prediction.
        - image_size (tuple): The desired size of the image (default: (28, 28)).

        Returns:
        - prediction: The predicted shape of the image.
        """
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
        start_time = datetime.datetime.now()
        func_name = inspect.currentframe().f_code.co_name
        self._logger.info(f'{func_name} started at {start_time} ')

        images_1, labels_1 = self.load_images_from_folder(results_folder + 'object1', label=1)
        images_2, labels_2 = self.load_images_from_folder(results_folder + 'object2', label=2)
        images_3, labels_3 = self.load_images_from_folder(results_folder + 'object3', label=3)
        images_4, labels_4 = self.load_images_from_folder(results_folder + 'object4', label=4)
        images_5, labels_5 = self.load_images_from_folder(results_folder + 'object5', label=5)

        X = np.array(images_1 + images_2 + images_3 + images_4 + images_5)
        y = np.array(labels_1 + labels_2 + labels_3 + labels_4 + labels_5)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        mlp = MLPClassifier(max_iter=3000, random_state=42, early_stopping=True)
        parameter_space = {
            # 'hidden_layer_sizes': [(50,)],            # less values only for script testing
            # 'alpha': [0.0001],
            # 'solver': ['sgd'],
            # 'activation': ['tanh'],
            # 'learning_rate': ['constant'],
            # 'learning_rate_init': [0.001],
            # 'batch_size': [32, 64],

            'hidden_layer_sizes': [(50,), (100,), (50,50), (100,100), (100, 50)],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'activation': ['tanh', 'relu', 'logistic'],
            'learning_rate': ['constant', 'invscaling', 'adaptive'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
        }
        clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=5, verbose=2, return_train_score=True)
        clf.fit(X_train, y_train)

        self._logger.info(f'Best parameters found:\n \t{clf.best_params_} ')

        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self._logger.info(f'Accuracy: {accuracy}')
        self._logger.info(f'\n{classification_report(y_test, y_pred)}')

        model_file_path  = (f"models/{model_file_path}_"
                            f"accuracy_{format(accuracy, '.4f').replace('.', '_')}_"
                            f"{clf.best_params_['solver']}_"                    
                            f"{clf.best_params_['activation']}_"                    
                            f"{clf.best_params_['learning_rate']}_"                    
                            f"batch{clf.best_params_['batch_size']}_"                
                            f"{datetime.datetime.now().strftime("%d-%m_%H-%M")}"                
                            ".joblib")
        
        joblib.dump(clf, model_file_path)

        self._logger.info(f'Model saved at: {os.path.abspath(model_file_path)} ')
        end_time = datetime.datetime.now()
        self._logger.info(f'{func_name} ended at {end_time} ')
        self._logger.info(f'{func_name} took {end_time - start_time} ')

class NN_Classification:
    shape_colors = {
        'None': (0, 0, 0),          # Black for unknown shape
        'Label1': (255, 0, 0),      # Blue for Label1
        'Label2': (0, 255, 0),      # Green for Label2
        'Label3': (0, 0, 255),      # Red for Label3
        'Label4': (255, 255, 0),    # Cyan for Label4
        'Label5': (255, 255, 255)   # White for Label5
    }
    shape_labels = ['None', 'Label1', 'Label2', 'Label3', 'Label4', 'Label5']

    def __init__(self, roi_points):
        self.image_processor = ImageProcessor()
        self.roi_points = roi_points

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'NN_Classification({self}) was initialized.')


    def prepare_image_from_path(self, image_path):
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

    def prepare_frame(self, frame):
        mtx, distortion = self.image_processor.load_camera_params('CameraParams.npz')
        img = frame
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ignore background                
        iimg, roi_contours = self.image_processor.ignore_background(img, self.roi_points)
        # binarize
        buimg = self.image_processor.apply_binarization(iimg, 'standard')
        # find contour + crop
        objects, coordinates, _ = self.image_processor.crop_image(buimg)
        # save
        return buimg, objects, coordinates, roi_contours
    
    def predict_shape(self, img, model, image_size=(28, 28)):
        img = transform.resize(img, image_size)
        img = img.flatten().reshape(1, -1)
        prediction = model.predict(img)
        return prediction