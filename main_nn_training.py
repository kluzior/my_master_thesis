from packages2.neural_network import NN_Preparing, NN_Training
from packages2.logger_configuration import configure_logger
from packages2.image_processor import ImageProcessor
import numpy as np
import cv2


if __name__ == "__main__":
    configure_logger("_", path="debug_nn_xxx")
    reference_folder = 'images/images_26_06_24/objects/refs/'
    # results_folder = 'images/images_26_06_24/objects/results/'
    results_folder = 'images/recznie_posortowane/'

    reference_folder2 = 'images_for_new_training/refs/'
    results_folder2 = 'images_for_new_training/results/'
    NN_prep = NN_Preparing()
    NN_train = NN_Training()
    image_processor = ImageProcessor()

    # table_roi_points = image_processor.get_roi_points(cv2.imread("images/images_26_06_24/objects/refs/object1/001.png"))
    # table_roi_points2 = image_processor.get_roi_points(cv2.imread("images_for_new_training/refs/object1/0007.jpg"))

    # NN_prep.run(reference_folder, results_folder, table_roi_points)
    # NN_prep.run(reference_folder2, results_folder2, table_roi_points2)

    NN_train.run_with_random_search(results_folder, results_folder2)
