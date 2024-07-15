from packages2.neural_network import NN_Preparing, NN_Training
from packages2.logger_configuration import configure_logger
from packages2.image_processor import ImageProcessor
import numpy as np
import cv2


if __name__ == "__main__":
    configure_logger("debug_nn")
    reference_folder = 'images/images_26_06_24/objects/refs/'
    results_folder = 'images/images_26_06_24/objects/results/'

    NN_prep = NN_Preparing()
    NN_train = NN_Training()
    image_processor = ImageProcessor()

    table_roi_points = image_processor.get_roi_points(cv2.imread("images/images_26_06_24/objects/refs/object1/001.png"))
    # table_roi_points = np.array([[35, 100], [15, 360], [600, 380], [580, 90]])

    NN_prep.run(reference_folder, results_folder, table_roi_points)

    NN_train.run_with_grid_search(results_folder)
