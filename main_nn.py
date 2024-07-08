from packages2.neural_network import NN_Preparing, NN_Training
from packages2.logger_configuration import configure_logger
import numpy as np


if __name__ == "__main__":
    configure_logger()
    reference_folder = 'images/images_26_06_24/objects/refs/'
    results_folder = 'images/images_26_06_24/objects/results/'

    NN_prep = NN_Preparing()
    NN_train = NN_Training()

    table_roi_points = np.array([[35, 100], [15, 360], [600, 380], [580, 90]])

    NN_prep.run(reference_folder, results_folder, table_roi_points)

    NN_train.run_with_grid_search(results_folder)
