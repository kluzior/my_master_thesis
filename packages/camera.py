import logging

class Camera:
    """Parent class representing a camera"""

    def __init__(self, index):
        self.index = index
        self.matrix = None
        self.distortion = None

        self._logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self._logger.debug(f'Camera({self}) was initialized.')
        

    def set_calibration(self, matrix, distortion):
        self.matrix = matrix
        self.distortion = distortion
        self._logger.debug(f'new camera parameters was saved!')
