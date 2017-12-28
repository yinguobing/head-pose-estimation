"""
Using Kalman Filter as a point stabilizer to stabiliz a 2D point.
"""
import numpy as np

import cv2


class Stabilizer():
    """Using Kalman filter as a point stabilizer."""

    def __init__(self, measure_num=2, state_num=4):
        """Initialization"""
        # Store the measurement result.
        self.measure_num = measure_num
        self.measurement = np.zeros((self.measure_num, 1), dtype=float)

        # Store the state
        self.state_num = state_num
        self.state = np.zeros((self.state_num, 1), dtype=float)
