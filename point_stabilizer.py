"""
Using Kalman Filter as a point stabilizer to stabiliz a 2D point.
"""
import numpy as np

import cv2


class Stabilizer:
    """Using Kalman filter as a point stabilizer."""

    def __init__(self, state_num=4, measure_num=2):
        """Initialization"""
        # The filter itself.
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)

        # # Store the state (x,y,detaX,detaY)
        # self.state = np.zeros((state_num, 1), dtype=np.float)

        # # Store the measurement result.
        self.measurement = np.array((2, 1), np.float32)

        # # The prediction.
        self.prediction = np.zeros((2, 1), np.float32)

        # state = 0.1 * np.random.randn(4, 2)
        self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)

        self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)

        self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * 0.01

        self.filter.measurementNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * 0.0001

    def update(self, point):
        """Update the filter"""
        # Get new measurement
        self.measurement = np.array([[np.float32(point[0])],
                                     [np.float32(point[1])]])

        # Correct according to mesurement
        self.filter.correct(self.measurement)

        # Make kalman prediction
        self.prediction = self.filter.predict()
