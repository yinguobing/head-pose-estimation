"""
Using Kalman Filter as a point stabilizer to stabiliz a 2D point.
"""
import numpy as np

import cv2


class Stabilizer:
    """Using Kalman filter as a point stabilizer."""

    def __init__(self,
                 state_num=4,
                 measure_num=2,
                 cov_noise=0.0001,
                 cov_measure=0.1):
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
                                                [0, 0, 0, 1]], np.float32) * cov_noise

        self.filter.measurementNoiseCov = np.array([[1, 0],
                                                    [0, 1]], np.float32) * cov_measure

    def update(self, point):
        """Update the filter"""
        # Make kalman prediction
        self.prediction = self.filter.predict()

        # Get new measurement
        self.measurement = np.array([[np.float32(point[0])],
                                     [np.float32(point[1])]])

        # Correct according to mesurement
        self.filter.correct(self.measurement)


def main():
    """Test code"""
    global mp
    mp = np.array((2, 1), np.float32)  # measurement

    def onmouse(k, x, y, s, p):
        global mp
        mp = np.array([[np.float32(x)], [np.float32(y)]])

    cv2.namedWindow("kalman")
    cv2.setMouseCallback("kalman", onmouse)
    kalman = Stabilizer(4, 2)
    frame = np.zeros((480, 640, 3), np.uint8)  # drawing canvas

    while True:
        kalman.update(mp)
        point = kalman.prediction
        cv2.circle(frame, (point[0], point[1]), 2, (0, 255, 0), -1)
        cv2.imshow("kalman", frame)
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break


if __name__ == '__main__':
    main()
