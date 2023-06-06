"""Human facial landmark detector based on Convolutional Neural Network."""
import os

import cv2
import numpy as np
import onnxruntime as ort


class MarkDetector:
    """Facial landmark detector by Convolutional Neural Network"""

    def __init__(self, model_file):
        """Initialize a mark detector.

        Args:
            model_file (str): ONNX model path.
        """
        assert os.path.exists(model_file), f"File not found: {model_file}"
        self._input_size = 128
        self.model = ort.InferenceSession(
            model_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    def _preprocess(self, bgrs):
        """Preprocess the inputs to meet the model's needs.

        Args:
            bgrs (np.ndarray): a list of input images in BGR format.

        Returns:
            tf.Tensor: a tensor
        """
        rgbs = []
        for img in bgrs:
            img = cv2.resize(img, (self._input_size, self._input_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgbs.append(img)

        return rgbs

    def detect(self, images):
        """Detect facial marks from an face image.

        Args:
            images: a list of face images.

        Returns:
            marks: the facial marks as a numpy array of shape [Batch, 68*2].
        """
        inputs = self._preprocess(images)
        marks = self.model.run(["dense_1"], {"image_input": inputs})
        return np.array(marks)

    def visualize(self, image, marks, color=(255, 255, 255)):
        """Draw mark points on image"""
        for mark in marks:
            cv2.circle(image, (int(mark[0]), int(
                mark[1])), 1, color, -1, cv2.LINE_AA)
