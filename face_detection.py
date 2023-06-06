"""This module provides a face detection implementation backed by YuNet.
The ONNX model file could be find here:
https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet
"""
import numpy as np
import cv2


class FaceDetector:
    """Detect human faces from an image."""

    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        """Initialize a face detector.

        Note the `inputSize` should be set according to the actual input image. You can set this later with `set_input_size()`.

        Args:
            modelPath (str): model file path
            inputSize (list, optional): the input image size (width, height). Defaults to [320, 320].
            confThreshold (float, optional): confidence threshold. Defaults to 0.6.
            nmsThreshold (float, optional): NMS threshold. Defaults to 0.3.
            topK (int, optional): prediction count for NMS. Defaults to 5000.
            backendId (int, optional): DNN backend ID. Defaults to 0.
            targetId (int, optional): DNN target ID. Defaults to 0.
        """
        self._modelPath = modelPath
        self._inputSize = tuple(inputSize)  # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def set_input_size(self, input_size):
        """Set the input size.

        Args:
            input_size (list): input size in (width, height).
        """
        self._model.setInputSize(tuple(input_size))

    def detect(self, image):
        """Run the detection on target image.

        Args:
            image (np.ndarray): a target image

        Returns:
            np.ndarray: detection results
        """
        return self._model.detect(image)[1]

    def visualize(self, image, results, box_color=(0, 255, 0), text_color=(0, 0, 0)):
        """Visualize the detection results.

        Args:
            image (np.ndarray): image to draw marks on.
            results (np.ndarray): face detection results.
            box_color (tuple, optional): color of the face box. Defaults to (0, 255, 0).
            text_color (tuple, optional): color of the face marks (5 points). Defaults to (0, 0, 255).
        """
        for det in results:
            bbox = det[0:4].astype(np.int32)
            conf = det[-1]
            cv2.rectangle(image, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]), box_color)
            label = f"face: {conf:.2f}"
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (bbox[0], bbox[1] - label_size[1]),
                          (bbox[0] + bbox[2],
                           bbox[1] + base_line),
                          box_color, cv2.FILLED)
            cv2.putText(image, label, (bbox[0], bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
