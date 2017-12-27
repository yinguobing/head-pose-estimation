"""Human facial landmark detector based on Convulutional Neural Network."""
import os

import cv2

CWD_PATH = os.getcwd()

DNN_PROTOTXT = 'assets/deploy.prototxt'
DNN_MODEL = 'assets/res10_300x300_ssd_iter_140000.caffemodel'

FACE_NET = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_MODEL)

def get_faceboxes(image=None, threshold=0.5):
    """
    Get the bounding box of faces in image using dnn.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    confidences = []
    faceboxes = []

    FACE_NET.setInput(cv2.dnn.blobFromImage(
        image, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
    detections = FACE_NET.forward()

    for result in detections[0, 0, :, :]:
        confidence = result[2]
        if confidence > threshold:
            x_left_bottom = int(result[3] * cols)
            y_left_bottom = int(result[4] * rows)
            x_right_top = int(result[5] * cols)
            y_right_top = int(result[6] * rows)
            confidences.append(confidence)
            faceboxes.append(
                [x_left_bottom, y_left_bottom, x_right_top, y_right_top])

    return confidences, faceboxes
