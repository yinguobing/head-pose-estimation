"""Human facial landmark detector based on Convulutional Neural Network."""
import os

import numpy as np
import tensorflow as tf

import cv2

CWD_PATH = os.getcwd()

DNN_PROTOTXT = 'assets/deploy.prototxt'
DNN_MODEL = 'assets/res10_300x300_ssd_iter_140000.caffemodel'
FACE_NET = cv2.dnn.readNetFromCaffe(DNN_PROTOTXT, DNN_MODEL)

CNN_INPUT_SIZE = 128
MARK_MODEL = 'assets/frozen_inference_graph.pb'


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


def draw_all_result(image, confidences, faceboxes):
    """Draw the detection result on image"""
    for result in zip(confidences, faceboxes):
        conf = result[0]
        facebox = result[1]

        cv2.rectangle(image, (facebox[0], facebox[1]),
                      (facebox[2], facebox[3]), (0, 255, 0))
        label = "face: %.4f" % conf
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),
                      (facebox[0] + label_size[0],
                       facebox[1] + base_line),
                      (0, 255, 0), cv2.FILLED)
        cv2.putText(image, label, (facebox[0], facebox[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


def draw_box(image, faceboxes, box_color=(255, 255, 255)):
    """Draw square boxes on image"""
    for facebox in faceboxes:
        cv2.rectangle(image, (facebox[0], facebox[1]),
                      (facebox[2], facebox[3]), box_color)


def move_box(box, offset):
    """Move the box to direction specified by vector offset"""
    left_x = box[0] + offset[0]
    top_y = box[1] + offset[1]
    right_x = box[2] + offset[0]
    bottom_y = box[3] + offset[1]
    return [left_x, top_y, right_x, bottom_y]


def get_square_box(box):
    """Get a square box out of the given box, by expanding it."""
    left_x = box[0]
    top_y = box[1]
    right_x = box[2]
    bottom_y = box[3]

    box_width = right_x - left_x
    box_height = bottom_y - top_y

    # Check if box is already a square. If not, make it a square.
    diff = box_height - box_width
    delta = int(abs(diff) / 2)

    if diff == 0:                   # Already a square.
        return box
    elif diff > 0:                  # Height > width, a slim box.
        left_x -= delta
        right_x += delta
        if diff % 2 == 1:
            right_x += 1
    else:                           # Width > height, a short box.
        top_y -= delta
        bottom_y += delta
        if diff % 2 == 1:
            bottom_y += 1

    # Make sure box is always square.
    assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

    return [left_x, top_y, right_x, bottom_y]


def box_in_image(box, image):
    """Check if the box is in image"""
    rows = image.shape[0]
    cols = image.shape[1]
    return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows


def extract_cnn_facebox(image):
    """Extract face area from image."""
    _, raw_boxes = get_faceboxes(image=image, threshold=0.9)

    for box in raw_boxes:
        # Move box down.
        diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
        offset_y = int(abs(diff_height_width / 2))
        box_moved = move_box(box, [0, offset_y])

        # Make box square.
        facebox = get_square_box(box_moved)

        if box_in_image(facebox, image):
            return facebox

    return None


def get_tf_session(inference_pb_file=MARK_MODEL):
    """Get a TensorFlow session ready to do landmark detection"""
    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(inference_pb_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        return detection_graph, tf.Session(graph=detection_graph)


MARK_GRAPH, MARK_SESS = get_tf_session(inference_pb_file=MARK_MODEL)


def detect_marks(image_np, sess, detection_graph):
    """Detect marks from image"""
    # Get result tensor by its name.
    logits_tensor = detection_graph.get_tensor_by_name('logits/BiasAdd:0')

    # Actual detection.
    predictions = sess.run(
        logits_tensor,
        feed_dict={'input_image_tensor:0': image_np})

    # Convert predictions to landmarks.
    marks = np.array(predictions).flatten()
    marks = np.reshape(marks, (-1, 2))

    return marks


def draw_marks(image, marks, color=None):
    """Draw mark points on image"""
    if color is None:
        color = (255, 255, 255)
    for mark in marks:
        cv2.circle(image, (int(mark[0]), int(
            mark[1])), 1, color, -1, cv2.LINE_AA)
