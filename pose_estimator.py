"""Estimate head pose according to the facial landmarks"""
import cv2
import numpy as np

# Image size
SIZE = (480, 640)

# 3D model points.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corne
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

# Camera internals
FOCAL_LENGTH = SIZE[1]
CAMERA_CENTER = (SIZE[1] / 2, SIZE[0] / 2)
CAMERA_MATRIX = np.array(
    [[FOCAL_LENGTH, 0, CAMERA_CENTER[0]],
     [0, FOCAL_LENGTH, CAMERA_CENTER[1]],
     [0, 0, 1]], dtype="double"
)

# Assuming no lens distortion
DIST_COEFFS = np.zeros((4, 1))


def solve_pose(image_points):
    """
    Solve pose from image points
    Return (rotation_vector, translation_vector) as pose.
    """
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS)

    return (rotation_vector, translation_vector)


def draw_annotation_box(image, rotation_vector, translation_vector):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    rear_size = 300
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 400
    front_depth = 400
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      CAMERA_MATRIX,
                                      DIST_COEFFS)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(point_2d[6]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(point_2d[7]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(point_2d[8]), (255, 255, 255), 1, cv2.LINE_AA)
    return image


def get_pose_marks(marks):
    """Get marks ready for pose estimation from 68 marks"""
    pose_marks = []
    pose_marks.append(marks[30])    # Nose tip
    pose_marks.append(marks[8])     # Chin
    pose_marks.append(marks[36])    # Left eye left corner
    pose_marks.append(marks[45])    # Right eye right corner
    pose_marks.append(marks[48])    # Left Mouth corner
    pose_marks.append(marks[54])    # Right mouth corner
    return pose_marks
