"""A module provides a bunch of helper functions."""
import numpy as np


def refine(boxes, max_width, max_height, shift=0.1):
    """Refine the face boxes to suit the face landmark detection's needs.

    Args:
        boxes: [[x1, y1, x2, y2], ...]
        max_width: Value larger than this will be clipped.
        max_height: Value larger than this will be clipped.
        shift (float, optional): How much to shift the face box down. Defaults to 0.1.

    Returns:
       Refined results.
    """
    refined = boxes.copy()
    width = refined[:, 2] - refined[:, 0]
    height = refined[:, 3] - refined[:, 1]

    # Move the boxes in Y direction
    shift = height * shift
    refined[:, 1] += shift
    refined[:, 3] += shift
    center_x = (refined[:, 0] + refined[:, 2]) / 2
    center_y = (refined[:, 1] + refined[:, 3]) / 2

    # Make the boxes squares
    square_sizes = np.maximum(width, height)
    refined[:, 0] = center_x - square_sizes / 2
    refined[:, 1] = center_y - square_sizes / 2
    refined[:, 2] = center_x + square_sizes / 2
    refined[:, 3] = center_y + square_sizes / 2

    # Clip the boxes for safety
    refined[:, 0] = np.clip(refined[:, 0], 0, max_width)
    refined[:, 1] = np.clip(refined[:, 1], 0, max_height)
    refined[:, 2] = np.clip(refined[:, 2], 0, max_width)
    refined[:, 3] = np.clip(refined[:, 3], 0, max_height)

    return refined
