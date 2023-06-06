"""A module provides a bunch of helper functions."""
import numpy as np


def refine(boxes, max_width, max_height, shift=0.1):
    refined = boxes.copy()

    # Move the boxes in Y direction
    shift = refined[:, 3] * shift
    refined[:, 3] += shift
    center_x = refined[:, 0] + refined[:, 2] / 2
    center_y = refined[:, 1] + refined[:, 3] / 2

    # Make the boxes squares
    _square_sizes = np.maximum(refined[:, 2], refined[:, 3])
    refined[:, 0] = center_x - _square_sizes / 2
    refined[:, 1] = center_y - _square_sizes / 2
    refined[:, 2] = _square_sizes
    refined[:, 3] = _square_sizes

    # Clip the boxes for safety
    np.clip(refined[:, 0], 0, max_width)
    np.clip(refined[:, 1], 0, max_height)
    np.clip(refined[:, 2], 0, max_width)
    np.clip(refined[:, 3], 0, max_height)

    return refined
