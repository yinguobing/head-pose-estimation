"""This module provides a face detection implementation backed by SCRFD.
https://github.com/deepinsight/insightface/tree/master/detection/scrfd
"""
import os

import cv2
import numpy as np
import onnxruntime


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class FaceDetector:

    def __init__(self, model_file):
        """Initialize a face detector.

        Args:
            model_file (str): ONNX model file path.
        """
        assert os.path.exists(model_file), f"File not found: {model_file}"

        self.center_cache = {}
        self.nms_threshold = 0.4
        self.session = onnxruntime.InferenceSession(
            model_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # Get model configurations from the model file.
        # What is the input like?
        input_cfg = self.session.get_inputs()[0]
        input_name = input_cfg.name
        input_shape = input_cfg.shape
        self.input_size = tuple(input_shape[2:4][::-1])

        # How about the outputs?
        outputs = self.session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        self.input_name = input_name
        self.output_names = output_names

        # And any key points?
        self._with_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1

        if len(outputs) == 6:
            self._offset = 3
            self._strides = [8, 16, 32]
            self._num_anchors = 2
        elif len(outputs) == 9:
            self._offset = 3
            self._strides = [8, 16, 32]
            self._num_anchors = 2
            self._with_kps = True
        elif len(outputs) == 10:
            self._offset = 5
            self._strides = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(outputs) == 15:
            self._offset = 5
            self._strides = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self._with_kps = True

    def _preprocess(self, image):
        inputs = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = inputs - np.array([127.5, 127.5, 127.5])
        inputs = inputs / 128
        inputs = np.expand_dims(inputs, axis=0)
        inputs = np.transpose(inputs, [0, 3, 1, 2])

        return inputs.astype(np.float32)

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []

        inputs = self._preprocess(img)
        predictions = self.session.run(
            self.output_names, {self.input_name: inputs})

        input_height = inputs.shape[2]
        input_width = inputs.shape[3]
        offset = self._offset

        for idx, stride in enumerate(self._strides):
            scores_pred = predictions[idx]
            bbox_preds = predictions[idx + offset] * stride
            if self._with_kps:
                kps_preds = predictions[idx + offset * 2] * stride

            # Generate the anchors.
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-3:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))

                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

                # solution-1, c style:
                # anchor_centers = np.zeros( (height, width, 2), dtype=np.float32 )
                # for i in range(height):
                #    anchor_centers[i, :, 1] = i
                # for i in range(width):
                #    anchor_centers[:, i, 0] = i

                # solution-2:
                # ax = np.arange(width, dtype=np.float32)
                # ay = np.arange(height, dtype=np.float32)
                # xv, yv = np.meshgrid(np.arange(width), np.arange(height))
                # anchor_centers = np.stack([xv, yv], axis=-1).astype(np.float32)

            # Filter the results by scores and threshold.
            pos_inds = np.where(scores_pred >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores_pred[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self._with_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def _nms(self, detections):
        """None max suppression."""
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            _x1 = np.maximum(x1[i], x1[order[1:]])
            _y1 = np.maximum(y1[i], y1[order[1:]])
            _x2 = np.minimum(x2[i], x2[order[1:]])
            _y2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, _x2 - _x1 + 1)
            h = np.maximum(0.0, _y2 - _y1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect(self, img, threshold=0.5, input_size=None, max_num=0, metric='default'):
        input_size = self.input_size if input_size is None else input_size

        # Rescale the image?
        img_height, img_width, _ = img.shape
        ratio_img = float(img_height) / img_width

        input_width, input_height = input_size
        ratio_model = float(input_height) / input_width

        if ratio_img > ratio_model:
            new_height = input_height
            new_width = int(new_height / ratio_img)
        else:
            new_width = input_width
            new_height = int(new_width * ratio_img)

        det_scale = float(new_height) / img_height
        resized_img = cv2.resize(img, (new_width, new_height))

        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, threshold)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list) / det_scale

        if self._with_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]

        keep = self._nms(pre_det)

        det = pre_det[keep, :]

        if self._with_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]])

            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)

            if metric == 'max':
                values = area
            else:
                # some extra weight on the centering
                values = area - offset_dist_squared * 2.0

            # some extra weight on the centering
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]

            if kpss is not None:
                kpss = kpss[bindex, :]

        return det, kpss

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
                          (bbox[2], bbox[3]), box_color)
            label = f"face: {conf:.2f}"
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (bbox[0], bbox[1] - label_size[1]),
                          (bbox[2], bbox[1] + base_line), box_color, cv2.FILLED)
            cv2.putText(image, label, (bbox[0], bbox[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
