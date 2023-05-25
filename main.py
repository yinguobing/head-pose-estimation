"""Demo code showing how to estimate human head pose.

There are three major steps:
1. Detect and crop the human faces in the video frame.
2. Run facial landmark detection on the face image.
3. Estimate the pose by solving a PnP problem.

For more details, please refer to:
https://github.com/yinguobing/head-pose-estimation
"""
from argparse import ArgumentParser

import cv2
import numpy as np

from face_detection import FaceDetector
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from utils import refine

# Parse arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=0,
                    help="The webcam index.")
args = parser.parse_args()


print(__doc__)
print("OpenCV version: {}".format(cv2.__version__))


def run():
    # Before estimation started, there are some startup works to do.

    # Initialize the video source from webcam or video file.
    video_src = args.cam if args.video is None else args.video
    cap = cv2.VideoCapture(video_src)

    # Get the frame size. This will be used by the following detectors.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup a face detector to detect human faces.
    face_detector = FaceDetector(
        "assets/face_detection_yunet_2022mar.onnx", [frame_width, frame_height])

    # Setup a mark detector to detect landmarks.
    mark_detector = MarkDetector("assets/facemarks.onnx")

    # Setup a pose estimator to solve pose.
    pose_estimator = PoseEstimator(img_size=(frame_height, frame_width))

    # Measure the performance with a tick meter.
    tm = cv2.TickMeter()

    # Now, let the frames flow.
    while True:

        # Read a frame.
        frame_got, frame = cap.read()
        if frame_got is False:
            break

        # If the frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Step 1: Get faces from current frame.
        faces = face_detector.detect(frame)

        # Any face found?
        if faces is not None:
            tm.start()

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector.
            refined = refine(faces, frame_width, frame_height)
            num_faces = refined.shape[0]
            patches = []
            for face in refined:
                x, y, w, h = face[:4].astype(int)
                patch = frame[y:y+h, x:x+w]
                patches.append(patch)

            # Run the detection.
            marks = mark_detector.detect(patches)

            # Convert the locations from local face area to the global image.
            marks *= refined[:, 2, None]
            marks = marks.reshape([num_faces, 68, 2])
            marks[:, :, 0] += refined[:, 0, None]
            marks[:, :, 1] += refined[:, 1, None]

            # Step 3: Try pose estimation with 68 points.
            poses = [pose_estimator.solve_pose_by_68_points(m) for m in marks]

            tm.stop()

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            for pose in poses:
                pose_estimator.draw_annotation_box(
                    frame, pose[0], pose[1], color=(0, 255, 0))

                # Do you want to see the axes?
                pose_estimator.draw_axes(frame, pose[0], pose[1])

            # Do you want to see the marks?
            for m in marks:
                mark_detector.visualize(frame, m, color=(0, 255, 0))

            # Do you want to see the face bounding boxes?
            face_detector.visualize(frame, refined)

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    run()
