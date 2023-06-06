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

from face_detection import FaceDetector
from mark_detection import MarkDetector
from pose_estimation import PoseEstimator
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
    print(f"Video source: {video_src}")

    # Get the frame size. This will be used by the following detectors.
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup a face detector to detect human faces.
    face_detector = FaceDetector("assets/face_detector.onnx")

    # Setup a mark detector to detect landmarks.
    mark_detector = MarkDetector("assets/face_landmarks.onnx")

    # Setup a pose estimator to solve pose.
    pose_estimator = PoseEstimator(frame_width, frame_height)

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
        faces, _ = face_detector.detect(frame, 0.7)

        # Any valid face found?
        if len(faces) > 0:
            tm.start()

            # Step 2: Detect landmarks. Crop and feed the face area into the
            # mark detector. Note only the first face will be used for
            # demonstration.
            face = refine(faces, frame_width, frame_height, 0.15)[0]
            x1, y1, x2, y2 = face[:4].astype(int)
            patch = frame[y1:y2, x1:x2]

            # Run the mark detection.
            marks = mark_detector.detect([patch])[0].reshape([68, 2])

            # Convert the locations from local face area to the global image.
            marks *= (x2 - x1)
            marks[:, 0] += x1
            marks[:, 1] += y1

            # Step 3: Try pose estimation with 68 points.
            pose = pose_estimator.solve(marks)

            tm.stop()

            # All done. The best way to show the result would be drawing the
            # pose on the frame in realtime.

            # Do you want to see the pose annotation?
            pose_estimator.visualize(frame, pose, color=(0, 255, 0))

            # Do you want to see the axes?
            # pose_estimator.draw_axes(frame, pose)

            # Do you want to see the marks?
            # mark_detector.visualize(frame, marks, color=(0, 255, 0))

            # Do you want to see the face bounding boxes?
            # face_detector.visualize(frame, faces)

        # Draw the FPS on screen.
        cv2.rectangle(frame, (0, 0), (90, 30), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, f"FPS: {tm.getFPS():.0f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Show preview.
        cv2.imshow("Preview", frame)
        if cv2.waitKey(1) == 27:
            break


if __name__ == '__main__':
    run()
