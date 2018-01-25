"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
import numpy as np

import cv2
from mark_detector import MarkDetector
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

CNN_INPUT_SIZE = 128


def main():
    """MAIN"""
    # Get frame from webcam or video file
    video_src = 0
    cam = cv2.VideoCapture(video_src)

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Introduce pose estimator to solve pose.
    pose_estimator = PoseEstimator(img_size=(480, 640))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cam.read()
        if frame_got is False:
            break

        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            frame = cv2.flip(frame, 2)

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose
        facebox = mark_detector.extract_cnn_facebox(frame)
        if facebox is not None:
            # Detect landmarks from image of 128x128.
            face_img = frame[facebox[1]: facebox[3],
                             facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            marks = mark_detector.detect_marks(face_img)

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # Uncomment following line to show raw marks.
            # mark_detector.draw_marks(
            #     frame_cnn, marks, color=(0, 255, 0))

            # Try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)

            # Stabilize the pose.
            stabile_pose = []
            pose_np = np.array(pose).flatten()
            for value, ps_stb in zip(pose_np, pose_stabilizers):
                ps_stb.update([value])
                stabile_pose.append(ps_stb.state[0])
            stabile_pose = np.reshape(stabile_pose, (-1, 3))

            # Uncomment following line to draw pose annotaion on frame.
            # pose_estimator.draw_annotation_box(
            #     frame, pose[0], pose[1], color=(255, 128, 128))

            # Uncomment following line to draw stabile pose annotaion on frame.
            pose_estimator.draw_annotation_box(
                frame, stabile_pose[0], stabile_pose[1], color=(128, 255, 128))

        # Show preview.
        cv2.imshow("Preview", frame)

        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    main()
