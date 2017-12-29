"""Test code for mark detector"""
import numpy as np

import cv2
import mark_detector
import pose_estimator
import point_stabilizer

INPUT_SIZE = 128


def main():
    """MAIN"""
    # Get frame from webcam or video file
    cam = cv2.VideoCapture(
        # '/home/robin/Documents/landmark/dataset/300VW_Dataset_2015_12_14/009/vid.avi'
        0
    )

    # writer = cv2.VideoWriter(
    #     './clip.avi', cv2.VideoWriter_fourcc(*'MJPG'), 25, (1280, 480), True)

    # Introduce point stabilizer.
    stabilizers = [point_stabilizer.Stabilizer(
        cov_process=0.01, cov_measure=0.1) for _ in range(68)]

    while True:
        # Read frame
        frame_got, frame = cam.read()
        if frame_got is False:
            break
        # frame = frame[0:480, 300:940]
        frame_cnn = frame.copy()
        # frame_dlib = frame.copy()

        # CNN benchmark.
        facebox = mark_detector.extract_cnn_facebox(frame_cnn)
        if facebox is not None:
            face_img = frame[
                facebox[1]: facebox[3],
                facebox[0]: facebox[2]]

            # Detect landmarks from image of 128x128.
            face_img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            landmarks = mark_detector.detect_marks(
                face_img, mark_detector.MARK_SESS, mark_detector.MARK_GRAPH)

            # Convert the marks locations from local CNN to global image.
            landmarks *= (facebox[2] - facebox[0])
            landmarks[:, 0] += facebox[0]
            landmarks[:, 1] += facebox[1]

            # Unconment the following line to show raw marks.
            # mark_detector.draw_marks(frame_cnn, landmarks)

            # All kind of detections faces a common issue: jitter. Usually this is
            # solved by kinds of estimator, like partical filter or Kalman filter, etc.
            # Here an Extended Kalman Filter is introduced as the target is not always
            # in the same state. An optical flow tracker also proved to be helpfull to
            # tell which state the target is currently in.
            stabile_marks = []
            # TODO: use optical flow to determine how to set stabilizer.
            if True:
                cov_process = 0.0001
                cov_measure = 0.1
                for stabilizer in stabilizers:
                    stabilizer.set_q_r(cov_process=cov_process,
                                       cov_measure=cov_measure)
            else:
                cov_process = 0.1
                cov_measure = 0.001
                for stabilizer in stabilizers:
                    stabilizer.set_q_r(cov_process=cov_process,
                                       cov_measure=cov_measure)

            for point, stabilizer in zip(landmarks, stabilizers):
                stabilizer.update(point)
                stabile_marks.append([stabilizer.prediction[0],
                                      stabilizer.prediction[1]])
            mark_detector.draw_marks(
                frame_cnn, stabile_marks, color=(0, 255, 0))

            # Try pose estimation
            pose_marks = pose_estimator.get_pose_marks(stabile_marks)
            pose_marks = np.array(pose_marks, dtype=np.float32)
            pose = pose_estimator.solve_pose(pose_marks)
            frame_cnn = pose_estimator.draw_annotation_box(
                frame_cnn, pose[0], pose[1])

        cv2.imshow("Preview", frame_cnn)
        # writer.write(frame_cmb)

        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    main()
