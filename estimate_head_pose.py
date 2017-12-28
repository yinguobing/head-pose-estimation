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
    stabilizer_strong = [point_stabilizer.Stabilizer() for _ in range(68)]
    stabilizer_light = [point_stabilizer.Stabilizer(
        cov_noise=0.01, cov_measure=0.1) for _ in range(68)]

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

            # Detect landmarks
            face_img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            landmarks = mark_detector.detect_marks(
                face_img, mark_detector.MARK_SESS, mark_detector.MARK_GRAPH)

            # Convert the marks locations from local CNN to global image.
            landmarks *= (facebox[2] - facebox[0])
            landmarks[:, 0] += facebox[0]
            landmarks[:, 1] += facebox[1]
            mark_detector.draw_marks(frame_cnn, landmarks)

            # Stabiliz all marks?
            stabile_marks = []
            #TODO: use optical flow to determine which stabilizer to use.
            if False:
                stabilizer = stabilizer_light
            else:
                stabilizer = stabilizer_strong

            for point, stblzer in zip(landmarks, stabilizer):
                stblzer.update(point)
                stabile_marks.append([stblzer.prediction[0],
                                      stblzer.prediction[1]])
            mark_detector.draw_marks(
                frame_cnn, stabile_marks, color=(0, 255, 0))

            # Try pose estimation
            pose_marks = pose_estimator.get_pose_marks(stabile_marks)
            pose_marks = np.array(pose_marks, dtype=np.float)
            pose = pose_estimator.solve_pose(pose_marks)
            frame_cnn = pose_estimator.draw_annotation_box(
                frame_cnn, pose[0], pose[1])

        cv2.imshow("Preview", frame_cnn)
        # writer.write(frame_cmb)

        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    main()
