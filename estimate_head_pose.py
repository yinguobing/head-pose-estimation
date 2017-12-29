"""Test code for mark detector"""
import dlib
import numpy as np

import cv2
import mark_detector
import optical_flow_tracker
import point_stabilizer
import pose_estimator

INPUT_SIZE = 128


def main():
    """MAIN"""
    # Get frame from webcam or video file
    video_src = 0
    cam = cv2.VideoCapture(video_src)

    # Construct a dlib shape predictor
    predictor = dlib.shape_predictor(
        'assets/shape_predictor_68_face_landmarks.dat')

    # Introduce point stabilizer.
    stabilizers = [point_stabilizer.Stabilizer(
        cov_process=0.1, cov_measure=1) for _ in range(68)]
    target_latest_state = 0     # 1: moving; 0: still.
    target_current_state = 1

    stabilizers_dlib = [point_stabilizer.Stabilizer(
        cov_process=0.1, cov_measure=1) for _ in range(68)]

    # Introduce an optical flow tracker to help to decide how kalman filter
    # should be configured. Alos keep one frame for optical flow tracker.
    tracker = optical_flow_tracker.Tracker()
    frame_prev = cam.read()
    frame_count = 0
    tracker_threshold = 2

    while True:
        # Read frame, corp it, flip it, suits your needs.
        frame_got, frame = cam.read()
        if frame_got is False:
            break
        if video_src == 0:
            frame = cv2.flip(frame, 2)
        # frame = frame[0:480, 300:940]
        cv2.rectangle(frame, (4, 28), (70, 4), (70, 70, 70), -1)
        frame_count += 1

        # Optical flow tracker should work before kalman filter.
        frame_opt_flw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if len(tracker.tracks) > 0:
            if tracker.get_average_track_length() > tracker_threshold:
                target_current_state = 1
                cv2.putText(frame, "Moving", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (78, 207, 219))
            else:
                target_current_state = 0
                cv2.putText(frame, "Still", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 118))

            tracker.update_tracks(frame_prev, frame_opt_flw)

        frame_prev = frame_opt_flw

        # CNN benchmark.
        frame_cnn = frame.copy()
        facebox = mark_detector.extract_cnn_facebox(frame_cnn)
        if facebox is not None:
            # Set face area as mask for optical flow tracker.
            target_box = [facebox[0], facebox[2], facebox[1], facebox[3]]
            # Update state check threshold
            tracker_threshold = abs(facebox[2] - facebox[0]) * 0.005
            if frame_count % 30 == 0:
                tracker.get_new_tracks(frame_opt_flw, target_box)
            # tracker.draw_track(frame_cnn)

            # Detect landmarks from image of 128x128.
            face_img = frame_cnn[
                facebox[1]: facebox[3],
                facebox[0]: facebox[2]]
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
            if target_current_state != target_latest_state:
                if target_current_state == 1:
                    cov_process = 0.1
                    cov_measure = 0.001
                else:
                    cov_process = 0.1
                    cov_measure = 1

                target_latest_state = target_current_state

                for stabilizer in stabilizers:
                    stabilizer.set_q_r(cov_process=cov_process,
                                       cov_measure=cov_measure)

            for point, stabilizer in zip(landmarks, stabilizers):
                stabilizer.update(point)
                stabile_marks.append([stabilizer.prediction[0],
                                      stabilizer.prediction[1]])
            # mark_detector.draw_marks(frame_cnn, stabile_marks)

            # Try pose estimation
            pose_marks = pose_estimator.get_pose_marks(stabile_marks)
            pose_marks = np.array(pose_marks, dtype=np.float32)
            pose = pose_estimator.solve_pose(pose_marks)
            frame_cnn = pose_estimator.draw_annotation_box(
                frame_cnn, pose[0], pose[1])

        # Dlib benchmark.
        frame_dlib = frame.copy()
        if facebox is not None:
            # Detect landmarks
            expd_ratio = 0.1
            dlib_box = dlib.rectangle(int(facebox[0] * (1 + expd_ratio)),
                                      int(facebox[1] * (1 + expd_ratio)),
                                      int(facebox[2] * (1 - expd_ratio)),
                                      int(facebox[3] * (1 - expd_ratio)))
            dlib_shapes = predictor(frame_dlib, dlib_box)
            dlib_mark_list = []
            for shape_num in range(68):
                dlib_mark_list.append(
                    [dlib_shapes.part(shape_num).x,
                     dlib_shapes.part(shape_num).y])

            stabile_marks_dlib = []
            for point, stabilizer in zip(dlib_mark_list, stabilizers_dlib):
                stabilizer.update(point)
                stabile_marks_dlib.append([stabilizer.prediction[0],
                                           stabilizer.prediction[1]])
            # Visualization of the result.
            # mark_detector.draw_marks(frame_dlib, dlib_mark_list)

            # Try pose estimation
            pose_marks_dlib = pose_estimator.get_pose_marks(stabile_marks_dlib)
            pose_marks_dlib = np.array(pose_marks_dlib, dtype=np.float32)
            pose_dlib = pose_estimator.solve_pose(pose_marks_dlib)
            frame_dlib = pose_estimator.draw_annotation_box(
                frame_dlib, pose_dlib[0], pose_dlib[1])

        # Combine two videos together.
        frame_cmb = np.concatenate((frame_dlib, frame_cnn), axis=1)
        cv2.imshow("Preview", frame_cmb)

        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    main()
