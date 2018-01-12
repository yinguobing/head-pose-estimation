"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
import numpy as np

import cv2
import mark_detector
# import optical_flow_tracker
from stabilizer import Stabilizer
import pose_estimator

INPUT_SIZE = 128


def main():
    """MAIN"""
    # Get frame from webcam or video file
    video_src = 0
    cam = cv2.VideoCapture(video_src)

    # Introduce point stabilizers for landmarks.
    stabilizers = [Stabilizer(
        cov_process=0.001, cov_measure=0.1) for _ in range(68)]

    # # Remember the user state for updating kalman filter parameters.
    # # 1: moving; 0: still.
    # target_latest_state = 0
    # target_current_state = 1

    # # Introduce an optical flow tracker to help to decide how kalman filter
    # # should be configured. Alos keep one frame for optical flow tracker.
    # tracker = optical_flow_tracker.Tracker()
    # frame_prev = cam.read()
    # frame_count = 0
    # tracker_threshold = 2

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

        # Draw a dark background for user state text on frame.
        # cv2.rectangle(frame, (4, 28), (70, 4), (70, 70, 70), -1)
        # frame_count += 1

        # # Optical flow tracker should work before kalman filter.
        # frame_opt_flw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # # If optical flow is already working, get user state from it.
        # if len(tracker.tracks) > 0:
        #     if tracker.get_average_track_length() > tracker_threshold:
        #         # User is moving.
        #         target_current_state = 1
        #         cv2.putText(frame, "Moving", (10, 20),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (78, 207, 219))
        #     else:
        #         target_current_state = 0
        #         # User is still.
        #         cv2.putText(frame, "Still", (10, 20),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 118))

        #     tracker.update_tracks(frame_prev, frame_opt_flw)

        # # Store current frame for next frame's optical flow.
        # frame_prev = frame_opt_flw

        # Pose estimation by 3 steps:
        # 1. detect face;
        # 2. detect landmarks;
        # 3. estimate pose
        frame_cnn = frame.copy()
        facebox = mark_detector.extract_cnn_facebox(frame_cnn)
        if facebox is not None:
            # # Set mask equal to face area for optical flow tracker.
            # target_box = [facebox[1], facebox[3],
            #               facebox[0], facebox[2]]

            # # Update state check threshold
            # tracker_threshold = abs(facebox[2] - facebox[0]) * 0.005

            # # Track might vanish in the current frame, get new one if needed.
            # if frame_count % 30 == 0:
            #     tracker.get_new_tracks(frame_opt_flw, target_box)

            # Uncomment following line to show optical flow tracks.
            # tracker.draw_track(frame_cnn)

            # Detect landmarks from image of 128x128.
            face_img = frame_cnn[facebox[1]: facebox[3],
                                 facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (INPUT_SIZE, INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            marks = mark_detector.detect_marks(
                face_img, mark_detector.MARK_SESS, mark_detector.MARK_GRAPH)

            # Stabilize the marks.
            stabile_marks = []
            for point, point_stabilizer in zip(marks, stabilizers):
                point_stabilizer.update(point)
                stabile_marks.append([point_stabilizer.state[0],
                                      point_stabilizer.state[1]])
            stabile_marks = np.reshape(stabile_marks, (-1, 2))

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            stabile_marks *= (facebox[2] - facebox[0])
            stabile_marks[:, 0] += facebox[0]
            stabile_marks[:, 1] += facebox[1]

            # Uncomment the following line to show raw marks.
            # mark_detector.draw_marks(frame_cnn, landmarks)

            # All kind of detections face a common issue: jitter. Usually this is
            # solved by kinds of estimators, like partical filter or Kalman filter, etc.
            # Here an Extended Kalman Filter is introduced as the target is not always
            # in the same state. An optical flow tracker also proved to be helpfull to
            # tell which state the target is currently in.
            # Different state require different kalman filter parameters. The filters
            # should be re-configured only if the state has changed, for better performance.
            # stabile_marks = []
            # if target_current_state != target_latest_state:
            #     if target_current_state == 1:
            #         # Target is moving.
            #         cov_process = 0.1
            #         cov_measure = 0.01
            #     else:
            #         # Traget is still.
            #         cov_process = 0.0001
            #         cov_measure = 0.1

            #     # Update state.
            #     target_latest_state = target_current_state

            #     # Re-configure the filters.
            #     for stabilizer in stabilizers:
            #         stabilizer.set_q_r(cov_process=cov_process,
            #                            cov_measure=cov_measure)

            # # Filters re-configured, stabilize the marks.
            # for point, stabilizer in zip(landmarks, stabilizers):
            #     stabilizer.update(point)
            #     stabile_marks.append([stabilizer.filter.statePost[0],
            #                           stabilizer.filter.statePost[1]])

            # # Uncomment following line to show stabile marks.
            # mark_detector.draw_marks(
            #     frame_cnn, marks, color=(0, 255, 0))
            # mark_detector.draw_marks(
            #     frame_cnn, stabile_marks, color=(255, 0, 0))

            # Try pose estimation
            pose_marks = pose_estimator.get_pose_marks(stabile_marks)
            pose_marks = np.array(pose_marks, dtype=np.float32)
            pose = pose_estimator.solve_pose(pose_marks)

            # Draw pose annotaion on frame.
            frame_cnn = pose_estimator.draw_annotation_box(
                frame_cnn, pose[0], pose[1])

        # Show preview.
        cv2.imshow("Preview", frame_cnn)

        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    main()
