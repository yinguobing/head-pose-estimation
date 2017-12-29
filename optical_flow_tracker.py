'''
Lucas-Kanade sparse optical flow tracker. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
'''
from math import sqrt

import numpy as np

import cv2


class Tracker:
    """Lucas-Kanade sparse optical flow tracker"""

    def __init__(self):
        self.track_len = 5
        self.tracks = []
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=500,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

    def update_tracks(self, img_old, img_new):
        """Update tracks."""
        # Get old points, using the latest one.
        points_old = np.float32([track[-1]
                                 for track in self.tracks]).reshape(-1, 1, 2)

        # Get new points from old points.
        points_new, _st, _err = cv2.calcOpticalFlowPyrLK(
            img_old, img_new, points_old, None, **self.lk_params)

        # Get inferred old points from new points.
        points_old_inferred, _st, _err = cv2.calcOpticalFlowPyrLK(
            img_new, img_old, points_new, None, **self.lk_params)

        # Compare between old points and inferred old points
        error_term = abs(
            points_old - points_old_inferred).reshape(-1, 2).max(-1)
        point_valid = error_term < 1

        new_tracks = []
        for track, (x, y), good_flag in zip(self.tracks, points_new.reshape(-1, 2), point_valid):
            # Track is good?
            if not good_flag:
                continue

            # New point is good, add to track.
            track.append((x, y))

            # Need to drop first old point?
            if len(track) > self.track_len:
                del track[0]

            # Track updated, add to track groups.
            new_tracks.append(track)

        # New track groups got, do update.
        self.tracks = new_tracks

    def get_new_tracks(self, frame, roi):
        """Get new tracks every detect_interval frames."""
        # Using mask to determine where to look for feature points.
        mask = np.zeros_like(frame)
        mask[roi[0]:roi[1], roi[2]:roi[3]] = 255

        # Get good feature points.
        feature_points = cv2.goodFeaturesToTrack(
            frame, mask=mask, **self.feature_params)

        if feature_points is not None:
            for x, y in np.float32(feature_points).reshape(-1, 2):
                self.tracks.append([(x, y)])

    def get_average_track_length(self):
        """Get the average track length"""
        length = 0
        tracks = np.array(self.tracks)
        def distance(track):
            """Get distance between the first and last point."""
            delta_x = abs(track[-1][0] - track[0][0])
            delta_y = abs(track[-1][1] - track[0][1])
            return sqrt(delta_x*delta_x + delta_y*delta_y)
        for track in tracks:
            length += distance(track)
        return length / len(tracks)

    def draw_track(self, image):
        """Draw track lines on image."""
        cv2.polylines(image, [np.int32(track)
                              for track in self.tracks], False, (0, 255, 0))


def main():
    """Test code"""
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    tracker = Tracker()

    cam = cv2.VideoCapture(video_src)
    detect_interval = 5
    frame_idx = 0

    prev_gray = cam.read()
    while True:
        _ret, frame = cam.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update tracks.
        if len(tracker.tracks) > 0:
            tracker.update_tracks(prev_gray, frame_gray)

        # Get new tracks every detect_interval frames.
        target_box = [100, 400, 100, 400]
        if frame_idx % detect_interval == 0:
            tracker.get_new_tracks(frame_gray, target_box)

        # Draw tracks
        tracker.draw_track(frame)

        frame_idx += 1
        prev_gray = frame_gray
        cv2.imshow('lk_track', frame)
        ch = cv2.waitKey(1)
        if ch == 27:
            break


if __name__ == '__main__':
    main()
