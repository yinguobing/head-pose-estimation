'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''
import numpy as np

import cv2


class Tracker:
    def __init__(self):
        self.track_len = 10
        self.tracks = []
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=500,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

    def update_tracks(self, img_old, img_new, vis):
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

            cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

        # New track groups got, do update.
        self.tracks = new_tracks

        cv2.polylines(vis, [np.int32(track)
                            for track in self.tracks], False, (0, 255, 0))

    def get_new_tracks(self, frame, roi):
        """Get new tracks every detect_interval frames."""
        # Using mask to determine where to look for feature points.
        mask = np.zeros_like(frame)
        mask[roi[0]:roi[1], roi[2]:roi[3]] = 255

        for x, y in [np.int32(track[-1]) for track in self.tracks]:
            cv2.circle(mask, (x, y), 5, 0, -1)

        # Get good feature points.
        feature_points = cv2.goodFeaturesToTrack(
            frame, mask=mask, **self.feature_params)

        if feature_points is not None:
            for x, y in np.float32(feature_points).reshape(-1, 2):
                self.tracks.append([(x, y)])


def main():
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
        vis = frame.copy()

        # Update tracks.
        if len(tracker.tracks) > 0:
            tracker.update_tracks(prev_gray, frame_gray, vis)

        # Get new tracks every detect_interval frames.
        target_box = [100, 400, 100, 400]
        if frame_idx % detect_interval == 0:
            tracker.get_new_tracks(frame_gray, target_box)

        frame_idx += 1
        prev_gray = frame_gray
        cv2.imshow('lk_track', vis)

        ch = cv2.waitKey(1)
        if ch == 27:
            break


if __name__ == '__main__':
    main()
