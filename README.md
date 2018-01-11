# Head pose estimation

Use CNN and OpenCV to estimate head poses.

![demo](https://github.com/yinguobing/head-pose-estimation/raw/master/demo.gif)

## How it works

This repo shows how to detect human head pose from image.

There are three major steps in the code, listed below.

1. Face detection. I use an face detector in OpenCV which provides a box contains a human face. The box is espanded and transformed to a square to suit the need of later step.

2. Facial landmark detection. In this step, a custom trained facial landmark detector based on TensorFlow is responsible for output 68 facial landmarks from face image of step 1.

3. Pose estimation. Once we got the 68 facial landmarks, a mutual PnP algorithms is adopted to calculate the pose.

## Other important techniques

A Kalman filter is used to stabilize the facial landmarks.

A optical flow tracker is used to detect head motion, which is useful for setting kalman filter parameters.
