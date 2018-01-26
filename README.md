# Head pose estimation

This repo shows how to estimate human head pose from videos using TensorFlow and OpenCV.

![demo](https://github.com/yinguobing/head-pose-estimation/raw/master/demo.gif)
![demo](https://github.com/yinguobing/head-pose-estimation/raw/master/demo1.gif)

## How it works

There are three major steps:

1. Face detection. A face detector is adopted to provide a face box containing a human face. Then the face box is expanded and transformed to a square to suit the needs of later steps.

2. Facial landmark detection. A custom trained facial landmark detector based on TensorFlow is responsible for output 68 facial landmarks.

3. Pose estimation. Once we got the 68 facial landmarks, a mutual PnP algorithms is adopted to calculate the pose.

