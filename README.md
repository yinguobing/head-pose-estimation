# Head pose estimation

This repo shows how to estimate human head pose from videos using TensorFlow and OpenCV.

![demo](https://github.com/yinguobing/head-pose-estimation/raw/master/demo.gif)
![demo](https://github.com/yinguobing/head-pose-estimation/raw/master/demo1.gif)

## Dependence
- TensorFlow 1.4
- OpenCV 3.3
- Python 3

The code is tested under Ubuntu 16.04.

## How it works

There are three major steps:

1. Face detection. A face detector is adopted to provide a face box containing a human face. Then the face box is expanded and transformed to a square to suit the needs of later steps.

2. Facial landmark detection. A custom trained facial landmark detector based on TensorFlow is responsible for output 68 facial landmarks.

3. Pose estimation. Once we got the 68 facial landmarks, a mutual PnP algorithms is adopted to calculate the pose.

## Miscellaneous
- The marks is detected frame by frame, which result in small variance between adjacent frames. This makes the pose unstaible. A Kalman filter is used to solve this problem, you can draw the original pose to observe the difference.

- The 3D model of face comes from OpenFace, you can find the original file [here](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/lib/local/LandmarkDetector/model/pdms/In-the-wild_aligned_PDM_68.txt).

- The build in face detector comes from OpenCV. https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

## License
The code is licensed under the MIT license. However, the pre-trained TensorFlow model file is trained with various public data sets which have their own licenses. Please refer to them before using this code.

- 300-W: https://ibug.doc.ic.ac.uk/resources/300-W/
- 300-VW: https://ibug.doc.ic.ac.uk/resources/300-VW/
- LFPW: https://neerajkumar.org/databases/lfpw/
- HELEN: http://www.ifp.illinois.edu/~vuongle2/helen/
- AFW: https://www.ics.uci.edu/~xzhu/face/
- IBUG: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

(Currently we are trying to empower embeded devices to be more powerful with deep learning, please get in touch if you are interested and would like to join us.)
