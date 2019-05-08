# Head pose estimation

This repo shows how to estimate human head pose from videos using TensorFlow and OpenCV.

![demo](https://github.com/yinguobing/head-pose-estimation/raw/master/demo.gif)
![demo](https://github.com/yinguobing/head-pose-estimation/raw/master/demo1.gif)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- TensorFlow 1.4. It seems v1.12 also works.
- OpenCV 3.3 or higher.
- Python 3.5

The code is tested under Ubuntu 16.04.

### Installing

This repository comes with a pre-trained model for facial landmark detection. Just git clone then you are good to go.

```bash
# From the directory where you want to put this project:
git clone https://github.com/yinguobing/head-pose-estimation.git
```

### Running
The entrance file is `estimate_head_pose.py`. This will use your usb camera as the video source for demonstration.

```bash
# From the project directory, run:
python3 estimate_head_pose.py
```

You can change the video source to any video file that OpenCV supports.

## How it works

There are three major steps:

1. Face detection. A face detector is adopted to provide a face box containing a human face. Then the face box is expanded and transformed to a square to suit the needs of later steps.

2. Facial landmark detection. A custom trained facial landmark detector based on TensorFlow is responsible for output 68 facial landmarks.

3. Pose estimation. Once we got the 68 facial landmarks, a mutual PnP algorithms is adopted to calculate the pose.

The marks is detected frame by frame, which result in small variance between adjacent frames. This makes the pose unstable. A Kalman filter is used to solve this problem, you can draw the original pose to observe the difference.

## Retrain the model

To reproduce the facial landmark detection model, you can refer to this [series](https://yinguobing.com/deeplearning/) of posts(in Chinese only).


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
The pre-trained TensorFlow model file is trained with various public data sets which have their own licenses. Please refer to them before using this code.

- 300-W: https://ibug.doc.ic.ac.uk/resources/300-W/
- 300-VW: https://ibug.doc.ic.ac.uk/resources/300-VW/
- LFPW: https://neerajkumar.org/databases/lfpw/
- HELEN: http://www.ifp.illinois.edu/~vuongle2/helen/
- AFW: https://www.ics.uci.edu/~xzhu/face/
- IBUG: https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

The 3D model of face comes from OpenFace, you can find the original file [here](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/lib/local/LandmarkDetector/model/pdms/In-the-wild_aligned_PDM_68.txt).

The build in face detector comes from OpenCV. 
https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector

## Finally
If you are interested in Deep Learning and happened to be seeking for a job opportunity, feel free to get in touch.
