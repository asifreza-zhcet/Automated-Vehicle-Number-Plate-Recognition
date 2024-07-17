
# **Title**: Automated Vehicle Number Plate Detection with OpenCV, EasyOCR, and YOLO
![](https://github.com/asifreza-zhcet/Automated-Vehicle-Number-Plate-Recognition/blob/main/sample_video/sample.gif)


## Description:
This project implements a real-time vehicle number plate detection system using OpenCV, EasyOCR, and a custom YOLO model. It processes video input, detects cars, extracts license plates using the YOLO model, and recognizes the plate text using EasyOCR.

## Features:

Real-time Detection: Analyzes video streams for vehicles and license plates.
YOLO-based Plate Extraction: Employs a custom YOLO model for accurate license plate cropping.
EasyOCR Integration: Leverages EasyOCR's pre-trained text recognition model for efficient plate number extraction. 

## Requirements:

Python 3.x (https://www.python.org/downloads/)  
OpenCV (https://opencv.org/)  
EasyOCR (https://www.jaided.ai/easyocr/documentation/)  
PyTorch (https://pytorch.org/) (for YOLO model loading)  
NumPy (https://numpy.org/) 

## Usage:

Replace video_file (line 17) in main.py with the path to your video file.

Run the script:
```bash
  python main.py

```

The output video with detected license plates will be saved in the output folder.

## Explanation:

### main.py:

Loads the video and performs frame-by-frame processing.
Uses OpenCV's object detection capabilities or a pre-trained car detection model (consider mentioning specific models if used) to identify cars in each frame.
For each detected car, crops the region of interest (ROI) around the car.
Passes the cropped car image to the plate function in utils.py for license plate extraction.
Once the plate is extracted, passes the image to the preprocessing function in utils.py for text recognition using EasyOCR.
Displays the original video with bounding boxes and recognized plate text overlaid.
Saves the final output video to the output folder.

### utils.py:

**Defines the plate function:**  
Loads the custom YOLO model.
Performs object detection within the car image to locate the license plate.
Returns the cropped image containing the license plate.  
**Defines the preprocessing function:**
Converts the cropped license plate image to a format suitable for EasyOCR.
Uses EasyOCR to recognize the text in the image.
Returns the recognized plate number as a string.


## Training a Custom YOLO Model for License Plate Detection
The training code is implemented in the training_plate_detection.ipynb Jupyter Notebook.

Dataset
The training data for this project was sourced from the Roboflow Universe license plate recognition dataset available at 
https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4. This dataset provides labeled images containing vehicles with bounding boxes around license plates




**Click on the Youtube link given below**
[![Watch the video](https://img.youtube.com/vi/oWmAqrceugM/hqdefault.jpg)](https://www.youtube.com/watch?v=oWmAqrceugM)