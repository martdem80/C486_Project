# Real-time ASL Recognition with Fine-tuned YOLOv8

This repository contains code for model training, implementation and App development of a real-time American Sign Language (ASL) recognition system using fine-tuned YOLOv8 classification model.

## 📋 Overview

This project uses fine-tuned YOLOv8 model, a state-of-the-art object detection and segmentation model, to recognize ASL hand gestures in real-time through a webcam feed. 
The system is trained to detect and classify the 24 letters of the English alphabet in ASL (except "J" and "Z" gestures due to the motions). 

## 🔧 Prerequisites

- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- CUDA-capable GPU (recommended)

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/martdem80/C486_Project
cd C486_Project

# Create a Conda environment
conda create -n asl_recognition python=3.8
conda activate asl_recognition

# Install dependencies
pip install -r requirements.txt
```

## 📁 Dataset Preparation

The dataset should be organized in the following structure:

```
Dataset/
├── train/
│   └── images/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│   └── labels
│       ├── label1.jpg
│       ├── label2.jpg
│       └── ...
├── valid/
│   └── images/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
│   └── labels
│       ├── label1.jpg
│       ├── label2.jpg
│       └── ...
└── test/
    └── images/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    └── labels
        ├── label1.jpg
        ├── label2.jpg
        └── ...
```

**Note: Each image should have a corresponding label file in YOLO format!!!**

## 🚀 Model Training

The `model_training_yolov8.py` script handles the training process:

```bash
python model_training_yolov8.py
```

The training process includes:

1. Creation of a dataset configuration YAML file
2. Fine-tuning YOLOv8 on the ASL dataset
3. Generating performance metrics and visualizations
4. Saving the best model as 'best.pt'

## 📷 Real-time Implementation

The `implementation_yolov8.py` script provides real-time ASL recognition using a webcam:

```bash
python implementation_yolov8.py
```

The implementation:
1. Loads the trained model
2. Captures video from the default camera
3. Processes each frame through the YOLOv8 model
4. Displays bounding boxes and class predictions for detected hand gestures
5. Continues until 'q' is pressed

## 📱 Mobile Application

*This section will be updated once the application development is complete.*

## 🔍 Code Structure

```
.
├── model_training_yolov8.py     # Script for training the YOLOv8 model
├── implementation_yolov8.py     # Script for real-time webcam implementation
├── app.py                       # Application
├── requirements.txt             # Required dependencies
└── README.md                    # This file
```

## 🐛 Troubleshooting

Common issues:
- **CUDA out of memory**: Reduce batch size or image size
- **Camera not detected**: Check camera index in `implementation_yolov8.py`
- **Low detection accuracy**: Try increasing training epochs or adjusting confidence threshold

## 📄 License

-  

## 👏 Acknowledgments

- Ultralytics for the YOLOv8 framework
