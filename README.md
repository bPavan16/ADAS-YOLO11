# Traffic Sign Detection and Recognition Using YOLO v11

## Project Overview
This project aims to develop a computer vision system (ADAS) capable of detecting and recognizing traffic signs in real-time from video or image inputs. By leveraging the YOLO (You Only Look Once) v11 model, the system can quickly and accurately identify various traffic signs on the road. The primary goal is to assist autonomous vehicles and advanced driver-assistance systems (ADAS) in understanding and reacting to road signs, thus improving safety and aiding in navigation.

---

## Objectives
1. **Traffic Sign Detection**: Locate the traffic signs in a given image or video frame and draw bounding boxes around them.
2. **Traffic Sign Recognition**: Classify the detected traffic signs into specific categories (e.g., Stop, Yield, Speed Limit) to help the vehicle or system understand road conditions.
3. **Real-time Performance**: Ensure the model processes images and video frames at a high frame rate for practical real-time applications.

---

## Tools and Technologies
- **YOLO v11 Model**: The original YOLO model for object detection, known for its single-shot detection approach.
- **Dataset**: Use datasets like the *Indian Traffic Sign Recognition Benchmark* or similar datasets with annotated traffic sign images.
- **Deep Learning Frameworks**: TensorFlow or PyTorch for model training and testing.
- **OpenCV**: For image processing tasks, such as pre-processing and displaying detection results.

---

## Project Scope and Workflow
### 1. Dataset Preparation
- Download and prepare the traffic sign dataset.
- Annotate the images with bounding boxes and label classes for each traffic sign.
- Split the dataset into training, validation, and test sets.

### 2. Model Training
- Configure YOLO v11 to detect and recognize traffic signs.
- Train the model on the traffic sign dataset, adjusting hyperparameters for optimal performance.
- Use techniques like data augmentation to improve model generalization.

### 3. Testing and Evaluation
- Test the model on the test dataset to assess detection accuracy, classification accuracy, and inference speed.
- Evaluate performance using metrics such as mean Average Precision (mAP) and F1-score.
