# Face Mask Detection Using Deep Learning (CNN, CNN+Data Augumentation, Realtime detection)
<!-- Inserting the image with HTML and Markdown -->
<p align="center">
  <img src="icon_2.png" alt="Icon" width="200" height="200">
</p>

## Project Overview

The **Face Mask Detection Using CNN and Data Augmentation** project consists of three sub-projects aimed at detecting whether a person is wearing a face mask using convolutional neural networks (CNNs) and data augmentation techniques. The sub-projects are as follows:

1. **Face-Mask-Detection-Using-CNN**: This sub-project focuses on building a face mask detection model using CNNs without data augmentation.

2. **Face-Mask-Detection-Using-CNN-with-Data-Augmentation**: In this sub-project, data augmentation techniques are incorporated to improve the performance of the face mask detection model.

3. **Real-time Deployment Using Webcam**: The final sub-project involves deploying the trained model in real-time using a webcam or an IP camera. Haarcascade frontal face detection is utilized during real-time detection to detect faces in the input video stream.

## Problem Statement

With the ongoing COVID-19 pandemic, the wearing of face masks has become a crucial measure to prevent the spread of the virus. The problem addressed in this project involves building computer vision systems capable of automatically detecting whether individuals in images or video streams are wearing face masks. This technology can be used in various applications, including monitoring compliance with mask-wearing regulations in public spaces.

## Tools and Technologies

The project leverages the following tools and technologies:

- **Python**: The primary programming language for implementing the deep learning models, data augmentation techniques, and real-time deployment.
- **TensorFlow and Keras**: Deep learning frameworks used for building and training the convolutional neural network (CNN) models.
- **OpenCV**: A computer vision library used for image processing tasks, real-time deployment, and face detection using Haarcascade.
- **Scikit-image**: A library for image processing and computer vision algorithms, including data augmentation techniques.
- **Matplotlib**: A plotting library used for visualizing images, model performance metrics, and real-time detection results.

## Sub-Projects Overview

### 1. Face-Mask-Detection-Using-CNN

This sub-project focuses on building a face mask detection model using CNNs. The model is trained on a dataset of annotated images of faces with and without masks. The CNN architecture consists of multiple convolutional layers followed by max-pooling layers to extract features from input images. The final layers of the network include fully connected (dense) layers and softmax activation to perform classification into mask and non-mask classes.

### 2. Face-Mask-Detection-Using-CNN-with-Data-Augmentation

In this sub-project, data augmentation techniques are incorporated to improve the performance of the face mask detection model. Data augmentation helps increase the diversity of the training data and reduces overfitting by generating augmented images with variations in rotation, flipping, zooming, brightness, contrast, and shearing.

### 3. Real-time Deployment Using Webcam

The final sub-project involves deploying the trained model in real-time using a webcam or an IP camera. Haarcascade frontal face detection is utilized during real-time detection to detect faces in the input video stream. Once faces are detected, the face mask detection model is applied to classify whether individuals are wearing masks or not. Real-time detection results are displayed on the video stream, enabling monitoring of mask compliance in real-time.

## Dataset

The dataset used in this project consists of annotated images of faces with and without masks. Each image is labeled with the corresponding class (mask or non-mask). The dataset may be sourced from publicly available datasets or collected and annotated manually for specific use cases.

## Model Evaluation

The performance of the trained models is evaluated using metrics such as accuracy, precision, recall, and F1-score on a separate validation dataset. Additionally, the models' performance may be visualized using confusion matrices and ROC curves to assess their ability to discriminate between mask and non-mask classes.

## Contact Information

For any inquiries or feedback regarding this project, please feel free to contact:

- Email: gaddisaolex@gmail.com

