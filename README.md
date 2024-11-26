# Image Classification with CIFAR-10

This project demonstrates how to classify images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained on the CIFAR-10 dataset and can predict the class of an image based on its content.

## Overview

The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The classes are:

- Plane
- Car
- Bird
- Deer
- Dog
- Horse
- Ship
- Truck

This project involves two main parts:

1. **Training the Model**: The model is built and trained using the CIFAR-10 dataset to classify images into the 10 classes.
2. **Image Prediction**: A pre-trained model is used to predict the class of a given image.

## Requirements

Before running the code, make sure you have the following dependencies installed:

- `TensorFlow` (for model training and inference)
- `OpenCV` (for image processing)
- `NumPy` (for numerical operations)

You can install these dependencies using pip:

```bash
pip install tensorflow opencv-python numpy
