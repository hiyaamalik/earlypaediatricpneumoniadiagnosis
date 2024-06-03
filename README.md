# Early Paediatric Pneumonia diagnosis
A comparative study between a pre-trained model (VGG-19) and a custom CNN model (from scratch) for early pneumonia diagnosis among children

## Table of Contents
 1. Introduction
 2. Methodology
    Transfer Learning with VGG-19
    Custom CNN Model
 3. Dataset
 4. Results

## Introduction
This project aims to classify pediatric pneumonia from chest X-rays using deep learning techniques. Two approaches were explored: transfer learning with the pre-trained VGG-19 model and a custom-designed CNN model.

## Methodology
### Transfer Learning with VGG-19
Model Preparation:
The pre-trained VGG-19 model was used, with its fully connected layers removed to preserve the convolutional bases.
A custom dense layer with softmax activation was appended to serve as the output layer for binary classification.
All layers of the VGG-19 model, except the newly added dense layer, were frozen to retain pre-learned features and enable effective class distinction.
Model Compilation:
Optimizer: Adam with a learning rate of 1e-3.
Loss Function: Binary cross-entropy.
Evaluation Metric: Accuracy.
Training:
The model was trained using a dataset with images resized to 224x224 pixels.
Performance was validated using a validation set to ensure generalization and prevent overfitting.

### Custom CNN Model
Data Preparation:
Image data augmentation techniques (rotation, shifting, zooming, shearing, horizontal flipping) were applied to enhance generalization.
Model Architecture:
The custom CNN consisted of multiple convolutional layers with max-pooling layers to reduce spatial dimensions and extract hierarchical features.
ReLU activation was used to introduce non-linearity.
Batch normalization was applied for model stability and faster convergence.
Two fully connected layers with a 20% dropout rate were included to address overfitting.
The output layer had two nodes with softmax activation for binary classification.
Model Compilation:
Optimizer: Stochastic Gradient Descent (SGD).
Loss Function: Binary cross-entropy.
Training:
The model was trained over multiple epochs, adjusting weights and biases through backpropagation.
Validation data was used to assess model performance during training.

## Dataset
The dataset comprises 8,287 JPEG images of pediatric chest X-rays collected at Guangzhou Women and Children's Medical Center, China. The dataset includes:

### Training Set:
Normal: 805 images
Pneumonia: 3,406 images
### Test Set:
Normal: 234 images
Pneumonia: 390 images
### Validation Set:
Normal: 536 images
Pneumonia: 2,916 images
The images were resized to 256x256 pixels, and expert physicians assigned diagnostic labels to ensure accuracy.

## Results
The project successfully implemented transfer learning with VGG-19 and a custom CNN model to classify pediatric pneumonia from chest X-rays. Technically, custom CNN gave a better performance accuracy and also, it was comparatively faster and less complex than the pre-trained VGG 19 model
