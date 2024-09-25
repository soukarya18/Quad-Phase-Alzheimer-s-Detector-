# Quad Phase Alzheimer's Detector
This project implements a deep learning model that classifies Alzheimer's disease into four distinct stages using MRI scan images. The model is built with a Convolutional Neural Network (CNN) architecture that processes the image data to accurately predict the stage of the disease. The project is designed for early diagnosis and classification of Alzheimer's disease, aiding in medical decision-making.

## Features
CNN Architecture: The model is based on a Sequential CNN architecture, specifically designed for image classification tasks.  
The architecture includes ->     
**Convolution Layers**: Extract features from MRI images using convolution operations.  
**MaxPooling Layers**: Reduce the dimensionality of the feature maps while retaining essential information.  
**Flatten Layer**: Transforms the 2D matrix into a 1D vector to feed into the Dense layers.  
**Dense Layers**: Fully connected layers that handle the final classification task with a softmax activation function.  
## Classification:  
The model classifies Alzheimer's disease into **four stages** based on MRI scan data:  
Mild Dementia  
Moderate Dementia   
Very Mild Dementia  
Non-Demented  

## Model Architecture
The model has the following layers:  



  model = Sequential([  

    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  
    MaxPooling2D((2, 2)),  
    Conv2D(32, (3, 3), activation='relu'),  
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(300, activation='relu'),
    Dense(4, activation='softmax')
])   

**Activation Function**: ReLU is used in convolutional layers for non-linear activations.
**Optimizer**: Adam optimizer for efficient training.
**Loss Function**: Categorical cross-entropy since this is a multiclass classification problem.
Dataset
The dataset consists of MRI images categorized into four stages of Alzheimer's. The images are preprocessed to ensure uniformity in dimensions and to enhance model performance. Each image is resized to (128, 128, 3) for the CNN input.

## Performance
The model is trained on MRI scan data and achieves **high accuracy(95%)** in classifying the stages of Alzheimer's disease. It is optimized to generalize well across unseen data.
