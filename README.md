# assignment
Machine learning model for classification between defective and non defective industrial equipment
# Industrial Equipment Defect Classification using Machine Learning

## Project Overview
This project aims to classify images of industrial equipment into two categories: **Defective** and **Non-Defective**, using machine learning. The dataset consists of images representing industrial parts, with defects being visually identifiable. The project leverages **Random Forest Classifier** and **Histogram of Oriented Gradients (HOG)** feature extraction to classify the images.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The goal of this project is to automate the classification of industrial equipment images into "Defective" and "Non-Defective" categories. The model uses traditional machine learning techniques, specifically **Random Forest**, and image features derived from **HOG (Histogram of Oriented Gradients)** to train a classifier.

## Dataset
This project uses the **Kolektor Surface Defect Dataset (KolektorSDD)**, which contains images of industrial equipment with labels for defective and non-defective categories. The dataset is split into:
- **Train**: Images used to train the model.
- **Validation**: Images used to evaluate the model during training.
- **Test**: Images used to evaluate the final model's performance.

Each folder has two categories: 
1. **Defective**
2. **Non-Defective**

### Data Preprocessing
The dataset is preprocessed by:
- Organizing images into **Defective** and **Non-Defective** categories.
- Extracting **HOG features** from each image to represent the underlying patterns.
- Splitting the data into training, validation, and test sets.

## Methodology

### Feature Extraction: HOG (Histogram of Oriented Gradients)
To extract meaningful patterns from the images, the **HOG** feature descriptor is used. This method extracts gradient-based features that capture edge information useful for classification.

### Model: Random Forest Classifier
The **Random Forest Classifier** is trained on the extracted HOG features to classify the images into the two categories: **Defective** and **Non-Defective**. The model is then evaluated using accuracy, precision, recall, and F1-score.

## Installation

### Prerequisites
To run this project, ensure that you have the following libraries installed:
- Python 3.x
- `numpy`
- `scikit-image`
- `scikit-learn`
- `joblib`
- `matplotlib` (for visualization, optional)
- `opencv-python` (for additional image processing, optional)

### Install Required Libraries
You can install the necessary libraries using **pip**:

#Usage
Training the Model
###To train the model, run the script train_model.py, which will:

Load images from the Train directory.
#Extract HOG features from the images.
Train a Random Forest Classifier using the extracted features.
Evaluate the model's performance on the Validation set.
python train_model.py

#Testing the Model
After training, you can test the model on the Test set by running the script test_model.py, which will:

Load the saved model.
Evaluate the model on the Test images.
Print the accuracy and classification report.

# Load the model
model = joblib.load('defect_classification_model_rf.joblib')

# Load and preprocess the image
image_path = 'path_to_new_image.jpg'
image = imread(image_path, as_gray=True)
fd, _ = hog(image, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

# Predict the class
prediction = model.predict([fd])
print("Predicted class:", "Defective" if prediction[0] == 1 else "Non-Defective")
Saving the Model

# To save the trained model, use joblib to serialize the model:
import joblib
joblib.dump(model, "defect_classification_model_rf.joblib")
Loading the Model

# To load a previously saved model and make predictions:
import joblib
model = joblib.load('defect_classification_model_rf.joblib')

# Results
Validation Accuracy: 85%
Test Accuracy: 87%

# Classification Report:
Precision, recall, and F1-scores for each category are provided for better understanding of model performance.
Model Performance
Precision: 0.87
Recall: 0.85
F1-Score: 0.86
![Screenshot 2024-11-13 130510](https://github.com/user-attachments/assets/db47ea11-a5a2-4533-9366-359b064e40b0)

These results show that the model is performing well in distinguishing between defective and non-defective equipment, with the validation and test sets providing strong generalization performance.

# Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request with your changes. For any issues or enhancements, feel free to open an issue in the GitHub repository.

# License
This project is licensed under the MIT License - see the LICENSE file for details.


This markdown format is ready to be copied into your GitHub repository's **README.md** file.
