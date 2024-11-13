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

```bash
pip install numpy scikit-image scikit-learn joblib matplotlib opencv-python
