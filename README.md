# Potato Leaf Disease Detection Using Deep Learning

## Project Overview

This project explores the application of Deep Learning in agriculture through image-based classification of potato leaf diseases.

The objective is to develop and evaluate image classification models that identify potato leaves as:

- Healthy
- Early Blight
- Late Blight

## Problem Statement

Potato diseases can negatively affect crop health and productivity. Early identification can help farmers make more informed crop-management decisions.

This project investigates how Convolutional Neural Networks (CNNs) and transfer learning can be used to classify potato leaf images.

## Dataset

**Source:** [PlantVillage Dataset](https://github.com/spMohanty/PlantVillage-Dataset)

The project uses the colour images from the potato subset.

| Class | Number of Images |
|---|---:|
| Early Blight | 1,000 |
| Late Blight | 1,000 |
| Healthy | 152 |
| **Total** | **2,152** |

## Planned Methodology

1. Dataset exploration and preprocessing.
2. Image resizing and data preparation.
3. Training a baseline CNN.
4. Training and comparing transfer learning models.
5. Evaluating model performance.
6. Developing an interactive Streamlit dashboard.

## Model Evaluation

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Interactive Dashboard

A Streamlit dashboard will allow users to upload potato leaf images, view model predictions and explore model performance.

## Project Status

Dataset acquisition completed. Image exploration, preprocessing, model development and dashboard implementation are ongoing.

## Tools and Technologies

- Python
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Git and GitHub

## Disclaimer

This is an academic project. Predictions should not be treated as a substitute for professional agricultural diagnosis.