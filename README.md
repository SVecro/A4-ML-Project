# Satellite Imagery Airplane Detection

## Overview

This repository presents an end-to-end machine learning workflow applied to the **PlanesNet** satellite imagery dataset. The objective is to develop a binary classification model capable of detecting airplanes within **20×20 pixel RGB** satellite image tiles.

This project was developed by engineering students in the *Industry & Robotics – Aerospace & Defense* track to address real-world challenges in aerial surveillance and automated monitoring. The task is approached as a supervised learning problem, progressing from raw pixel processing to feature engineering and classical machine learning models.

---

## Dataset Description

**Source:**  
PlanesNet – PlanetScope satellite imagery  
https://www.kaggle.com/datasets/rhammell/planesnet

**License:**  
Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)  
https://creativecommons.org/licenses/by-sa/4.0/

### Dataset Statistics

- **Total Samples:** 32,000  
- **Image Size:** 20×20 pixels, RGB  
- **Classes:**  
  - **Class 0 – No Plane:** 24,000 images (75%)  
  - **Class 1 – Plane:** 8,000 images (25%)

---

## Exploratory Data Analysis

Early quantitative observations identified meaningful differences between airplane and non-airplane tiles:

- **Sharpness:** Plane images exhibit higher Laplacian variance (≈727) compared to background tiles (≈570).
- **Intensity:** Plane tiles show higher brightness levels, especially in the upper-intensity range.

These characteristics indicate the presence of detectable textural and contrast-based patterns.

---

## Methodology

The pipeline consists of data preprocessing, feature extraction, dimensionality reduction, and model training.

### 1. Data Preprocessing

To standardize inputs and reduce noise:

- Denoising: Median filter (kernel size 3) applied to preserve edges.  
- Contrast Normalization: CLAHE applied to reduce lighting inconsistencies.  
- Standardization: Pixel values normalized using training-set statistics (mean = 0.687, std = 0.190).  
- Data Augmentation: Random rotations (±10°), flips, and Gaussian noise to mitigate overfitting.

### 2. Feature Extraction (HOG)

Histogram of Oriented Gradients (HOG) features were computed for all images.

- **Parameters:**  
  - 9 orientations  
  - 4×4 pixels per cell  
  - 2×2 cells per block  
- **Output:** 576-dimensional feature vector per image

### 3. Dimensionality Reduction (PCA)

PCA was applied to the HOG vectors to reduce dimensionality.

- **Retained Components:** 190  
- **Explained Variance:** 95.61%

### 4. Class Balancing (SMOTE)

To counter the 75/25 class imbalance, SMOTE was applied only on the training set after PCA, producing a balanced 50/50 distribution.

---

## Experimental Results

Models were evaluated on a stratified validation set of 4,800 images.

| Model            | Accuracy | Precision (Plane) | Recall (Plane) | F1-Score (Plane) |
|------------------|----------|--------------------|----------------|------------------|
| Decision Tree    | 76.25%   | 0.52               | 0.69           | 0.59             |
| Random Forest    | 88.50%   | 0.84               | 0.67           | 0.74             |
| XGBoost          | 88.65%   | 0.77               | 0.77           | 0.77             |
| SVM (RBF Kernel) | 93.90%   | 0.93               | 0.82           | 0.87             |

---

## Performance Analysis

### SVM (RBF Kernel)
The best-performing model. The RBF kernel effectively handles the high-dimensional, PCA-transformed feature space and captures non-linear decision boundaries.

### Tree-Based Models (Random Forest, XGBoost)
Although strong, these models underperformed relative to SVM. PCA produces linear combinations of features, leading to oblique boundaries that tree models approximate inefficiently with axis-aligned splits.

### Decision Tree (Baseline)
Served as a baseline, highlighting the need for more expressive models such as ensembles or kernel-based classifiers.

---

## Installation and Usage

### Prerequisites

- Python 3.8+
- Required libraries:  
  `numpy`, `pandas`, `opencv-python`, `scikit-learn`, `scikit-image`, `xgboost`, `imbalanced-learn`, `matplotlib`

### Steps

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Notebooks.
