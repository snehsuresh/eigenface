# Dimensionality Reduction Techniques for Face Recognition

## Overview

This repository contains a Python implementation and analysis of Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) for face recognition. The code utilizes the Carnegie Mellon University Pose, Illumination, and Expression (CMU PIE) dataset to evaluate the performance of these dimensionality reduction techniques.

## Introduction

The goal of this project is to enhance face recognition accuracy through efficient feature extraction and dimensionality reduction. The code focuses on PCA and LDA, two widely used techniques in machine learning.

### Principal Component Analysis (PCA)

PCA is employed to capture the maximum variance in the data. The `myPCA` function centralizes the data, computes the covariance matrix, performs eigen-decomposition, and selects the top eigenvectors for dimensionality reduction.

### Linear Discriminant Analysis (LDA)

LDA goes beyond PCA by not only reducing dimensionality but also enhancing class separability. The `myLDA` function calculates within-class and between-class scatter matrices, performs eigen-decomposition, and selects eigenvectors for optimal class separation.

## Dataset and Classification

The CMU PIE dataset, consisting of facial images from 68 subjects under varying lighting conditions, is utilized. The dataset is split into training and testing sets for different training sizes (5, 10, 15 samples per subject). The K-Nearest Neighbors (KNN) classifier is employed for face recognition.

## Results and Analysis

The code iterates through different training sizes, applies PCA and LDA, and records the running time and accuracy for each approach. The results are displayed in a tabular format, providing insights into the performance of PCA and LDA under various scenarios.

### PCA

PCA is applied to project the data into a reduced space, and the explained variance ratio is calculated to determine the optimal number of components.

### LDA

LDA is employed to enhance class separability. Within-Class and Between-Class Scatter matrices are computed, and the resulting eigenvectors are utilized for dimensionality reduction.

## Conclusion

This project exemplifies the synergy between dimensionality reduction and classification techniques, offering valuable insights into their combined impact on face recognition tasks. Further analysis, optimization, and exploration of additional datasets could extend the applicability and robustness of these techniques in real-world scenarios.

Feel free to explore the code and experiment with different datasets to gain a deeper understanding of PCA and LDA in the context of face recognition.
