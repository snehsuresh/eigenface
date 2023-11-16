# Dimensionality Reduction Techniques for Face Recognition

## Overview

The Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) analyses for face recognition are implemented in Python and are available in this repository. The code assesses the effectiveness of these dimensionality reduction strategies using the Carnegie Mellon University Pose, Illumination, and Expression (CMU PIE) dataset.

The goal of this project is to enhance face recognition accuracy through efficient feature extraction and dimensionality reduction. The code focuses on PCA and LDA, two widely used techniques in machine learning.

### Principal Component Analysis (PCA)

This project aims to improve face recognition accuracy by reducing dimensionality and extracting features in an efficient manner. The code primarily uses two popular machine learning techniques: PCA and LDA.

### Linear Discriminant Analysis (LDA)

Beyond PCA, LDA improves class separability in addition to dimensionality reduction. The `myLDA` function chooses eigenvectors for the best possible class separation, does eigen-decomposition, and computes within- and between-class scatter matrices.

## Dataset and Classification

The CMU PIE dataset is used, which consists of 68 subjects' face images taken in various lighting conditions. To accommodate varying training sizes (5, 10, 15 samples per subject), the dataset is divided into training and testing sets. For face recognition, the K-Nearest Neighbours (KNN) classifier is used.

## Results and Analysis

The code applies PCA and LDA, cycles through various training sizes, and logs the accuracy and running time for each method. The tabular format of the results displays the results and offers insights into how well PCA and LDA perform in different scenarios.

### PCA

PCA is applied to project the data into a reduced space, and the explained variance ratio is calculated to determine the optimal number of components.

### LDA

LDA is employed to enhance class separability. Within-Class and Between-Class Scatter matrices are computed, and the resulting eigenvectors are utilized for dimensionality reduction.

## Conclusion

This project provides insights into the combined effects of dimensionality reduction and classification techniques on face recognition tasks, and is an excellent example of how these techniques work well together. Extensive examination, refinement, and investigation of supplementary datasets may expand the suitability and resilience of these methodologies in practical situations.

To learn more about PCA and LDA in the context of face recognition, feel free to experiment with various datasets and explore the code.
