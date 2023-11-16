# Dimensionality Reduction Techniques for Face Recognition

### Method 1 (pca_lda.py)

The Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) analyses for face recognition are implemented in Python and are available in this repository. The code assesses the effectiveness of these dimensionality reduction strategies using the Carnegie Mellon University Pose, Illumination, and Expression (CMU PIE) dataset.

The goal of this project is to enhance face recognition accuracy through efficient feature extraction and dimensionality reduction. The code focuses on PCA and LDA, two widely used techniques in machine learning.

#### Principal Component Analysis (PCA)

This project aims to improve face recognition accuracy by reducing dimensionality and extracting features in an efficient manner. The code primarily uses two popular machine learning techniques: PCA and LDA.

#### Linear Discriminant Analysis (LDA)

Beyond PCA, LDA improves class separability in addition to dimensionality reduction. The `myLDA` function chooses eigenvectors for the best possible class separation, does eigen-decomposition, and computes within- and between-class scatter matrices.

#### Dataset and Classification

The CMU PIE dataset is used, which consists of 68 subjects' face images taken in various lighting conditions. To accommodate varying training sizes (5, 10, 15 samples per subject), the dataset is divided into training and testing sets. For face recognition, the K-Nearest Neighbours (KNN) classifier is used.

#### Results and Analysis

The code applies PCA and LDA, cycles through various training sizes, and logs the accuracy and running time for each method. The tabular format of the results displays the results and offers insights into how well PCA and LDA perform in different scenarios.

#### PCA

PCA is applied to project the data into a reduced space, and the explained variance ratio is calculated to determine the optimal number of components.

#### LDA

LDA is employed to enhance class separability. Within-Class and Between-Class Scatter matrices are computed, and the resulting eigenvectors are utilized for dimensionality reduction.

#### Conclusion

This project provides insights into the combined effects of dimensionality reduction and classification techniques on face recognition tasks, and is an excellent example of how these techniques work well together. Extensive examination, refinement, and investigation of supplementary datasets may expand the suitability and resilience of these methodologies in practical situations.

To learn more about PCA and LDA in the context of face recognition, feel free to experiment with various datasets and explore the code.


### Method 2 (enhanced_pca_lda.py)

#### Linear Discriminant Analysis (LDA)

##### Modifications and Options

The code checks for specific options and modifications:
1. **Regularization (Regu):** If specified, options like regularization type (Ridge by default) and alpha are considered.
2. **Fisherface:** If specified, PCA ratio is set to the total samples minus the number of classes.
3. **Cholesky Decomposition (Chol):** Optimization if the number of samples is greater than features.

##### Implementation

1. **Data Preprocessing:** Subtracting mean and Cholesky decomposition if applicable.
2. **Singular Value Decomposition (SVD):** Perform SVD on data. Adjust eigenvectors based on PCA conditions.
3. **Class Scatter Matrix (Hb):** Calculate class scatter matrix.
4. **Final Eigen-decomposition:** Perform eigen-decomposition based on conditions and options.
5. **Normalization:** Normalize resulting eigenvectors.

#### Principal Component Analysis (PCA) Integration

Integration of PCA into the process allows for a comparison between PCA and PCA+LDA for face recognition.

##### Key Steps

1. **Data Preprocessing:** Subtract mean.
2. **SVD:** Perform SVD on data, resulting in eigenvectors and eigenvalues.
3. **Adjustment Based on Options:** Adjust eigenvectors based on specified options.
4. **Normalization:** Normalize resulting eigenvectors.

#### Results and Analysis

The code iterates through different training sizes, applies PCA, and integrates LDA. Results are recorded in a table, including method, train size, running time, and accuracy.

#### Comparison with Previous Method

Adding LDA introduces class-aware dimensionality reduction, potentially improving face recognition accuracy compared to PCA.

#### Optimization Techniques

The code incorporates optimization techniques like Cholesky decomposition and regularization.

#### Conclusion

This code expands on previous dimensionality reduction techniques, incorporating LDA. The comparison with PCA allows for understanding the impact of class-aware dimensionality reduction on face recognition tasks. The implementation introduces modifications and optimizations for a comprehensive exploration.

Feel free to experiment with different datasets and parameters for a deeper understanding of these techniques in diverse scenarios.


# Principal Component Analysis (PCA) and Eigenfaces Visualization

The Python code (eigen_faces.py) performs Principal Component Analysis (PCA) on face images from the CMU PIE dataset and visualizes the resulting eigenfaces.

## Functions:

### 1. PCA

The `PCA` function takes a data matrix as input and performs Principal Component Analysis.

- **Parameters:**
  - `data`: Data matrix where each row vector represents a data point.
  - `options.ReducedDim`: The dimensionality of the reduced subspace. If 0, all dimensions will be kept (default is 0).
  - `options.PCARatio`: The ratio of the sum of preserved eigenvalues to the total sum of eigenvalues.

- **Returns:**
  - `eigvector`: Each column is an embedding function. For a new data point `x`, `y = x @ eigvector` will be the embedding result of `x`.
  - `eigvalue`: The sorted eigenvalues of the PCA eigen-problem.

### 2. plot_faces

The `plot_faces` function visualizes a set of face images in a grid layout.

- **Parameters:**
  - `faces`: Matrix of face images.

## Code Execution:

1. **Loading Data:**
   - The code loads face images and their corresponding labels from the CMU PIE dataset.

2. **Performing PCA:**
   - PCA is applied to the entire dataset (`X`) with the option to preserve all eigenvalues (`PCARatio: 1`).
   - The first five eigenvectors are selected from the result.

3. **Visualizing Original Faces (Commented Out):**
   - There are commented-out sections for selecting specific individuals and visualizing original faces. Uncomment these sections for further exploration.

4. **Visualizing Eigenfaces:**
   - The selected eigenvectors are reshaped and normalized to create eigenfaces.
   - Eigenfaces are then plotted in a row for visualization.

5. **Displaying Plots:**
   - The final plots include a row of original faces (if uncommented) and a row of eigenfaces.

