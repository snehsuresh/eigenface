import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.sparse import issparse
from scipy.sparse.linalg import svds


def PCA(data, options=None):
    """
    Principal Component Analysis

    Parameters:
    - data: Data matrix. Each row vector of fea is a data point.
    - options.ReducedDim: The dimensionality of the reduced subspace. If 0, all the dimensions will be kept.
                          Default is 0.
    - options.PCARatio: The ratio of the sum of preserved eigenvalues to the total sum of eigenvalues.

    Returns:
    - eigvector: Each column is an embedding function. For a new data point (row vector) x,  y = x @ eigvector
                 will be the embedding result of x.
    - eigvalue: The sorted eigvalue of PCA eigen-problem.
    """

    if options is None:
        options = {}

    ReducedDim = options.get('ReducedDim', 0)

    nSmp, nFea = data.shape

    if ReducedDim > nFea or ReducedDim <= 0:
        ReducedDim = nFea

    if issparse(data):
        data = data.toarray()

    sampleMean = np.mean(data, axis=0)
    data = data - np.tile(sampleMean, (nSmp, 1))

    num_components = min(data.shape) - 1 if ReducedDim <= 0 else min(ReducedDim, min(data.shape) - 1)
    eigvector, _, eigvalue = svds(data.T, k=num_components)

    eigvalue = eigvalue ** 2

    if 'PCARatio' in options:
        sumEig = np.sum(eigvalue)
        sumEig *= options['PCARatio']
        sumNow = 0
        for idx in range(len(eigvalue)):
            sumNow += eigvalue[idx, idx]  # Access the individual element
            if sumNow >= sumEig:
                break
        eigvector = eigvector[:, :idx + 1]

    return eigvector, eigvalue


def plot_faces(faces, title):
    n_faces = faces.shape[0]
    n_cols = min(5, n_faces)
    n_rows = (n_faces // n_cols) + int(n_faces % n_cols > 0)

    plt.figure(figsize=(15, 5))
    for i in range(n_faces):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(faces[i, :].reshape((30, 30)).T, cmap='gray')
        plt.title(f"Face {i + 1}")
        plt.axis('off')

    plt.suptitle(title)
    plt.show()


mat_data = loadmat('./Assignment_2/Code&Data/PIE.mat')

# Access the variables
X = mat_data['Data']
L = mat_data['Label']
n_per = mat_data['n_per'][0, 0]
n_sub = mat_data['n_sub'][0, 0]

# Select individuals for which you want to use the complete dataset
# selected_individuals = [1, 2, 3]
#
# # Select a few sample faces
# sample_faces = X[:5, :]
# 
# # Visualize original faces
# plot_faces(sample_faces, "Original Faces")

# Perform PCA
pca_eigvector, _ = PCA(X, options={'PCARatio': 1})

# Select the first five eigenvectors
selected_eigvectors = pca_eigvector[:, :5]

# Reshape and normalize each eigenvector for visualization
eigenfaces = []
for i in range(selected_eigvectors.shape[1]):
    eigenface = selected_eigvectors[:, i].reshape((30, 30))
    eigenface = (eigenface - np.min(eigenface)) / (np.max(eigenface) - np.min(eigenface))
    eigenfaces.append(eigenface)

print("Hi")
# Plot the EigenFaces
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i, ax in enumerate(axes):
    ax.imshow(eigenfaces[i], cmap='gray')
    ax.set_title(f'EigenFace {i + 1}')
    ax.axis('off')

plt.show()
