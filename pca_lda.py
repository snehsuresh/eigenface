import numpy as np
from scipy.linalg import eig
import scipy.io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time

def myPCA(X, d):
    eigvector = np.empty((0, d))
    eigvalue = np.empty((0, d))

    # Centralize X by subtracting the mean vector from each sample
    meanX = np.mean(X, axis=1, keepdims=True)
    X_centered = X - meanX

    # Compute the covariance matrix of X after centralization
    C = np.cov(X_centered) #Xtranspose.X in your notes

    # Compute eigen-decomposition of covariance matrix and return the first d
    # eigenvectors and eigenvalues
    eigvalue, eigvector = np.linalg.eig(C)

    # Sort eigenvectors in descending order based on eigenvalues
    sorted_indices = np.argsort(eigvalue)[::-1]
    eigvector = eigvector[:, sorted_indices]

    # Select the first d eigenvectors
    eigvector = eigvector[:, :d]
    eigvector = np.real(eigvector)
    eigvalue = np.real(eigvalue)

    return eigvector

def myLDA(X, L, d1, d2):
    # X is a D by n data matrix where D is the dimension of the original
    # feature space, and n is the number of samples.
    # L is a n by 1 label vector. Check latex doc to know more.
    # d1 is the dimensionality of data after PCA
    # d2 is the intended number of eigenvectors to return
    # eigvector and eigvalue are the results of eigen-decomposition of the
    # objective function of LDA
    eigvector = np.empty((0, d2))
    eigvalue = np.empty((0, d2))
    Sw = np.zeros((X.shape[0], X.shape[0])) #Within Class
    Sb = np.zeros((X.shape[0], X.shape[0])) #Between Class
    for i in np.unique(L):
        Xi = X[:, L.flatten() == i]
        meanXi = np.mean(Xi, axis=1, keepdims=True)
        Sw += np.cov(Xi, rowvar=True) * (Xi.shape[1] - 1)

    ## Compute between-class scatter matrix Sb
    overall_mean = np.mean(X, axis=1, keepdims=True)
    for i in np.unique(L):
        Xi = X[:, L.flatten() == i]
        meanXi = np.mean(Xi, axis=1, keepdims=True)
        Sb += Xi.shape[1] * (meanXi - overall_mean) @ (meanXi - overall_mean).T

    ## Compute the eigen-decomposition
    eigvalues, eigvectors = eig(np.linalg.inv(Sw) @ Sb)

    ## Sort eigenvectors based on eigenvalues
    sorted_indices = np.argsort(eigvalues.real)[::-1]
    eigvectors = eigvectors[:, sorted_indices]
    eigvalues = eigvalues[sorted_indices]

    ## Select the first d2 eigenvectors
    eigvector = eigvectors[:, :d2]
    eigvalue = eigvalues[:d2]
    eigvector = np.real(eigvector)
    eigvalue = np.real(eigvalue)

    return eigvector, eigvalue

# Load the MATLAB dataset
mat_data = scipy.io.loadmat('./Assignment_2/Code&Data/PIE.mat')

# Access the variables
X = mat_data['Data']
L = mat_data['Label']
n_per = mat_data['n_per'][0, 0]
n_sub = mat_data['n_sub'][0, 0]
print("Shape of X:", X.shape)
print("Shape of L:", L.shape)
# Specify trainNum values
trainNum_values = [5, 10, 15]
knn = KNeighborsClassifier(n_neighbors=1)
results_table = []

for trainNum in trainNum_values:
    # Separate data into training and testing sets
    X_train = []
    L_train = []
    X_test = []
    L_test = []
    count = 0
    for i in np.unique(L):

        # Extract indices of samples for the current class
        indices = np.where(L.flatten() == i)[0]
        count += len(indices)
        if count > X.shape[1]:
            break
        # Ensure the requested training size does not exceed the available samples
        if trainNum >= len(indices):
            print(
                f"Warning: Train size ({trainNum}) exceeds the available samples for class {i} ({len(indices)}). Skipping this class.")
            continue

        # Split samples into training and testing
        X_train.append(X[:, indices[:trainNum]])
        L_train.extend([i] * trainNum)

        X_test.append(X[:, indices[trainNum:]])
        L_test.extend([i] * (len(indices) - trainNum))
    X_train = np.concatenate(X_train, axis=1)
    X_test = np.concatenate(X_test, axis=1)
    L_train = np.array(L_train)
    L_test = np.array(L_test)

    # Measure time for PCA
    start_time_pca = time.time()
    d_pca = X_train.shape[1]
    d_pca = 100
    # Determine the number of classes in your dataset
    n_classes = len(np.unique(L))
    # Decide on the value of d_lda (less than or equal to n_classes - 1)
    d_lda = min(d_pca, n_classes - 1)

    # Apply PCA
    pca_eigvector = myPCA(X_train, 100) ## Use all components initially
    print(pca_eigvector)
    # Calculate explained variance
    # This line calculates the cumulative sum of the absolute values of the diagonal elements of the covariance matrix of the principal components. This cumulative sum represents the cumulative explained variance as you increase the number of components. It's essentially a measure of how much information each principal component captures.
    # explained_variance_ratio = np.cumsum(np.abs(np.diag(pca_eigvector.T @ pca_eigvector))) / np.sum(
    #     np.abs(np.diag(pca_eigvector.T @ pca_eigvector)))
    #
    # # Find the number of components that explain at least 70% variance
    # d_pca = np.argmax(explained_variance_ratio >= 0.70) + 1

    # print(d_pca)
    # # Project data onto the reduced space
    # X_train_pca = pca_eigvector[:, :d_pca].T @ X_train
    # X_test_pca = pca_eigvector[:, :d_pca].T @ X_test

    X_train_pca = pca_eigvector.T @ X_train
    X_test_pca = pca_eigvector.T @ X_test

    end_time_pca = time.time()

    # Train the KNN classifier with PCA-transformed data
    knn.fit(X_train_pca.T, L_train)

    # Make predictions on the test set with PCA-transformed data
    predictions_pca = knn.predict(X_test_pca.T)

    # Calculate accuracy with PCA
    accuracy_pca = accuracy_score(L_test, predictions_pca)

    # Record the running time and accuracy for PCA
    results_table.append({
        'Method': 'PCA',
        'Train Size': trainNum,
        'Running Time': end_time_pca - start_time_pca,
        'Accuracy': accuracy_pca
    })

    # Record the start time for LDA
    start_time_lda = time.time()

    # Apply LDA
    lda_eigvector, _ = myLDA(X_train, L_train, d_pca, d_lda)
    X_train_lda = lda_eigvector.T @ X_train
    X_test_lda = lda_eigvector.T @ X_test

    # Record the end time for LDA
    end_time_lda = time.time()

    # Train the KNN classifier with LDA-transformed data
    knn.fit(X_train_lda.T, L_train)

    # Make predictions on the test set with LDA-transformed data
    predictions_lda = knn.predict(X_test_lda.T)

    # Calculate accuracy with LDA
    accuracy_lda = accuracy_score(L_test, predictions_lda)

    # Record the running time and accuracy for LDA
    results_table.append({
        'Method': 'LDA',
        'Train Size': trainNum,
        'Running Time': end_time_lda - start_time_lda,
        'Accuracy': accuracy_lda
    })

# Print the results table
print("Results Table:")
print("Method\tTrain Size\tRunning Time\tAccuracy")
for row in results_table:
    print(f"{row['Method']}\t{row['Train Size']}\t\t{row['Running Time']:.4f}\t\t{row['Accuracy'] * 100:.2f}%")
