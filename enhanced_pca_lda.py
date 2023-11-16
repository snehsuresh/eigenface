import numpy as np
from scipy.linalg import svd, eigh
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score
from scipy.sparse.linalg import svds
from scipy.io import loadmat
import time
from sklearn.neighbors import KNeighborsClassifier
def LDA(gnd, options, data):
    MAX_MATRIX_SIZE = 1600
    EIGVECTOR_RATIO = 0.1

    if 'Regu' not in options or not options['Regu']:
        bPCA = 1
        if 'PCARatio' not in options:
            options['PCARatio'] = 1
    else:
        bPCA = 0
        if 'ReguType' not in options:
            options['ReguType'] = 'Ridge'
        if 'ReguAlpha' not in options:
            options['ReguAlpha'] = 0.1

    nSmp, nFea = data.shape
    if len(gnd) != nSmp:
        raise ValueError('gnd and data mismatch!')

    classLabel = np.unique(gnd)
    nClass = len(classLabel)
    Dim = nClass - 1

    if bPCA and 'Fisherface' in options and options['Fisherface']:
        options['PCARatio'] = nSmp - nClass

    if issparse(data):
        data = data.toarray()
    sampleMean = np.mean(data, axis=0)
    data = data - np.tile(sampleMean, (nSmp, 1))

    bChol = 0
    if bPCA and (nSmp > nFea + 1) and (options['PCARatio'] >= 1):
        DPrime = np.dot(data.T, data)
        DPrime = np.maximum(DPrime, DPrime.T)
        R, p = np.linalg.cholesky(DPrime, lower=True)

        if p == 0:
            bPCA = 0
            bChol = 1

    if bPCA:
        U, S, Vt = svd(data, full_matrices=False)
        U, S, Vt = CutonRatio(U, S, Vt, options)
        eigvalue_PCA = np.diag(S)

        data = U
        eigvector_PCA = np.dot(Vt.T, np.diag(1 / eigvalue_PCA))
    else:
        if not bChol:
            DPrime = np.dot(data.T, data)

            if 'ReguAlpha' in options:
                options['ReguAlpha'] = nSmp * options['ReguAlpha']

                if options['ReguType'].lower() == 'ridge':
                    for i in range(DPrime.shape[0]):
                        DPrime[i, i] = DPrime[i, i] + options['ReguAlpha']
                elif options['ReguType'].lower() in ['tensor', 'custom']:
                    DPrime = DPrime + options['ReguAlpha'] * options['regularizerR']
                else:
                    raise ValueError('ReguType does not exist!')

                DPrime = np.maximum(DPrime, DPrime.T)

    nSmp, nFea = data.shape

    Hb = np.zeros((nClass, nFea))
    for i in range(nClass):
        index = np.where(gnd == classLabel[i])[0]
        classMean = np.mean(data[index, :], axis=0)
        Hb[i, :] = np.sqrt(len(index)) * classMean

    if bPCA:
        _, _, eigvector = svd(Hb, full_matrices=False)
        eigvalue = np.diag(eigvalue_PCA)

        eigIdx = np.where(eigvalue < 1e-3)[0]
        eigvalue = np.delete(eigvalue, eigIdx)
        eigvector = np.dot(eigvector_PCA, eigvector)
    else:
        WPrime = np.dot(Hb.T, Hb)
        WPrime = np.maximum(WPrime, WPrime.T)

        dimMatrix = WPrime.shape[1]
        if Dim > dimMatrix:
            Dim = dimMatrix

        if 'bEigs' in options:
            bEigs = options['bEigs']
        else:
            if (dimMatrix > MAX_MATRIX_SIZE) and (Dim < dimMatrix * EIGVECTOR_RATIO):
                bEigs = 1
            else:
                bEigs = 0

        if bEigs:
            option = {'disp': 0}
            if bChol:
                option['cholB'] = 1
                _, _, eigvector = svds(WPrime, k=Dim, which='LM', **option)
            else:
                _, eigvector = eigh(WPrime, DPrime, eigvals=(dimMatrix - Dim, dimMatrix - 1), **option)
                eigvector = np.flip(eigvector, axis=1)

            eigvalue = np.diag(eigvector)
        else:
            eigvalue, eigvector = eigh(WPrime, DPrime)
            eigvalue = np.diag(eigvalue)

            index = np.argsort(-eigvalue)
            eigvalue = eigvalue[index]
            eigvector = eigvector[:, index]

            if Dim < eigvector.shape[1]:
                eigvector = eigvector[:, :Dim]
                eigvalue = eigvalue[:Dim]

    for i in range(eigvector.shape[1]):
        eigvector[:, i] = eigvector[:, i] / np.linalg.norm(eigvector[:, i])

    return eigvector


def CutonRatio(U, S, V, options):
    if 'PCARatio' not in options:
        options['PCARatio'] = 1

    eigvalue_PCA = np.diag(S)
    if options['PCARatio'] > 1:
        idx = options['PCARatio']
        if idx < len(eigvalue_PCA):
            U = U[:, :idx]
            V = V[:, :idx]
            S = S[:idx, :idx]
    elif options['PCARatio'] < 1:
        sumEig = np.sum(eigvalue_PCA)
        sumEig = sumEig * options['PCARatio']
        sumNow = 0
        for idx in range(len(eigvalue_PCA)):
            sumNow = sumNow + eigvalue_PCA[idx]
            if sumNow >= sumEig:
                break
        U = U[:, :idx]
        V = V[:, :idx]
        S = S[:idx, :idx]

    return U, S, V

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


# Load the MATLAB dataset
mat_data = loadmat('./Assignment_2/Code&Data/PIE.mat')

# Access the variables
X = mat_data['Data']
L = mat_data['Label']
n_per = mat_data['n_per'][0, 0]
n_sub = mat_data['n_sub'][0, 0]

# Number of training data per person
train_sizes = [5, 10, 15]

# Create a table to store results
results_table = []

for train_size in train_sizes:
    trainInd = []
    testInd = []

    # Generate train/test index by selecting the first n samples from the dataset as training data
    for i in range(1, n_per + 1):
        trainInd.extend(range((i - 1) * n_sub, (i - 1) * n_sub + train_size))
        testInd.extend(range((i - 1) * n_sub + train_size, i * n_sub))

    # Generate training and testing data
    trainFea = X[trainInd, :]
    trainLabel = L[trainInd, :]
    testFea = X[testInd, :]
    testLabel = L[testInd, :]

    # PCA using existing codes
    pca_eigvector, _ = PCA(trainFea, options={'PCARatio': 1})
    pcaTrainFea = trainFea @ pca_eigvector
    pcaTestFea = testFea @ pca_eigvector

    # Record the start time for LDA
    start_time_lda = time.time()

    # Apply LDA
    lda_eigvector = LDA(trainLabel.flatten(), {'Regu': True, 'ReguAlpha': 0.1}, pcaTrainFea)
    print("Dimensions of pcaTrainFea:", pcaTrainFea.shape)
    print("Dimensions of lda_eigvector before modification:", lda_eigvector.shape)

    # Reshape lda_eigvector
    lda_eigvector = lda_eigvector.reshape((lda_eigvector.shape[0], -1))[:, :66]

    print("Dimensions of lda_eigvector after modification:", lda_eigvector.shape)

    ldaTrainFea = pcaTrainFea @ lda_eigvector
    ldaTestFea = pcaTestFea @ lda_eigvector
    # Ensure the dimensions match
    if lda_eigvector.shape[1] > pcaTrainFea.shape[1]:
        lda_eigvector = lda_eigvector[:, :pcaTrainFea.shape[1]]

    ldaTrainFea = pcaTrainFea @ lda_eigvector

    # Record the end time for LDA
    end_time_lda = time.time()

    # Call nearest neighbor classifier of scikit-learn with PCA+LDA transformed data
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(ldaTrainFea, trainLabel.flatten())
    predictLabel = knn.predict(ldaTestFea)

    # Calculate accuracy with PCA+LDA
    accuracy_lda = accuracy_score(testLabel.flatten(), predictLabel)

    # Record the running time and accuracy for PCA+LDA
    results_table.append({
        'Method': 'PCA+LDA',
        'Train Size': train_size,
        'Running Time': end_time_lda - start_time_lda,
        'Accuracy': accuracy_lda
    })

# Print the results table
print("Results Table:")
print("Method\tTrain Size\tRunning Time\tAccuracy")
for row in results_table:
    print(f"{row['Method']}\t{row['Train Size']}\t\t{row['Running Time']:.4f}\t\t{row['Accuracy']:.4f}")
