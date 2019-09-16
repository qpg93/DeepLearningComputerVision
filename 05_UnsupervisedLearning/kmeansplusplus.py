from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

class KMeansPP(object):

    def __init__(self, centers=1, init='random', maxIter=300, tol=1e-4, random_seed=0):
        '''
        :param centers (int)        Numbers of centroids
        :param init (string)        How to init centroids.(random or k-means++)
        :param maxIter (int)        Max iterations
        :param tol (float)          Tolerance of movements after iteration
        :param random_seed (int)   Random seed
        '''
        np.random.seed(random_seed)
        self.centers = centers
        if init != 'random' and init != 'k-means++':
            raise AttributeError("Error initialization mode!")
        self.init = init
        self.maxIter = maxIter
        self.tol = tol

    def fit_predict(self, X):
        '''
        Fit training data
        :param X: array like, shape = [n_samples, n_features]
            Train data
        :return: array like, shape = [n_samples]
            Data labels
        '''
        X_norm = self._norm(X)

        if self.init == 'random':
            self.cluster_centers = self._randInit(X_norm)
        else:
            self.cluster_centers = self._kMeansPP(X_norm)
        print(cluster_centers)

        # Initialize the cluster labels
        self.labels = np.zeros(X_norm.shape[0])

        # Start iteration
        for _ in range(self.maxIter):
            