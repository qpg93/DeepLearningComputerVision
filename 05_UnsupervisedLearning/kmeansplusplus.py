from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

class KMeans(object):

    def __init__(self, centroids=1, init='random', maxIter=300, tol=1e-4, random_seed=0):
        '''
        :param centroids: int
            Numbers of centroids
        :param init: string
            Method of centroids initialization (random or k-means++)
        :param maxIter: int
            Max iterations
        :param tol: float
            Tolerance of movements after iteration
        :param random_seed: int
            Random seed
        '''
        np.random.seed(random_seed)
        self.centroids = centroids
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
        # Normalize data
        X_norm = self._norm(X)

        # Choose method of centroid initialization
        if self.init == 'random':
            self.cluster_centroids = self._randInit(X_norm)
        else:
            self.cluster_centroids = self._kMeansPP(X_norm)

        # Initialize the cluster labels
        self.labels = np.zeros(X_norm.shape[0])
        # Start iteration
        for _ in range(self.maxIter):
            for i in range(self.centroids): # For each cluster
                self.labels = self._get_label(X_norm)
                # Calculate new centroids with the mean of all points in the same cluster
                new_centroid = np.mean(X_norm[self.labels == i], axis=0)
                # If centroid movement shorter than threshold, the algorithm converges
                if self._euc_dist(new_centroid, self.cluster_centroids[i, :]) < self.tol:
                    break
                # Update centroids if centroid movement is beyond tolerance
                self.cluster_centroids[i, :] = new_centroid

        # Centroids were normalized and it is undone here
        self.cluster_centroids = self._reverse_norm(X, self.cluster_centroids)
        return self.labels

    def _get_label(self, X):
        '''
        Get the closest centroids' labels
        :param X: shape = [n_samples, n_features]
            Train data
        :return: shape = [n_samples]
            Label of centroids which each point closest with
        '''
        size = X.shape[0]
        self.dist = np.zeros((size, self.centroids))
        for i in range(size):
            for j in range(self.centroids):
                # Calculate Euclidean distance
                self.dist[i, j] = self._euc_dist(X[i, :], self.cluster_centroids[j, :])

        # Find the indices of minimum values along the axis
        label = np.argmin(self.dist, axis=1)
        return label

    def _norm(self, X):
        '''
        Normalize data
            MinMax
        :param X: Data
        :return:
        '''
        norm = (X - np.min(X)) / (np.max(X) - np.min(X))
        return norm

    def _reverse_norm(self, x, norm):
        '''
        For cluster centroids
        :param x: array like
            Train data
        :param norm: array like
            Point after normalization
        :return: array like
            Data before scale
        '''
        return norm * (np.max(x) - np.min(x)) + np.min(x)

    def _euc_dist(self, x, y):
        '''
        Euclidean distance
        :param x: Point1
        :param y: Point2
        :return: Euclidean Metric
        '''
        return np.sqrt(np.sum((x - y) ** 2))

    def _man_dist(self, x, y):
        '''
        Manhattan distance (for test)
        :param x: Point1
        :param y: Point2
        :return: Manhattan distance
        '''
        return np.sum(np.abs(x - y))

    """
    def _distort_func(self):
        '''
        Distortion Function (to find a good k)
        :param X: array like
            Data with label
        :return: float
            Sum of Euclidean distance of each cluster
        '''
        distort = 0
        for i in range(self.centroids):
            distort += np.sum(self.dist[self.labels == i, i])
        return distort
    """

    def _kMeansPP(self, X):
        '''
        K-Means++
        :param X: array like, shape=[n_samples, n_features]
            Train data
        :return: array like, shape=[n_centers, n_features]
            Initial centroids
        '''
        tmp_centroids = []
        # Select randomly a point as the initial/first centroid
        tmp_centroids.append(X[np.random.randint(0, X.shape[0]), :])
        # dist: the shortest distance from each point to centroid
        dist = np.zeros(X.shape[0])
        for _ in range(1, self.centroids):
            sum = 0.0
            # Calculate shortest distance of each sample then sum up
            for i, pts in enumerate(X):
                dist[i] = self._get_minDist(pts, tmp_centroids)
                sum += dist[i]
            # Equal probability between (0, sum), 0 everywhere else
            sum = np.random.uniform(0, sum)
            # Bigger the distance, bigger the weight in probability
            # More possible to select a farther point to be centroid
            for i, d in enumerate(dist):
                sum -= d
                if sum < 0.0:
                    tmp_centroids.append(X[i, :])
                    break
        return np.array(tmp_centroids)

    def _get_minDist(self, pts, centroids):
        '''
        Get the closest distance of centroids and points
        :param pts: Points
        :param centroids: Current centroids array.
        :return:
        '''
        minDist = np.inf
        for i, centroid in enumerate(centroids):
            dist = self._euc_dist(centroid, pts)
            if dist < minDist:
                minDist = dist
        return minDist

    def _randInit(self, X):
        '''
        Random initialize centroids
        :param X: array like
            Train data
        :return: array like, shape = [k, n_features]
            Random centroids
        '''
        m = X.shape[1]
        centroids = np.zeros((self.centroids, m))
        for i in range(m):
            centroids[:, i] = np.random.uniform(min(X[:, i]), max(X[:, i]), size=self.centroids)
        return centroids

if __name__ == '__main__':
    # Generate isotropic Gaussian blobs for clustering
    X, y = make_blobs(n_samples=200, n_features=2, centers=3,
                      cluster_std=0.5, shuffle=True, random_state=0)

    plt.scatter(X[:, 0], X[:, 1], c='white', marker='o', edgecolors='k', s=50)

    kmr = KMeans(centroids=3, init='k-means++', maxIter=300, tol=1e-04, random_seed=15)
    y_km = kmr.fit_predict(X)

    plt.scatter(X[y_km == 0, 0], X[y_km == 0, 1],
                s=50, color='lightgreen', edgecolors='k', marker='s', label='cluster1')

    plt.scatter(X[y_km == 1, 0], X[y_km == 1, 1],
                s=50, color='lightblue', edgecolors='k', marker='v', label='cluster2')

    plt.scatter(X[y_km == 2, 0], X[y_km == 2, 1],
                s=50, color='orange', edgecolors='k', marker='o', label='cluster3')

    plt.scatter(kmr.cluster_centroids[:, 0], kmr.cluster_centroids[:, 1], s=250,
                marker='*', color='r', edgecolors='k', label='centroid')

    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()