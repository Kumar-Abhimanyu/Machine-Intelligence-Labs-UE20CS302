import numpy as np


class KMeansClustering:
    """
    K-Means Clustering Model

    Args:
        n_clusters: Number of clusters(int)
    """

    def __init__(self, n_clusters, n_init=10, max_iter=1000, delta=0.001):

        self.n_cluster = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.delta = delta

    def init_centroids(self, data):
        idx = np.random.choice(
            data.shape[0], size=self.n_cluster, replace=False)
        self.centroids = np.copy(data[idx, :])

    def fit(self, data):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix(M data points with D attributes each)(numpy float)
        Returns:
            The object itself
        """
        if data.shape[0] < self.n_cluster:
            raise ValueError(
                'Number of clusters is grater than number of datapoints')

        best_centroids = None
        m_score = float('inf')

        for _ in range(self.n_init):
            self.init_centroids(data)

            for _ in range(self.max_iter):
                cluster_assign = self.e_step(data)
                old_centroid = np.copy(self.centroids)
                self.m_step(data, cluster_assign)

                if np.abs(old_centroid - self.centroids).sum() < self.delta:
                    break

            cur_score = self.evaluate(data)

            if cur_score < m_score:
                m_score = cur_score
                best_centroids = np.copy(self.centroids)

        self.centroids = best_centroids

        return self

    def get_euclidean_distance(A_matrix, B_matrix):
    # """
    #     Function computes euclidean distance between matrix A and B.
    #     E. g. C[2,15] is distance between point 2 from A (A[2]) matrix and point 15 from matrix B (B[15])
    #     Args:
    #         A_matrix (numpy.ndarray): Matrix size N1:D
    #         B_matrix (numpy.ndarray): Matrix size N2:D

    #     Returns:
    #         numpy.ndarray: Matrix size N1:N2
    # """

        A_square = np.reshape(np.sum(A_matrix * A_matrix, axis=1), (A_matrix.shape[0], 1))
        B_square = np.reshape(np.sum(B_matrix * B_matrix, axis=1), (1, B_matrix.shape[0]))
        AB = A_matrix @ B_matrix.T

        C = -2 * AB + B_square + A_square

        return np.sqrt(C)
        

    def e_step(self, data):
        """
        Expectation Step.
        Finding the cluster assignments of all the points in the data passed
        based on the current centroids
        Args:
            data: M x D Matrix (M training samples with D attributes each)(numpy float)
        Returns:
            Cluster assignment of all the samples in the training data
            (M) Vector (M number of samples in the train dataset)(numpy int)
        """
        #TODO
        distance = []
        data_length = len(data)
        centroid_length = len(self.centroids)

        for i in range(data_length):
            for j in range(centroid_length):
                distance.append(np.linalg.norm(self.centroids[j]-data[i]))
        distance = np.array(distance).reshape(data_length, centroid_length)

        clusterAssigned = []

        for i in range(data_length):
            clusterAssigned.append(np.argmin(distance[i]))
        
        return clusterAssigned

    def m_step(self, data, cluster_assgn):
        """
        Maximization Step.
        Compute the centroids
        Args:
            data: M x D Matrix(M training samples with D attributes each)(numpy float)
        Change self.centroids
        """
        #TODO
        row_length = len(self.centroids)
        column_length = len(self.centroids[0])

        computedCenter = np.zeros(shape=(row_length,column_length))
        
        counter = 0

        for i in cluster_assgn:
            computedCenter[i] = np.add(computedCenter[i],data[counter])
            counter += 1

        cluster_assgn = np.array(cluster_assgn)

        for k in range(len(computedCenter)):
            count = np.count_nonzero(cluster_assgn == k)
            computedCenter[k] = computedCenter[k]/count
        
        self.centroids = computedCenter

    def evaluate(self, data):
        """
        K-Means Objective
        Args:
            data: Test data (M x D) matrix (numpy float)
        Returns:
            metric : (float.)
        """
        #TODO
        distance = []
        data_length = len(data)
        centroid_length = len(self.centroids)

        for i in range(data_length):
            for j in range(centroid_length):
                distance.append((self.centroids[j]-data[i])*(self.centroids[j]-data[i]))
        distance = np.sum(distance, axis=1)
        finalVal = 0
        for i in distance:
            finalVal += i
        
        return finalVal
        

        