import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class kMeans:
    def __init__(self, K, max_iters=100, plot_steps=False) -> None:
        '''
        args:
            K: number of clusters
            max_iters: maximum number of iterations
            plot_steps: if True, plot the clusters

        '''

        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # initialize clusters as list of lists (one list for each cluster)
        self.clusters = [[] for _ in range(K)] 

        # centroids are the means of the clusters
        self.centroids = []
    
    def predict(self, X):
        '''
        args:
            X: numpy array of shape (n_samples, n_features)
        - returns:
            labels: numpy array of shape (n_samples,)

        '''

        self.X = X # X is the input data
        self.n_samples, self.n_features = self.X.shape # n_samples is the number of samples, n_features is the number of features

        # initialize centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]


        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            centroids_old = self.centroids
            self._get_centroids(self.clusters)
            if self._is_converged(centroids_old):
                break
        return self._get_cluster_labels()
   
    def _create_clusters(self, centroids): 
        '''
        args:
            centroids: list of centroids
        - returns:
            clusters: list of lists (one list for each cluster)
        '''

        clusters = [[] for _ in range(self.K)]

        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        
        return clusters
        
    def _closest_centroid(self, sample, centroids):

        '''
        args:
            sample: numpy array of shape (n_features,)
            centroids: list of centroids
        - returns:
            centroid_idx: index of the closest centroid
        '''

        return np.argmin([euclidean_distance(sample,centroid) for centroid in centroids])

    def _get_centroids(self, clusters):
        '''
        args:
            clusters: list of lists (one list for each cluster)
        - returns:
            centroids: list of centroids
        '''

        for cluster_idx, cluster in enumerate(clusters):
            self.centroids[cluster_idx] = np.mean(self.X[cluster], axis=0)

        return self.centroids

    def _is_converged(self, centroids_old):
        '''
        args:
            centroids_old: list of old centroids
        '''
        return np.array_equal(self.centroids, centroids_old)

    def _get_cluster_labels(self):
        '''
        - returns:
            labels: numpy array of shape (n_samples,)

        '''

        labels = np.empty(self.n_samples)
        for idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx]=idx
        return labels

    def plot(self):
        '''
        plots the clusters

        '''
        fig, ax = plt.subplots(figsize=(12,8))

        for idx, cluster in enumerate(self.clusters):
            ax.scatter(self.X[cluster,0], self.X[cluster,1],label=f'Cluster{idx}')
        
        for point in self.centroids:
            ax.scatter(*point,color='black', linewidths=5)
        plt.show()


if __name__=="__main__":

    from sklearn.datasets import make_blobs
    
    X, y = make_blobs(n_samples=500, n_features=2, centers=3, cluster_std=1.5, random_state=5)

    clusters = len(np.unique(y))

    kmeans = kMeans(K=clusters, plot_steps=True)
    labels = kmeans.predict(X)

    kmeans.plot()

            

