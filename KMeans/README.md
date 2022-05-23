# *KMeans Clustering Algorithm*

* Read Input data and number of clusters K
* Randomly initialize K centroids among the input samples
* Calculate distance from each sample in input data to K centroids
* Assign sample to the cluster with minimum distance from sample to cluster centroid
*  Do this for all samples
* Then take each cluster and calculate new centroid(mean of all samples of the cluster)
* Do this for K clusters to get K centroids
* Go to step 3 and repeat until there is no change in centroids.
* Plot the final clusters with their respective samples coded in different color.