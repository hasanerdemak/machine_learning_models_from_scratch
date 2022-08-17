import random

class KMeansClusterClassifier:
    def __init__(self, n_cluster):
        self.K = n_cluster # cluster number
        self.max_iterations = 100 # max iteration
        
    # fit data
    def fit(self, X):
        num_examples = len(X) 
        num_features = len(X[0])
        
        self.centroids = centroids = [[] for _ in range(self.K)] # create centroids list
        # initialize random centroids
        for k in range(self.K):
            centroid = X[random.choice(range(num_examples))] # random centroids
            centroids[k] = centroid

        # create cluster
        for _ in range(self.max_iterations):
            clusters = [[] for _ in range(self.K)]
            for point_idx, point in enumerate(X):
                closest_centroid = self._get_closest_centroid_index(point, centroids) # Closest centroid index using Euclidean Distance
                clusters[closest_centroid].append(point_idx)

            previous_centroids = centroids

            any_change = False
            # Calculate new centroids
            for idx, cluster in enumerate(clusters):
                total = [0,0,0,0]
                for i in cluster:
                    total[0] += X[i][0]
                    total[1] += X[i][1]
                    total[2] += X[i][2]
                    total[3] += X[i][3]
                
                # find the value for new centroids
                if len(cluster) != 0:
                    new_centroid = [(total[i]/len(cluster)) for i in range(num_features)]

                # Observe difference. If there is no difference between old and new centroids, then break the loop
                list_difference = [item for item in new_centroid if item not in centroids[idx]]
                if list_difference != []:
                    any_change = True
                    centroids[idx] = new_centroid
            
            if any_change == False:
                break

        self.clusters = clusters
        self.centroids = centroids

    # Returns the index of the centroid closest to the point based on Euclidean distance
    def _get_closest_centroid_index(self, point, centroids):
        distances = []
        for i in range(len(centroids)):
            total = 0
            for j in range(len(point)): #4
                diff = centroids[i][j] - point[j]
                total += diff * diff
            distances.append(total**(1/2))

        return distances.index(min(distances))
    
    # returns sum of squared distances of points from their closest centroid
    def get_sum_of_distances_from_their_closest_centroids(self, X):
        squared_distances = []
        for point_idx, point in enumerate(X):
            closest_centroid = self._get_closest_centroid_index(point, self.centroids)
            total = 0
            for j in range(len(point)): #4
                diff = self.centroids[closest_centroid][j] - point[j]
                total += diff * diff
            squared_distances.append(total)
        
        sum = 0
        for i in range(len(X)):
            sum += squared_distances[i]
        return sum

    # returns predicted labels
    def predict(self, X):
        y_pred = [0 for _ in range(len(X))] # fill with zero
        for point_idx, point in enumerate(X):
            closest_centroid = self._get_closest_centroid_index(point, self.centroids)
            y_pred[point_idx] = closest_centroid
        return y_pred