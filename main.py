import math
from sklearn.cluster import Birch
from sklearn.cluster import KMeans

import CFTree

class BIRCH:
    def __init__(self, branching_factor=50, threshold=1.0):
        self.branching_factor = branching_factor  # Max CFs per node
        self.threshold = threshold  # Threshold for clustering
        self.root = CFTree.CFTree(branching_factor=branching_factor,
                                  threshold=threshold)

    def fit(self, X):
        self.dataPoints = X
        self.Phase1BuildCFTree()
        # print("Tree root entries:", self.root)
        self.Phase2CondenseCFTree()
        self.Phase3GlobalClustering()
        self.Phase4ClusterRefining()
    
    def predict(self, X):
        return self.kmeans.predict(X)

    def Phase1BuildCFTree(self):
        for point in self.dataPoints:
            self.root.insert(point)

    def Phase2CondenseCFTree(self):
        # Condense the CF tree by merging nodes, handling overflow, and pruning unnecessary nodes.
        # TODO: Too advanced for my stupid example
        pass

    def Phase3GlobalClustering(self):
        # Perform final clustering using CF tree data
        # 1. Extract centroids from the CF Tree
        centroids = self._extract_centroids(self.root.root)
        if not centroids:
            print("No centroids to perform k-means clustering.")
            return

        # 2. Perform K-means clustering on the extracted centroids
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        self.kmeans.fit(centroids)

        # 3. Reassign each CF to its nearest cluster
        cluster_labels = self.kmeans.labels_
        for idx, cf in enumerate(self._get_all_cfs(self.root.root)):
            cf.cluster_id = cluster_labels[idx]  # Assign the cluster ID to each CF

        for i, centroid in enumerate(self.kmeans.cluster_centers_):
            print(f"Cluster {i}: Centroid at {centroid}")

    # Helper functions for CF Tree traversal
    def _extract_centroids(self, node):
        """Recursively extract centroids from all ClusteringFeatures in the CF Tree."""
        centroids = []
        if node.is_leaf:
            for cf in node.entries:
                centroids.append(cf.centroid())
        else:
            for child in node.children:
                centroids.extend(self._extract_centroids(child))
        return centroids

    def _get_all_cfs(self, node):
        """Recursively collect all CFs from the CF Tree."""
        all_cfs = []
        if node.is_leaf:
            all_cfs.extend(node.entries)
        else:
            for child in node.children:
                all_cfs.extend(self._get_all_cfs(child))
        return all_cfs


    def Phase4ClusterRefining(self):
        # Refine the clusters, e.g., by running k-means or further splitting
        # TODO: This is also too advanced
        pass

# Example usage
dataPoints = [
    [-100, 0], 
    [1, 0], [2, 0], [3, 0], [4, 0],
    [100, 0], [101, 0], [102, 0], 
    [200, 0], [201, 0], [202, 0]
]

brch_my = BIRCH(branching_factor=2, threshold=10.0)
brch_my.fit(dataPoints)
print(brch_my.predict(dataPoints))

brc_sk = Birch(n_clusters=None, threshold=10, branching_factor=2)
brc_sk.fit(dataPoints)
print(brc_sk.predict(dataPoints))