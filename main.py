import math
from sklearn.cluster import Birch

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
        print("Tree root entries:", self.root)
        self.Phase2CondenseCFTree()
        self.Phase3GlobalClustering()
        self.Phase4ClusterRefining()
    
    def predict(self, X):
        pass

    def Phase1BuildCFTree(self):
        for point in self.dataPoints:
            self.root.insert(point)

    def Phase2CondenseCFTree(self):
        """
        Condense the CF tree by merging nodes, handling overflow, and pruning unnecessary nodes.
        """
        # Too advanced for my stupid example
        pass

    def Phase3GlobalClustering(self):
        # Perform final clustering using CF tree data
        pass

    def Phase4ClusterRefining(self):
        # Refine the clusters, e.g., by running k-means or further splitting
        pass

    # Optional helper method: Calculate the distance between CFs
    def calculate_distance(self, cf1, cf2):
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