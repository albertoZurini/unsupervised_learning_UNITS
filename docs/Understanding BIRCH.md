Balanced Iterative Reducing and Clustering using Hierarchies is an unsupervised data-mining algorithm used to perform hierarchical clustering.

# CF-Tree
Cluster feature
Having a cluster $c_j = \{x_{i,j}, ..., x_{N,j}\}$, where $x_{i,j} = \lt v_{1, 1, j}, ..., v_{p, i, j} \gt$ a point in the p-dimensional space.
The corresponding cluster feature is $CF(c_{j}) = \lt N_j, Sum_j, SS_j \gt$ 
Where:
- $(Sum_j)_k = \sum_{i=1}^{N_j} v_{k, i, j}$
- $(SS_j)_k = \sum_{i=1}^{N_j} v_{k,i,j}^2$

Claim they make: those features are enough to do some sort of hierarchical clustering without knowing the actual identity of the points.
Suppose I have the features of two clusters: what can I do?
- Can I compute the distance between the two clusters? If I divide Sum/Number of points I get the centroids, thus I can calculate the distance between
- I can calculate the radius/diameter of a cluster
- You can't do simple link, complete link.

The whole point of the thing is that I can do data reduction.

# Data reduction

1. For each point in the dataset, I compute the CF
	1. $P_1 \rightarrow CF = \lt 1, P_1, P_1^2 \gt$
	2. $P_2 \rightarrow CF = \lt 2, P_1+P_2, P_1^2+P_2^2 \gt$ (only if $P_2$ is not too far from $P_1$)
	3. I do this as long as I have memory. I then replace in the original dataset a CF that summarizes the data points. In this way when a new point comes in, I just put it into a CF or otherwise treat it as an outlier.
2. 