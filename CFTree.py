from pprint import pformat
import numpy as np

class ClusteringFeature:
    def __init__(self, n=0, ls=None, ss=None):
        self.n = n  # Number of points
        self.ls = ls if ls is not None else [0.0, 0.0]  # Linear Sum (for 2D points as an example)
        self.ss = ss if ss is not None else [0.0, 0.0]  # Square Sum (for 2D points)

    def add_point(self, point):
        """Add a point to the clustering feature."""
        self.n += 1
        self.ls = [self.ls[i] + point[i] for i in range(len(point))]
        self.ss = [self.ss[i] + point[i] ** 2 for i in range(len(point))]

    def merge(self, other_cf):
        """Merge another ClusteringFeature into this one."""
        self.n += other_cf.n
        self.ls = [self.ls[i] + other_cf.ls[i] for i in range(len(self.ls))]
        self.ss = [self.ss[i] + other_cf.ss[i] for i in range(len(self.ss))]

    def centroid(self):
        return [x / self.n for x in self.ls]

    def radius(self):
        """Calculate the radius of the cluster."""
        centroid = self.centroid()
        variance = sum((self.ss[i] / self.n - centroid[i] ** 2) for i in range(len(self.ls)))
        return np.sqrt(variance)

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return f"ClusteringFeature(n={self.n}, ls={self.ls}, ss={self.ss})"


class CFNode:
    def __init__(self, is_leaf=True, max_entries=4):
        self.is_leaf = is_leaf
        self.max_entries = max_entries
        self.entries = []  # List of ClusteringFeatures
        self.children = []  # Pointers to child nodes

    def is_full(self):
        return len(self.entries) >= self.max_entries

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        entries_str = pformat(self.entries, indent=2, width=80)
        children_str = pformat(self.children, indent=2, width=80)
        node_type = "Leaf" if self.is_leaf else "Internal"
        return (
            f"{node_type} Node (\n"
            f"  Max Entries: {self.max_entries},\n"
            f"  Entries: {entries_str},\n"
            f"  Children: {children_str}\n"
            f")"
        )


class CFTree:
    def __init__(self, threshold=1.0, branching_factor=4):
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.root = CFNode()

    def insert(self, point):
        """Insert a point into the CF Tree."""
        self._insert_into_node(self.root, point)

    def _insert_into_node(self, node, point):
        # 1. Identifying the Appropriate Leaf Node
        if node.is_leaf:
            # Base case
            # Find or create a cluster in the leaf to add the point
            for cf in node.entries:
                temp_cf = ClusteringFeature(cf.n, cf.ls[:], cf.ss[:])
                temp_cf.add_point(point)
                radius = temp_cf.radius()
                
                if radius <= self.threshold:
                    # 2. Modifying the Leaf Node
                    cf.add_point(point)
                    return
            
            # If no existing cluster can absorb the point, create a new CF
            new_cf = ClusteringFeature()
            new_cf.add_point(point)
            node.entries.append(new_cf)

            # TODO: recursively update the parent to update the centroid summary information

            # Handle overflow
            if node.is_full():
                self._split_node(node)
        else:
            # Recursive step
            # Find the child node whose cluster is closest to the point
            i = self._find_closest_child(node, point)
            self._insert_into_node(node.children[i], point)
            
            # 3. Splitting the Leaf Node (if needed)
            if node.is_full():
                self._split_node(node)

    def _find_closest_child(self, node, point):
        """Find the child whose CF is closest to the point among the children of a node (not the children!)."""
        min_distance = float("inf")
        closest_i = -1

        for i, cf in enumerate(node.entries):
            # Calculate euclidean distance, without using the temporary data structure
            centroid = cf.centroid()
            distance = np.sqrt(sum((centroid[j] - point[j]) ** 2 for j in range(len(point)))) 

            if distance < min_distance:
                min_distance = distance
                closest_i = i

        return closest_i

    def _split_node(self, node):
        """Split a node into two and adjust the tree structure."""
        if node.is_leaf:
            # Split leaf node
            mid = len(node.entries) // 2
            # TODO: fix this, BIRCH requires selecting the farthest pair of entries as seeds for splitting
            # To work this around, entries to the implementation should be filled in in ascending order
            new_node = CFNode(is_leaf=True, max_entries=self.branching_factor)
            new_node.entries = node.entries[mid:]
            node.entries = node.entries[:mid]
        else:
            # TODO
            print("INCORRECT RESULTS! Split internal node needs to be implemented")

        self._modify_path_to_leaf(node, new_node)
    
    def _modify_path_to_leaf(self, node, new_node):
        # 4. Modifying the Path to the Leaf
        # Update parent
        if node == self.root:
            # Create new root (internal node)
            new_root = CFNode(is_leaf=False, max_entries=self.branching_factor)
            
            # For each child (original node and new_node), create a CF entry 
            # that merges ALL CFs in that child's entries
            for child in [node, new_node]:
                merged_cf = ClusteringFeature()
                for cf in child.entries:
                    merged_cf.merge(cf)  # Merge all CFs in the child
                new_root.entries.append(merged_cf)
            
            # Link children to the new root
            new_root.children = [node, new_node]
            self.root = new_root
        else:
            # TODO
            print("INCORRECT RESULTS! In real implementation, handle the parent adjustment recursively")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self._print_node(self.root, depth=0)

    def _print_node(self, node, depth=0):
        """Recursively pretty-print the tree."""
        indent = "  " * depth
        result = f"{indent}{'Leaf' if node.is_leaf else 'Internal'} Node:\n"
        result += f"{indent}  Entries: [\n"
        for entry in node.entries:
            result += f"{indent}    {entry},\n"
        result += f"{indent}  ]\n"

        if not node.is_leaf:
            result += f"{indent}  Children:\n"
            for child in node.children:
                result += self._print_node(child, depth + 1)
        return result


if __name__ == "__main__":

    # Testing ClusteringFeature
    P1 = [1, 0]
    P2 = [2, 0]
    a = ClusteringFeature()
    a.add_point(P1)
    b = ClusteringFeature()
    b.add_point(P2)

    print("Cluster features:")
    print(a)
    print(b)

    print("Now let's merge them:")
    a.merge(b)
    print(a)
    print("Centroid:", a.centroid())
    print("Radius:", a.radius())


    # Testing CFTree
    tree = CFTree(threshold=4, branching_factor=4)

    # Insertion should be done in ascending order for this implementation to work

    tree.insert([1, 0])
    tree.insert([2, 0])

    tree.insert([10, 0])
    tree.insert([10, 0])

    tree.insert([40, 0])
    tree.insert([40, 0])

    tree.insert([100, 0]) # This is going to split the node
    tree.insert([101, 0]) # This is going to use _find_closest_child and call _insert_into_node recursively

    tree.insert([110, 0]) # This will create a new node in the children

    tree.insert([3, 0]) # This will be embedded into an existing children
    
    print("TREE:")
    print(tree)
