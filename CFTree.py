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
        radius = sum(
            np.sqrt((self.ss[i] / self.n) - centroid[i] ** 2) for i in range(len(self.ls))
        )
        return radius

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
        if node.is_leaf:
            # Find or create a cluster in the leaf to add the point
            for cf in node.entries:
                temp_cf = ClusteringFeature(cf.n, cf.ls[:], cf.ss[:])
                temp_cf.add_point(point)
                if temp_cf.radius() <= self.threshold:
                    cf.add_point(point)
                    return
            # If no existing cluster can absorb the point, create a new CF
            new_cf = ClusteringFeature()
            new_cf.add_point(point)
            node.entries.append(new_cf)
            # Handle overflow
            if node.is_full():
                self._split_node(node)
        else:
            # Find the child node whose cluster is closest to the point
            closest_idx = self._find_closest_child(node, point)
            self._insert_into_node(node.children[closest_idx], point)
            # Update the parent clustering features
            node.entries[closest_idx].add_point(point)
            if node.is_full():
                self._split_node(node)

    def _find_closest_child(self, node, point):
        """Find the child whose CF is closest to the point."""
        min_distance = float("inf")
        closest_idx = -1
        for i, cf in enumerate(node.entries):
            centroid = [x / cf.n for x in cf.ls]
            distance = sum((centroid[j] - point[j]) ** 2 for j in range(len(point))) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        return closest_idx

    def _split_node(self, node):
        """Split a node into two and adjust the tree structure."""
        if node.is_leaf:
            # Split leaf node
            mid = len(node.entries) // 2
            new_node = CFNode(is_leaf=True, max_entries=self.branching_factor)
            new_node.entries = node.entries[mid:]
            node.entries = node.entries[:mid]
        else:
            # Split internal node
            mid = len(node.entries) // 2
            new_node = CFNode(is_leaf=False, max_entries=self.branching_factor)
            new_node.entries = node.entries[mid:]
            new_node.children = node.children[mid:]
            node.entries = node.entries[:mid]
            node.children = node.children[:mid]

        # Update parent
        if node == self.root:
            new_root = CFNode(is_leaf=False, max_entries=self.branching_factor)
            new_root.entries.append(ClusteringFeature())
            new_root.entries[-1].merge(node.entries[0])
            new_root.children = [node, new_node]
            self.root = new_root
        else:
            pass  # In real implementation, handle the parent adjustment recursively.

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

    # ClusteringFeature:
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


    # CFTree:
    tree = CFTree(threshold=4, branching_factor=4)
    tree.insert([1, 0])
    tree.insert([2, 0])

    tree.insert([10, 0])
    tree.insert([10, 0])

    tree.insert([40, 0])
    tree.insert([40, 0])

    tree.insert([100, 0])
    tree.insert([101, 0])

    tree.insert([111, 0])

    tree.insert([3, 0])
    

    # print(tree)
