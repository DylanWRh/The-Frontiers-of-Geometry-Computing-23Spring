import numpy as np 
from tqdm import tqdm


class OctreeNode:
    def __init__(self, max_dep):
        self.is_leaf = True 
        self.nodes = None 
        self.parent = None 
        self.depth = 0
        self.max_dep = max_dep
        self.id = -1

        self.width = 0
        self.chi = 0.0
        self.center = np.zeros(3)
        self.normal = np.zeros(3)

        self.point_ids = []

    def point_in_node(self, point) -> bool:
        halfw = self.width / 2
        bb_min = self.center - halfw
        bb_max = self.center + halfw
        return np.logical_and(np.less_equal(bb_min, point), np.less_equal(point, bb_max)).all()

    def split(self):
        if self.is_leaf and self.depth < self.max_dep:
            self.is_leaf = False 
            self.nodes = [OctreeNode(self.max_dep) for i in range(8)]

            for i, node in enumerate(self.nodes):
                node.width = self.width / 2
                node.depth = self.depth + 1 
                node.parent = self 

                for j in range(3):
                    if ((i >> j) & 1):
                        node.center[j] = self.center[j] + self.width / 4
                    else:
                        node.center[j] = self.center[j] - self.width / 4

    def add_point(self, points, point_id):
        if self.is_leaf:
            self.point_ids.append(point_id)
            if self.depth < self.max_dep:
                self.split()
                for point_id in self.point_ids:
                    for node in self.nodes:
                        if node.point_in_node(points[point_id]):
                            node.add_point(points, point_id)
                            break 
                self.is_leaf = False
            if not self.is_leaf:
                self.point_ids = []
        else:
            for node in self.nodes:
                if node.point_in_node(points[point_id]):
                    node.add_point(points, point_id)
                    break
    
    def set_all_nodes(self, all_nodes):
        self.id = len(all_nodes[self.depth])
        all_nodes[self.depth].append(self)
        if not self.is_leaf:
            for node in self.nodes:
                all_nodes = node.set_all_nodes(all_nodes)
        return all_nodes
    
    def get_all_leaves(self, leaves):
        if self.is_leaf:
            leaves.append(self)
        else:
            for node in self.nodes:
                leaves = node.get_all_leaves(leaves)
        return leaves
    
    def find_leaf(self, point):
        if self.is_leaf:
            return self 
        for node in self.nodes:
            if node.point_in_node(point):
                return node.find_leaf(point)
        return None
    
    def find_depth_node(self, point, dep):
        if self.depth == dep:
            return self 
        if not self.is_leaf:
            for node in self.nodes:
                if node.point_in_node(point):
                    return node.find_depth_node(point, dep)
        return None 
    
    def find_and_split(self, point, dep):
        if self.depth == dep:
            return self 
        if self.is_leaf:
            self.split()
        for node in self.nodes:
            if node.point_in_node(point):
                return node.find_and_split(point, dep)
        return None 


class Octree:
    def __init__(self, pcd, max_dep=8):
        self.max_dep = max_dep
        self.node_size = 0
        self.head = OctreeNode(max_dep)

        self.pcd = pcd
        self.all_nodes = [[] for i in range(max_dep+1)]
        self.all_leaves = []
        self.point_to_leaf = []

    def build_tree(self):
        points = self.pcd.points 
        bb = self.pcd.get_bb()
        
        self.head.width = bb.width * 1.2
        self.head.center = bb.center 

        print('-----------------Building Octree-----------------')
        for i, pt in enumerate(tqdm(points)):
            self.head.add_point(points, i)
        self.set_all_nodes()
        self.all_leaves = self.head.get_all_leaves(self.all_leaves)

    def find_leaf(self, point):
        if self.head.point_in_node(point):
            return self.head.find_leaf(point)
        return None
    
    def find_depth_node(self, point, dep):
        if self.head.point_in_node(point):
            return self.head.find_depth_node(point, dep)
        return None 
    
    def find_and_split(self, point, dep):
        if self.head.point_in_node(point):
            return self.head.find_and_split(point, dep)
    
    def set_all_nodes(self):
        self.all_nodes = [[] for i in range(self.max_dep+1)]
        self.all_nodes = self.head.set_all_nodes(self.all_nodes)

    def set_all_leaves(self):
        self.all_leaves = self.head.get_all_leaves([])
