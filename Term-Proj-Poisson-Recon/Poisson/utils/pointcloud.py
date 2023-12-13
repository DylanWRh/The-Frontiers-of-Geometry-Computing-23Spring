import numpy as np 


class PointCloud:
    def __init__(self, points, normals):
        self.points = points 
        self.normals = normals 

    class BoundingBox:
        def __init__(self, points):
            self.min_coord = np.min(points, axis=0)
            self.max_coord = np.max(points, axis=0)
            self.width = np.max(self.max_coord - self.min_coord)
            self.center = (self.min_coord + self.max_coord) / 2

    def get_bb(self):
        return self.BoundingBox(self.points)