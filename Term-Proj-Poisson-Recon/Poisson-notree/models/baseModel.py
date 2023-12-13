import numpy as np
import mcubes

class BaseReconstructor:
    def __init__(self, points, normals, grid_nums, padding):
        '''
        initialize the following variables
        points:         shaped (N, 3), points from the pointcloud
        normals:        shaped (N, 3), normal vectors of the points
        grid_nums:      input can be (3,) array or int
                        will be converted into shape (3,)
                        numbers of grids along each axis (after padding)
        padding:        input can be (3,) array or int
                        will be converted into shape (3,)
                        numbers of grids padded along each axis
        grid_size:      shaped (3,), length of a grid along each axis
        orig:           shaped (3,), left bottom front corner after padding
        '''
        self.points = points        # (N, 3)
        self.normals = normals      # (N, 3)

        self.grid_nums = None  
        if isinstance(grid_nums, int):
            self.grid_nums = np.array([grid_nums, grid_nums, grid_nums])
        else:
            self.grid_nums = np.array(grid_nums).astype(int)
        assert len(self.grid_nums) == 3

        self.padding = padding 
        if isinstance(padding, int):
            self.padding = np.array([padding, padding, padding])
        else:
            self.padding = np.array(padding).astype(int)
        assert len(self.padding) == 3
        
        bbox_size = np.max(self.points, axis=0) - np.min(self.points, axis=0)
        self.grid_size = bbox_size / self.grid_nums
        
        self.orig = np.min(self.points, axis=0) - self.padding * self.grid_size 

        self.grid_nums += 2 * self.padding

    def reconstruct(self):
        raise NotImplementedError 
    
    def save_obj(self, path, grids):
        vertices, triangles = mcubes.marching_cubes(grids, 0)
        vertices = vertices * self.grid_size + self.orig
        mcubes.export_obj(vertices, triangles, path)