from .baseModel import BaseReconstructor
import scipy.sparse as sp
import scipy.sparse.linalg as la
import numpy as np 
import time

class PoissonReconstructor(BaseReconstructor):
    def __init__(self, points, normals, grid_nums, padding):
        super().__init__(points, normals, grid_nums, padding)
    
    def reconstruct(self):
        # see the report for detailed mathematical derivation
        G = self.build_grad_operator()
        Wx = self.get_interpolation_weights_axis(axis=0)
        Wy = self.get_interpolation_weights_axis(axis=1)
        Wz = self.get_interpolation_weights_axis(axis=2)
        W = self.get_interpolation_weights_axis(axis=None)

        vx = Wx.T @ self.normals[:, 0]
        vy = Wy.T @ self.normals[:, 1]
        vz = Wz.T @ self.normals[:, 2]
        v = np.concatenate([vx, vy, vz])

        print('-----start solving-----')
        start_time = time.time()
        g, _ = la.cg(G.T @ G, G.T @ v, maxiter=2000, tol=1e-5)
        end_time = time.time()
        print('-----end solving-----')
        print(f'Time consumption: {end_time-start_time}')

        sigma = np.mean(W @ g)
        g -= sigma 
        g_grid = g.reshape(self.grid_nums)
        return g_grid 
    
    def build_grad_operator(self):
        ''' 
        build grad operator G
        (G.T) is nabla operator
        (G.T @ G) is Laplacian operator

        G is represented with sparce matrix
        shaped ((nx-1)*ny*nz+nx*(ny-1)*nz+nx*ny*(nz-1), nx*ny*nz)
        where ni is the number of grids along i-axis after padding
        '''
        hx, hy, hz = self.grid_size
        grad_x = self.build_grad_operator_axis(hx, axis=0)
        grad_y = self.build_grad_operator_axis(hy, axis=1)
        grad_z = self.build_grad_operator_axis(hz, axis=2)
        return sp.vstack([grad_x, grad_y, grad_z])
    
    def build_grad_operator_axis(self, h, axis=0):
        grid_idx = np.arange(np.prod(self.grid_nums)).reshape(self.grid_nums)
        nx, ny, nz = self.grid_nums
        assert axis in [0, 1, 2]
        if axis == 0:
            grad_grid_num = (nx-1) * ny * nz
            col_idx = np.concatenate((grid_idx[1:, :, :].flatten(), grid_idx[:-1, :, :].flatten()))
        elif axis == 1:
            grad_grid_num = nx * (ny-1) * nz
            col_idx = np.concatenate((grid_idx[:, 1:, :].flatten(), grid_idx[:, :-1, :].flatten()))
        elif axis == 2:
            grad_grid_num = nx * ny * (nz-1)
            col_idx = np.concatenate((grid_idx[:, :, 1:].flatten(), grid_idx[:, :, :-1].flatten()))
        
        row_idx = np.arange(grad_grid_num)
        row_idx = np.tile(row_idx, 2)

        data = [1/h] * grad_grid_num + [-1/h] * grad_grid_num
        return sp.csr_matrix((data, (row_idx, col_idx)), shape=(grad_grid_num, nx*ny*nz))

    def get_interpolation_weights_axis(self, axis=None):
        '''
        weights for linear interpolation to transform feature values (eg.normals)
        of cloud points to grid points

        return W: shaped (N, num_grids), where num_grids rely on given axis
        eg. if axis = 0, num_grids = (nx-1)*ny*nz in order to match shape of G
        axis = None means a global weight and num_grids is thus nx*ny*nz 
        '''
        assert (axis is None) or (axis in [0, 1, 2])
        orig = self.orig.copy()
        if axis in [0, 1, 2]:
            orig[axis] += 0.5 * self.grid_size[axis]
        
        rel_coords = (self.points - orig) / self.grid_size
        lower_indices = np.floor(rel_coords).astype(int)
        t = rel_coords - lower_indices  # (N, 3)

        nx, ny, nz = self.grid_nums

        if axis == 0:
            grad_grid_num = (nx-1) * ny * nz
            grad_grid_idx = np.arange(grad_grid_num).reshape((nx-1, ny, nz))
        elif axis == 1:
            grad_grid_num = nx * (ny-1) * nz
            grad_grid_idx = np.arange(grad_grid_num).reshape((nx, ny-1, nz))
        elif axis == 2:
            grad_grid_num = nx * ny * (nz-1)
            grad_grid_idx = np.arange(grad_grid_num).reshape((nx, ny, nz-1))
        else:
            grad_grid_num = nx * ny * nz
            grad_grid_idx = np.arange(grad_grid_num).reshape((nx, ny, nz))
        
        data = []
        row_idx = []
        col_idx = []
        
        N = self.points.shape[0]
        for dz in [0, 1]:
            for dy in [0, 1]:
                for dx in [0, 1]:
                    weight = np.prod([(1-t[:, i]) if d==0 else t[:, i] for i, d in enumerate([dx, dy, dz])], axis=0)
                    data.append(weight)
                    row_idx.append(np.arange(N))
                    col_idx.append(grad_grid_idx[lower_indices[:, 0]+dx, lower_indices[:, 1]+dy, lower_indices[:, 2]+dz])
        
        data = np.concatenate(data)
        row_idx = np.concatenate(row_idx)
        col_idx = np.concatenate(col_idx)

        W = sp.csr_matrix((data, (row_idx, col_idx)), shape=(N, grad_grid_num))
        return W