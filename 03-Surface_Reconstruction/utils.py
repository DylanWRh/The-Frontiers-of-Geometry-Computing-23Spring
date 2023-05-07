import torch
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(path, k=50):
    X = None
    with open(path, 'r') as f:
        lines = f.readlines()
        X = np.array([[float(i) for i in line.split()] for line in lines])
    X = torch.tensor(X).float()
    
    sigmas = []
    ptree = cKDTree(X)
    for p in np.array_split(X, 100, axis=0):
        sigmas.append(ptree.query(p, k+1)[0][:, -1])
    local_sigma = np.concatenate(sigmas)
    local_sigma = torch.from_numpy(local_sigma).float()
    
    return X, local_sigma


def get_grid(points, resolution):
    eps = 1e-1
    coord_min = np.min(points, axis=0)
    coord_max = np.max(points, axis=0)
    bounding_box = coord_max - coord_min
    x = np.linspace(coord_min[0]-eps, coord_max[0]+eps, resolution)
    y = np.linspace(coord_min[1]-eps, coord_max[1]+eps, resolution)
    z = np.linspace(coord_min[2]-eps, coord_max[2]+eps, resolution)
    xx, yy, zz = np.meshgrid(x, y, z)
    grid = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T).float().to(device)
    return grid, x, y, z


def save_obj(X, model, resolution=512, path='result.obj', threshold=0):
    pred = model(X).detach().cpu().numpy()
    points = X.detach().cpu().numpy()
    grid, x, y, z = get_grid(points, resolution)
    grid_val = []
    for i, ps in enumerate(torch.split(grid, 100000, dim=0)):
        grid_val.append(model(ps).detach().cpu().numpy())
    grid_val = np.concatenate(grid_val, axis=0)
    verts, faces, _, _ = marching_cubes(
        volume=grid_val.reshape(resolution, resolution, resolution),
        level=threshold,
        spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0])
    )
    verts = verts + np.array([x[0], y[0], z[0]])
    with open(path, 'w') as f:
        for vert in verts:
            f.write("v {0} {1} {2}\n".format(vert[0],vert[1],vert[2]))
        
        if np.min(faces) == 0:
            faces += 1
        for face in faces:
            f.write('f {0} {1} {2}\n'.format(face[0], face[1], face[2]))

    