import numpy as np

def load_data(path):
    X = None
    with open(path, 'r') as f:
        lines = f.readlines()
        X = np.array([[float(i) for i in line.split()] for line in lines])
    points = X[:, :3]
    normals = -X[:, 3:]
    return points, normals