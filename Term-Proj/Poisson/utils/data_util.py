import numpy as np 
from .pointcloud import PointCloud


def read_xyz(path):
    X = None 
    with open(path, 'r') as f:
        lines = f.readlines()
        X = np.array([[float(i) for i in line.split()] for line in lines])
    points = X[:, :3]
    normals = X[:, 3:]
    return points, normals

def build_pcd(path):
    points, normals = read_xyz(path)
    return PointCloud(points, normals)

def save_obj(path, verts, faces):
    with open(path, 'w') as f:
        for vert in verts:
            f.write("v {0} {1} {2}\n".format(vert[0],vert[1],vert[2]))
        
        if np.min(faces) == 0:
            faces += 1
        for face in faces:
            f.write('f {0} {1} {2}\n'.format(face[0], face[2], face[1]))
