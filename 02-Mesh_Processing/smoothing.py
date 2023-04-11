import openmesh as om 
import numpy as np 
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--sigma_s',            type=float, default=0.5)
parser.add_argument('--norm_update_iters',  type=int,   default=5)
parser.add_argument('--vert_update_iters',  type=int,   default=10)
args = parser.parse_args()

# Read in data

data_file = './data/smoothing.obj'
mesh = om.read_trimesh(data_file)

# Step 1 Calculate normal vectors, centers, and areas of faces

mesh.update_face_normals()

def calc_area(mesh, face) -> float:
    edges = mesh.fe(face)
    e1 = next(edges)
    e2 = next(edges)
    e1_vector = mesh.calc_edge_vector(e1)
    e2_vector = mesh.calc_edge_vector(e2)

    norm = np.cross(e1_vector, e2_vector)
    area = np.linalg.norm(norm) / 2
    return area 

def calc_cent(mesh, face)-> np.ndarray:
    return mesh.calc_face_centroid(face)

# Step 2 Calculate W_c and W_s

def calc_cent_dist(mesh, f1, f2):
    return np.linalg.norm(calc_cent(mesh, f1)-calc_cent(mesh, f2))

print('-------Calculating Sigma_C-------')
sigma_c = 0     # average distance of all adjacent facets
adjacent_num = 0
for face in tqdm(mesh.faces()):
    adjacent_faces = mesh.ff(face)
    adjacent_num += len(list(adjacent_faces))
    for a_face in adjacent_faces:
        sigma_c += calc_cent_dist(mesh, face, a_face)
sigma_s = args.sigma_s

def calc_Wc(mesh, f1, f2, sigma):
    cent_dist = calc_cent_dist(mesh, f1, f2)
    return np.exp(-0.5 * cent_dist ** 2 / (sigma ** 2))

def calc_Ws(mesh, f1, f2, sigma):
    norm1 = mesh.normal(f1)
    norm2 = mesh.normal(f2)
    norm_dist = np.linalg.norm(norm1 - norm2)
    return np.exp(-0.5 * norm_dist ** 2 / (sigma ** 2))

# Step 3 Update face normals

print('-------Updating Face Norms-------')
for norm_update_iter in range(args.norm_update_iters):
    new_norms_lst = []
    for face in tqdm(mesh.faces()):
        K = 0
        new_norm = np.zeros(3)
        for a_face in mesh.ff(face):
            area_a_face = calc_area(mesh, a_face)
            W_c = calc_Wc(mesh, face, a_face, sigma_c)
            W_s = calc_Ws(mesh, face, a_face, sigma_s)
            new_norm += area_a_face * W_c * W_s * mesh.normal(a_face)
            K += area_a_face * W_c * W_s
        new_norm = new_norm / K
        new_norms_lst.append(new_norm / np.linalg.norm(new_norm))
    for (i, face) in enumerate(mesh.faces()):
        mesh.set_normal(face, new_norms_lst[i])

# Step 4 Update Vertices

print('--------Updating Vertices--------')
for vert_update_iter in range(args.vert_update_iters):
    new_verts_lst = []
    for vert in tqdm(mesh.vertices()):
        new_vert = np.zeros(3)
        for a_face in mesh.vf(vert):
            new_vert += mesh.normal(a_face) * np.dot(mesh.normal(a_face), calc_cent(mesh, a_face) - mesh.point(vert))
        new_verts_lst.append(mesh.point(vert) + new_vert / len(list(mesh.vf(vert))))
    for (i, vert) in enumerate(mesh.vertices()):
        mesh.set_point(vert, new_verts_lst[i])

# Save the result

result_file = 'result.obj'
mesh = om.write_mesh(result_file, mesh)