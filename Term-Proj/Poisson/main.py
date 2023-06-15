import numpy as np 
from utils.octree import Octree
from utils.data_util import build_pcd, save_obj
import argparse
from tqdm import tqdm 
import scipy.sparse as sp
import scipy.sparse.linalg as la
from skimage.measure import marching_cubes
import time


def base_func_along_axis(x):
    if x <= -1.5 or x >= 1.5:
        return 0
    if x <= -0.5:
        return 0.5 * x * x + 1.5 * x + 1.125
    if x >= 0.5:
        return 0.5 * x * x - 1.5 * x + 1.125
    return 0.75 - x * x

def D_base_func_along_axis(x):
    if x <= -1.5 or x >= 1.5:
        return 0
    if x <= -0.5:
        return x + 1.5
    if x >= 0.5:
        return x - 1.5 
    return -2 * x

def DD_base_func_along_axis(x):
    if x <= -1.5 or x >= 1.5:
        return 0
    if x <= -0.5:
        return 1
    if x >= 0.5:
        return 1
    return -2

def base_func(q):
    return np.array([base_func_along_axis(x) for x in q]).prod()

def D_base_func(q):
    fx, fy, fz = base_func_along_axis(q[0]), base_func_along_axis(q[1]), base_func_along_axis(q[2])
    Dfx, Dfy, Dfz = D_base_func_along_axis(q[0]), D_base_func_along_axis(q[1]), D_base_func_along_axis(q[2])
    return np.array([fy*fz*Dfx, fz*fx*Dfy, fx*fy*Dfz])

def DD_base_func(q):
    fx, fy, fz = base_func_along_axis(q[0]), base_func_along_axis(q[1]), base_func_along_axis(q[2])
    DDfx, DDfy, DDfz = DD_base_func_along_axis(q[0]), DD_base_func_along_axis(q[1]), DD_base_func_along_axis(q[2])
    return np.array([fy*fz*DDfx, fz*fx*DDfy, fx*fy*DDfz])

def Poisson(pcd, max_depth=7):
    octree = Octree(pcd, max_depth)
    octree.build_tree()

    points, normals = pcd.points, pcd.normals

    # preprocessing, create neighbours for each leaf with point
    d_nodes = octree.all_nodes[-1]
    d_points = np.zeros((len(d_nodes), 3))
    d_normals = np.zeros((len(d_nodes), 3))

    print('-----------------Modifying Octree-----------------')
    for i, d_node in enumerate(tqdm(d_nodes)):
        pids = d_node.point_ids
        if len(pids):
            d_points[i] = np.mean(points[pids], axis=0)
            d_normals[i] = np.mean(normals[pids], axis=0)

    width = d_nodes[0].width
    halfw = d_nodes[0].width / 2
    depth = d_nodes[0].depth
    for d_point in tqdm(d_points):
        if np.linalg.norm(d_point) < 1e-12:
            continue
        for i in range(8):
            tmp_p = d_point[:]
            for j in range(3):
                if ((i >> j) & 1):
                    tmp_p[j] += halfw
                else:
                    tmp_p[j] -= halfw
        loc_node = octree.find_and_split(tmp_p, depth)
        if loc_node is not None:
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    for dz in range(-1, 2):
                        neighbour_node = loc_node.center + width * np.array([dx, dy, dz])
                        octree.find_and_split(neighbour_node, depth)
    
    octree.set_all_nodes()
    octree.set_all_leaves()

    print('-----------------Building Equation-----------------')
    for dep, d_nodes in enumerate(octree.all_nodes[1:]):
        print(f'depth: {dep+1}, nodes num: {len(d_nodes)}')
    
    d_nodes = octree.all_nodes[-1]
    d_points = np.zeros((len(d_nodes), 3))
    d_normals = np.zeros((len(d_nodes), 3))
    for i, d_node in enumerate(d_nodes):
        pids = d_node.point_ids
        if len(pids):
            d_points[i] = np.mean(points[pids], axis=0)
            d_normals[i] = np.mean(normals[pids], axis=0)

    # Find neighbour leaf nodes for each leaf node
    # js = []
    # d_node_width = d_nodes[0].width
    # d_node_center = np.array([node_i.center for node_i in d_nodes])
    # thresh = 1.5 * d_node_width
    # for i, node_i in enumerate(tqdm(d_nodes)):
    #     delta = np.abs(d_node_center - node_i.center)
    #     js_i = np.where(
    #        (delta[:, :, 0] < thresh) & (delta[:, :, 1] < thresh) & (delta[:, :, 2] < thresh)
    #     )[0]
    #     js.append(js_i)

    # Find neighbour leaf nodes for each leaf node
    # js = []
    # batch_size = 128
    # d_node_width = d_nodes[0].width
    # d_node_center = np.array([node_i.center for node_i in d_nodes])
    # thresh = 1.5 * d_node_width
    # for i in tqdm(range((len(d_nodes)+batch_size-1) // batch_size)):
    #     node_i_center = np.array([d_node.center for d_node in d_nodes[i*batch_size: (i+1)*batch_size]])
    #     delta = np.abs(node_i_center[:, None, :] - d_node_center[None, :, :])
    #     i_s, js_i, *_ = np.where(
    #         (delta[:, :, 0] < thresh) & (delta[:, :, 1] < thresh) & (delta[:, :, 2] < thresh)
    #     )
    #     for k in range(len(set(i_s))):
    #         js.append(js_i[i_s == k])

    # Find neighbour leaf nodes for each leaf node
    js = []
    d_node_width = d_nodes[0].width
    id2idx = {d_nodes[idx].id: idx for idx in range(len(d_nodes))}
    for i, node_i in enumerate(tqdm(d_nodes)):
        js_i = []
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    neighbour_center = node_i.center + d_node_width * np.array([dx, dy, dz])
                    neighbour_node = octree.find_depth_node(neighbour_center, node_i.depth)
                    if neighbour_node is not None:
                        js_i.append(id2idx[neighbour_node.id])
        js.append(js_i)

    # Compute discrete base function matrix for projection
    base_func_mat = []
    for i, node_i in enumerate(tqdm(d_nodes)):
        for j in js[i]:
            node_j = d_nodes[j]
            fval = base_func((node_i.center - node_j.center) / node_j.width) / (node_j.width ** 3)
            base_func_mat.append((j, i, fval))
    row, col, val = zip(*base_func_mat)
    base_func_mat = sp.csc_matrix((val, (row, col)), shape=(len(d_nodes), len(d_nodes)))
    
    # Compute weighted normal
    width = d_nodes[0].width
    halfw = d_nodes[0].width / 2
    depth = d_nodes[0].depth
    weighted_normal = np.zeros((len(d_nodes), 3))
    for i, d_point in enumerate(d_points):
        if np.linalg.norm(d_point) < 1e-12:
            continue 
        neighbour_nodes = []   
        for j in range(8):
            tmp_p = d_point[:]
            for k in range(3):
                if ((j >> k) & 1):
                    tmp_p[k] += halfw
                else:
                    tmp_p[k] -= halfw
            loc_node = octree.find_depth_node(tmp_p, depth)
            if loc_node is not None:
                neighbour_nodes.append(loc_node)
        for nid, neighbour_node in enumerate(neighbour_nodes):
            # For simplification, use normal ratio instead of trilinear interpolation weight
            weight = 1  - np.linalg.norm(neighbour_node.center - d_point) / (np.sqrt(3) * octree.head.width)
            weighted_normal[neighbour_node.id] -= weight * d_normals[i]
    
    # Compute grad operator
    grad_x = []
    grad_y = []
    grad_z = []
    for i, node_i in enumerate(tqdm(d_nodes)):
        for j in js[i]:
            node_j = d_nodes[j]
            Dval = D_base_func((node_i.center - node_j.center) / node_j.width) / (node_j.width ** 4)
            grad_x.append((i, j, Dval[0]))
            grad_y.append((i, j, Dval[1]))
            grad_z.append((i, j, Dval[2]))
    if len(grad_x):
        Xrow, Xcol, Xval = zip(*grad_x)
        Dx = sp.csc_matrix((Xval, (Xrow, Xcol)), shape=(len(d_nodes), len(d_nodes)))
    else:
        Dx = sp.csc_matrix(len(d_nodes), len(d_nodes))
    if len(grad_y):
        Yrow, Ycol, Yval = zip(*grad_y)
        Dy = sp.csc_matrix((Yval, (Yrow, Ycol)), shape=(len(d_nodes), len(d_nodes)))
    else:
        Dy = sp.csc_matrix(len(d_nodes), len(d_nodes))
    if len(grad_z):
        Zrow, Zcol, Zval = zip(*grad_z)
        Dz = sp.csc_matrix((Zval, (Zrow, Zcol)), shape=(len(d_nodes), len(d_nodes)))
    else:
        Dz = sp.csc_matrix(len(d_nodes), len(d_nodes))

    # Compute div V
    div_v = np.zeros((len(d_nodes), 3))
    div_v[:, 0] = Dx @ weighted_normal[:, 0]
    div_v[:, 1] = Dy @ weighted_normal[:, 1]
    div_v[:, 2] = Dz @ weighted_normal[:, 2]
    proj_div_v = (base_func_mat @ div_v).sum(axis=1)

    # Compute laplacian operator
    lap_x = []
    lap_y = []
    lap_z = []
    for i, node_i in enumerate(tqdm(d_nodes)):
        for j in js[i]:
            node_j = d_nodes[j]
            DDval = DD_base_func((node_i.center - node_j.center) / node_j.width) / (node_j.width ** 5)
            lap_x.append((i, j, DDval[0]))
            lap_y.append((i, j, DDval[1]))
            lap_z.append((i, j, DDval[2]))
    if len(lap_x):
        Xrow, Xcol, Xval = zip(*lap_x)
        DDx = sp.csc_matrix((Xval, (Xrow, Xcol)), shape=(len(d_nodes), len(d_nodes)))
    else:
        DDx = sp.csc_matrix(len(d_nodes), len(d_nodes))
    if len(lap_y):
        Yrow, Ycol, Yval = zip(*lap_y)
        DDy = sp.csc_matrix((Yval, (Yrow, Ycol)), shape=(len(d_nodes), len(d_nodes)))
    else:
        DDy = sp.csc_matrix(len(d_nodes), len(d_nodes))
    if len(lap_z):
        Zrow, Zcol, Zval = zip(*lap_z)
        DDz = sp.csc_matrix((Zval, (Zrow, Zcol)), shape=(len(d_nodes), len(d_nodes)))
    else:
        DDz = sp.csc_matrix(len(d_nodes), len(d_nodes))
    
    L = (base_func_mat @ (DDx + DDy + DDz))
    L.prune()

    print('-----------------Solving Equation-----------------')
    start_time = time.time()
    x, _ = la.cg(L, proj_div_v, maxiter=2000, tol=1e-5)
    for i in range(len(d_nodes)):
        d_nodes[i].chi = x[i]
    end_time = time.time()
    print(f'Time Consumption: {end_time - start_time}')
    
        
    print('-----------------Extracting Isosurface-----------------')
    # Compute iso value (a very simplified version)
    all_leaves = octree.all_nodes[-1]
    leaf_chis = np.array([leaf.chi for leaf in all_leaves if len(leaf.point_ids)])
    
    iso_value = leaf_chis.mean()

    resolution = 2 ** max_depth + 1

    head_center = octree.head.center 
    head_width = octree.head.width
    head_halfw = head_width / 2
    coords_min = head_center - head_halfw
    leaf_width = all_leaves[0].width
    
    grids = np.zeros((resolution, resolution, resolution))
    for leaf in tqdm(all_leaves):
        nx, ny, nz = ((leaf.center - coords_min) // leaf_width).astype(int)
        # grids[nx, ny, nz] = leaf.chi
        for dx in range(max(nx-1, 0), min(nx+3, resolution)):
            for dy in range(max(ny-1, 0), min(ny+3, resolution)):
                for dz in range(max(nz-1, 0), min(nz+3, resolution)):
                    q = np.array([dx, dy, dz]) * leaf_width + coords_min
                    fval = base_func((q - leaf.center) / leaf_width) / leaf_width ** 3
                    grids[dx, dy, dz] += fval * leaf.chi
    
    print('-----------------Generating Mesh-----------------')
    start_time = time.time()
    verts, faces, _, _ = marching_cubes(
        volume=grids, level=iso_value,
        spacing=(leaf_width, leaf_width, leaf_width)
    )
    verts = verts + coords_min
    end_time = time.time()
    print(f'Time Consumption: {end_time - start_time}')
    save_obj(f'res_{resolution-1}.obj', verts, faces)
    print(f'File saved as res_{resolution-1}.obj')

    # EPS = iso_value / 5
    # with open('point_res.obj', 'w') as f:
    #     for x in octree.all_nodes[-1]:
    #         if iso_value - EPS <= x.chi <= iso_value + EPS:
    #             f.write(f'v {x.center[0]} {x.center[1]} {x.center[2]}\n')

def main():
    path = './data/gargoyle.xyz'
    pcd = build_pcd(path)
    Poisson(pcd, max_depth=7)

if __name__ == '__main__':
    main()