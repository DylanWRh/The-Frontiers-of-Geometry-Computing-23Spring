from tables import edge_table, tri_table
import numpy as np


def marching_cubes(val, threshold=0):
    '''
    input: 
        val: ndarray shaped (X, Y, Z), grid values
    outputs:
        verts: ndarray shaped (N_verts, 3), coordinates of verts
        faces: ndarray shaped (N_faces, 3), index of verts
    '''
    verts = []
    faces = []
    X, Y, Z = val.shape
    for x in range(X-1):
        if x % (X // 10) == 0:
            print(f'Done {x} / {X}')
        for y in range(Y-1):
            for z in range(Z-1):
                grid_val = [val[x][y][z],       val[x][y+1][z], 
                            val[x+1][y+1][z],   val[x+1][y][z],
                            val[x][y][z+1],     val[x][y+1][z+1],
                            val[x+1][y+1][z+1], val[x+1][y][z+1]]
                grid_vert = np.array(
                    [[x, y, z],       [x, y+1, z], 
                     [x+1, y+1, z],   [x+1, y, z],
                     [x, y, z+1],     [x, y+1, z+1],
                     [x+1, y+1, z+1], [x+1, y, z+1]]
                )
                inter_edges = intersected_edges(grid_val, threshold)
                inter_verts = intersected_verts(edge_table[inter_edges], grid_vert, grid_val, threshold)
                tri_list = tri_table[inter_edges]
                inter_faces = intersected_faces(inter_verts, tri_list)

                local_to_global_index = [None for i in range(12)]
                for i in range(len(inter_verts)):
                    if inter_verts[i][0] is not None:
                        verts.append(inter_verts[i])
                        local_to_global_index[i] = len(verts)-1
                for face in inter_faces:
                    faces.append([local_to_global_index[i] for i in face])
    print(len(verts))
    print(len(faces))
    verts = np.array(verts, dtype=float)
    faces = np.array(faces, dtype=int)
    return verts, faces


def intersected_edges(grid, threshold=0):
    '''
    input:
        grid: list of 8 values
    output:
        index of edge_table (a number of 12 bits), which helps determine the edges intersected with the mesh
    '''
    edge_idx = 0
    for i in range(8):
        if grid[i] < threshold:
            edge_idx |= (2**i)
    return edge_idx


def intersected_verts(inter_edges, grid_vert, grid_val, threshold=0):
    '''
    input:
        grid_vert: ndarray shaped (8, 3)
        grid_val: a list of 8 float
    output:
        vert_list: a list of 12-element [x, y, z] coordinate
    '''
    vert_list = [[None, None, None] for i in range(12)]
    grid_vert_index = [[0, 1], [1, 2], [2, 3], [3, 0],
                       [4, 5], [5, 6], [6, 7], [7, 4],
                       [0, 4], [1, 5], [2, 6], [3, 7]]
    for i in range(12):
        if (inter_edges & (2**i)):
            p_1 = grid_vert[grid_vert_index[i][0]]
            p_2 = grid_vert[grid_vert_index[i][1]]
            val_1 = grid_val[grid_vert_index[i][0]]
            val_2 = grid_val[grid_vert_index[i][1]]
            vert_list[i] = calculate_intersection(p_1, p_2, val_1, val_2, threshold)
    return vert_list


def calculate_intersection(p_1, p_2, val_1, val_2, threshold=0):
    '''
    input:
        p_1 and p_2: coordinates of 2 point, ndarray shaped (3, )
        val_1 and val_2: values of p_1 and p_2, float
    output:
        coordinates of intersection point, ndarray shaped (3, )
    '''
    eps = 1e-6
    # special case
    if np.abs(val_1-threshold) < eps:
        return p_1
    if np.abs(val_2-threshold) < eps:
        return p_2
    if np.abs(val_1 - val_2) < eps:
        return (p_1+p_2)/2
    # linear interpolation
    return p_1 + (threshold - val_1) / (val_2 - val_1) * (p_2 - p_1)


def intersected_faces(inter_verts, tri_list):
    ''' 
    input:
        inter_verts: a list of 12-element [x, y, z] coordinate
        tri_list: a list of 16-element integer
    output:
        inter_faces: a list of [i, j, k] indicating intersected faces
    '''
    inter_faces = []
    i = 0
    while tri_list[i] != -1:
        inter_faces.append(tri_list[i: i+3])
        i += 3
    return inter_faces

