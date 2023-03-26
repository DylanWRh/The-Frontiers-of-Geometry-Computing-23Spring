import numpy as np
from models import marching_cubes


def read_sdf(filename):
    return np.fromfile(filename, dtype=float)


def get_values():
    val_1 = read_sdf('data\\01.sdf')
    val_2 = read_sdf('data\\02.sdf')
    return np.concatenate([val_1, val_2]).reshape((128, 128, 128))


def write_obj(filename, verts, faces):
    '''
    verts: ndarray shaped (N_verts, 3)
    faces: ndarray shaped (N_faces, 3)
    filename: "xxx.obj"
    ''' 
    with open(filename, 'w') as f:
        for vert in verts:
            f.write("v {0} {1} {2}\n".format(vert[0],vert[1],vert[2]))
        
        if np.min(faces) == 0:
            faces += 1
        for face in faces:
            f.write('f {0} {1} {2}\n'.format(face[0], face[1], face[2]))


def main():
    val = get_values()
    verts, faces = marching_cubes(val)
    output_filename = 'result.obj'
    write_obj(output_filename, verts, faces)

if __name__ == '__main__':
    main()