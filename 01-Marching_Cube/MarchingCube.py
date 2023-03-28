import numpy as np
from models import marching_cubes
import os


def read_sdf(filename):
    return np.fromfile(filename, dtype=float)


def get_values(filename, val_shape):
    val = read_sdf(filename)
    return val.reshape(val_shape)


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
    input_rt = './data'
    output_rt = './result'
    if not os.path.exists(output_rt):
        os.mkdir(output_rt)
    input_filenames = ['01.sdf', '02.sdf']
    output_filenames = ['result_01.obj', 'result_02.obj']
    val_shape = (128, 128, 64)
    for i_name, o_name in zip(input_filenames, output_filenames):
        input_path = os.path.join(input_rt, i_name)
        output_path = os.path.join(output_rt, o_name)
        val = get_values(input_path, val_shape)
        verts, faces = marching_cubes(val)
        write_obj(output_path, verts, faces)
    

if __name__ == '__main__':
    main()