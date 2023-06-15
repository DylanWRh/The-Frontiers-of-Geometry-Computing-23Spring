from models.poisson import PoissonReconstructor
from utils import load_data
import argparse
import open3d
import numpy as np


def build_model(args):
    points, normals = load_data(args.in_path)
    grid_nums = [args.nx, args.ny, args.nz]
    model = PoissonReconstructor(points, normals, args.nx, args.pad)
    return model

def main(args):
    model = build_model(args) 
    model.save_obj(args.out_path, model.reconstruct())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='./data/gargoyle.xyz')
    parser.add_argument('--out_path', type=str, default='result.obj')
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--ny', type=int, default=128)
    parser.add_argument('--nz', type=int, default=128)
    parser.add_argument('--pad', type=int, default=8)
    args = parser.parse_args()
    main(args)