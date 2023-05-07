from model import Model
from utils import load_data, save_obj
import numpy as np
import torch
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    start_time = time.time()
    print('-------------------------Loading Data-------------------------')
    DATA_PATH = './gargoyle.xyz'
    X, _ = load_data(DATA_PATH)
    X = X.to(device)

    print('--------------------------Inferencing--------------------------')
    MODEL_PATH = './checkpoint.pth'
    model = Model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    save_obj(X[:, :3], model, path='result.obj')
    
    print(f'Time consumption: {time.time() - start_time}')
    