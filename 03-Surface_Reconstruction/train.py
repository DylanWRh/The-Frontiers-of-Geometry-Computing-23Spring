import torch 
import numpy as np
import torch.nn as nn
import time
from model import Model
from tqdm import tqdm
from utils import load_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calc_grad(X, logits):
    X_grad = torch.autograd.grad(
        outputs=logits,
        inputs=X,
        grad_outputs=torch.ones_like(logits, requires_grad=False, device=logits.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, :]
    return X_grad


def get_sample(X, local_sigma, global_sigma):
    data_len = X.shape[0]
    sample_local = X + torch.randn_like(X) * local_sigma[:, None]
    sample_global = (torch.rand(data_len // 8, 3, device=X.device) * 2 - 1) * global_sigma
    sample = torch.cat([sample_local, sample_global], dim=0)
    return sample


def train(model, X, local_sigma, epoches=20000, lr=0.005, batch_size=16384, eval_every=2000):
    X = X.to(device)
    X.requires_grad_()
    local_sigma = local_sigma.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    global_sigma = 1.8
    lam = 0.1
    tau = 1.0
    
    best_loss = float('inf')
    best_model = None

    for epoch in tqdm(range(epoches)):
        model.train()
        
        batch = torch.tensor(np.random.choice(X.shape[0], batch_size, False))
    
        points = X[batch, :3]
        normals = X[batch, 3:]
        sigmas = local_sigma[batch]
        
        sample_points = get_sample(points, sigmas, global_sigma)
        
        points_pred = model(points)
        sample_pred = model(sample_points)
        
        points_grad = calc_grad(points, points_pred)
        sample_grad = calc_grad(sample_points, sample_pred)

        pred_loss = points_pred.abs().mean()
        norm_loss = (points_grad - normals).norm(2, dim=1).mean()
        grad_loss = ((sample_grad.norm(2, dim=-1) - 1) ** 2).mean()
        loss = pred_loss + lam * grad_loss + tau * norm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0:
            print(f'Epoch {epoch+1}, loss = {loss.item()}, pred_loss = {pred_loss.item()}, grad_loss = {grad_loss.item()}, norm_loss = {0}')
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model = model
                SAVE_PATH = './checkpoint1.pth'
                torch.save(best_model.state_dict(), SAVE_PATH)


def main():
    start_time = time.time()
    
    print('-------------------------Loading Data-------------------------')
    DATA_PATH = './gargoyle.xyz'
    X, local_sigma = load_data(DATA_PATH)

    print('---------------------------Training---------------------------')
    model = Model().to(device)
    train(model, X, local_sigma)
    
    print(f'Time consumption: {time.time() - start_time}')


if __name__ == '__main__':
    main()
