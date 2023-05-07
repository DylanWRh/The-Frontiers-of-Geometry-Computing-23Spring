import torch
import torch.nn as nn
import numpy as np


class Model(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=512, hidden_layers=8, skip_layer=[4], beta=100):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_layers = hidden_layers
        self.skip_layer = skip_layer
        self.in_channels = in_channels
        self.out_channels = 1
        
        self.fc_in = nn.Linear(in_channels, hidden_channels)
        self.fc_out = nn.Linear(hidden_channels, self.out_channels)
        
        for layer in range(hidden_layers-1):
            if layer+1 in skip_layer:
                hidden_out_channels = hidden_channels - in_channels
            else:
                hidden_out_channels = hidden_channels
            
            lin = nn.Linear(hidden_channels, hidden_out_channels)
            
            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(hidden_out_channels))
            setattr(self, 'fc_hid'+str(layer), lin)
        
        self.activation = nn.Softplus(beta=beta)

    def forward(self, points):
        X = points
        X = self.fc_in(X)
        X = self.activation(X)
        for layer in range(self.hidden_layers-1):
            lin = getattr(self, 'fc_hid'+str(layer))
            if layer in self.skip_layer:
                X = torch.cat([X, points], -1) / np.sqrt(2)
            X = lin(X)
            X = self.activation(X)
        X = self.fc_out(X)
        return X