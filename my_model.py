import torch 
from torch import nn 
import torch.nn.functional as F


class SimpleNet(nn.Module): 
    def __init__(self, n_features, n_hidden, n_out): 
        super().__init__()
        self.layer1 = nn.Linear(n_features, n_hidden, dtype=torch.float64) 
        self.layer2 = nn.Linear(n_hidden, n_out, dtype=torch.float64) 
        self.relu = nn.ReLU() 

    def forward(self, x): 
        x = self.relu(self.layer1(x)) 
        x = torch.sigmoid(self.layer2(x)) 
        return x 

