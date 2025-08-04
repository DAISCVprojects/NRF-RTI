import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader  as dataloader
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(52)
torch.cuda.manual_seed(seed=52)
np.random.seed(seed =52)

class sineActivation(nn.Module):
    def __init__(self):
        super(sineActivation, self).__init__()
        return
    
    def forward(self, x):
        return torch.sin(x)
    
def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)