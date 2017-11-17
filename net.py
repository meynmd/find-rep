import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as fn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

