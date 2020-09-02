import math
import numpy as np
import torch
import torch.nn as nn

act_dict = {
    'relu': lambda : nn.ReLU(inplace=True),
    'tanh': lambda : nn.Tanh(),
    'Htanh': lambda : nn.Hardtanh(inplace=True),
}

def default_conv(
        in_channels, out_channels, kernel_size, stride=1, bias=True):

    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2) if kernel_size.__class__ == int else tuple((np.array(kernel_size) // 2).tolist()), 
        stride=stride, bias=bias
    )

def default_linear(in_channels, out_channels, bias=True):
    return nn.Linear(in_channels, out_channels, bias=bias)

def default_norm(in_channels):
    return nn.BatchNorm2d(in_channels)

def default_act():
    return nn.ReLU(True)