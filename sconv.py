import torch
import torch.nn as nn
import torch.nn.functional as F

def binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return torch.sign(tensor)
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)

def quantize(x, k):
    n = float(2 ** k - 1)
    return torch.round(x * n) / n

def quantize10(x, k):
    n = float(10 ** k)
    return torch.round(x * n) / n

def scale_conv(
    in_channels, out_channels, kernel_size, stride=1, bias=True):    
    return SConv()

def gen_sconv(target):
    return (m for m in target.modules() if isinstance(m, SConv))

class SConv(nn.Module):
    n_bits = 16
    def __init__(self):
        super(SConv, self).__init__()
        self.k = -1
        self.channels = ('-', '-')

    def __repr__(self):
        s = '{}-{}-{}: {} -> {}'.format(
            self.__class__.__name__, self.k, SConv.n_bits, *self.channels
        )
        return s

    def set_params(
        self, source,
        kernels=None, scales=None, input_bits=32, scale_bits=16):

        self.sign = torch.sign(kernels).cuda()
        self.kh, self.kw = source.kernel_size
        self.channels = (source.in_channels, source.out_channels)
        self.kwargs = {'stride': source.stride, 'padding': source.padding}
        self.input_bits = input_bits
        self.scale_bits = scale_bits
        self.register_parameter('scales', scales)

        if hasattr(source, 'bias'):
            self.register_parameter('bias', source.bias)        

    def get_parameters(self):
        m_weight = self.sign
        weight = m_weight.view(*self.channels[::-1], self.kh, self.kw)
        scales = self.get_scales()
        weight = weight.cuda() * scales.cuda()
        bias = getattr(self, 'bias', None)

        return weight, bias

    def get_scales(self):
        if self.scale_bits == 16:
            scales = self.scales.half().float()
        else:
            scales = quantize10(self.scales.data, self.scale_bits).float()
        return scales

    def forward(self, x):
        if x.size(1) != 3:
            if self.input_bits == 16:
                x.data = x.data.half().float()
            elif self.input_bits == 32:
                pass
            else:
                x.data = quantize(x.data, self.input_bits)
            
        weight, bias = self.get_parameters()
        x = F.conv2d(x, weight, bias, **self.kwargs)

        return x
    