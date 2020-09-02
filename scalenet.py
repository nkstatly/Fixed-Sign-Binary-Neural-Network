import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import common
import sconv
import pretrainedmodels

from resnet import ResNet
from resnet_imagenet import ResNet_ImageNet

def make_model(args):
    conv1x1 = common.default_conv
    conv3x3 = sconv.scale_conv
    if args.dataset == 'cifar10':
        BaseNet = ResNet
    elif args.dataset == 'imagenet':
        BaseNet = ResNet_ImageNet
    
    class ScaleNet(BaseNet):
        def __init__(self, args):
            super(ScaleNet, self).__init__(args=args, conv3x3=conv3x3, conv1x1=conv1x1)
            
            if args.dataset == 'cifar10':
                parent = ResNet(args)
#                 wd = torch.load('./results/' + dir_of_full precision + '/model_best.pth.tar') # use the sign of full precision model
#                 parent.load_state_dict(wd['state_dict'])
            elif args.dataset == 'imagenet':
                parent = ResNet_ImageNet(args)
                model_name = args.model + str(args.depth)
                model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
                tmp = parent.state_dict()
                tmp1 = model.state_dict()
                for i,j in enumerate(tmp):
                    tmp[j] = tmp1[list(tmp1.keys())[i]]
                parent.load_state_dict(tmp)
                
            bnn_dict, s_dict = self.extract(parent)
            torch.save(bnn_dict, './bnn_dict.pth') # save the sign
            self.quantize_kernels(parent, bnn_dict, s_dict,
                                  args.input_bits, args.scale_bits)

        def gen(self, target, conv1x1=False):
            def _criterion(m):
                if isinstance(m, nn.Conv2d):
                    if conv1x1:
                        return m.kernel_size[0] * m.kernel_size[1] == 1
                    else:
                        return m.kernel_size[0] * m.kernel_size[1] > 1
                elif isinstance(m, nn.ConvTranspose2d):
                    return True
                return False
            
            gen = (m for m in target.modules() if _criterion(m))

            return gen

        def extract(self, parent):
            k_dict = {}
            s_dict = {}
            modules = [m for m in self.gen(parent)]
            
            for k, m in enumerate(modules):
                weights, scales = self.preprocess_kernels(m)
                k_dict[k] = weights
                s_dict[k] = nn.Parameter(scales, requires_grad=args.is_train)

            return k_dict, s_dict
    
        def preprocess_kernels(self, m):
            c_out, c_in, kh, kw = m.weight.size()
            weights = m.weight.data.view(c_out, c_in, kh * kw)
            scales = weights.norm(2, dim=2)

            scales.unsqueeze_(-1)
            scales.unsqueeze_(-1)  
            weights = weights.view(c_out * c_in, kh * kw)

            return weights, scales

        def quantize_kernels(
            self, parent, k_dict, s_dict, input_bits, scale_bits):

            modules_parent = [m for m in self.gen(parent)]
            modules_self = [m for m in sconv.gen_sconv(self)]
            
            for k, v in enumerate(modules_self):
                source = modules_parent[k]
                target = modules_self[k]

                target.set_params(
                    source,
                    kernels=k_dict[k],
                    scales=s_dict[k],
                    input_bits=input_bits,
                    scale_bits=scale_bits,
                )

        def state_dict(self, destination=None, prefix='', keep_vars=False):
            state = super(ScaleNet, self).state_dict(
                destination=destination, prefix=prefix, keep_vars=keep_vars)

            return state

        def load_state_dict(self, state_dict, strict=True, init=False):
            super(ScaleNet, self).load_state_dict(state_dict, strict=False)

    return ScaleNet(args)
