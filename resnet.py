import torch
import torch.nn as nn
import torch.nn.functional as F
import common

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=True,
        conv3x3=common.default_conv, norm=common.default_norm, act=common.default_act):

        modules = []
        modules.append(conv3x3(
            in_channels, out_channels, kernel_size, stride=stride, bias=bias
        ))
        if norm is not None: modules.append(norm(out_channels))
        if act is not None: modules.append(act())

        super(BasicBlock, self).__init__(*modules)
        
class ResBlock(nn.Module):
    def __init__(
        self, in_channels, planes, kernel_size, stride=1,
        conv3x3=common.default_conv,
        conv1x1=common.default_conv,
        norm=common.default_norm,
        act=common.default_act,
        downsample=None):

        super(ResBlock, self).__init__()
        m = [conv3x3(
            in_channels, planes, kernel_size, stride=stride, bias=False
        )]
        if norm: m.append(norm(planes))
        m.append(act())
        m.append(conv3x3(planes, planes, kernel_size, bias=False))
        if norm: m.append(norm(planes))

        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = act()

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None: x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out
    
class BottleNeck(nn.Module):
    def __init__(
        self, in_channels, planes, kernel_size, stride=1,
        conv3x3=common.default_conv,
        conv1x1=common.default_conv,
        norm=common.default_norm,
        act=common.default_act,
        downsample=None):

        super(BottleNeck, self).__init__()
        m = [conv1x1(in_channels, planes, 1, bias=False)]
        if norm: m.append(norm(planes))
        m.append(act())
        m.append(conv3x3(planes, planes, kernel_size, stride=stride, bias=False))
        if norm: m.append(norm(planes))
        m.append(act())
        m.append(conv1x1(planes, 4 * planes, 1, bias=False))
        if norm: m.append(norm(4 * planes))
        
        self.body = nn.Sequential(*m)
        self.downsample = downsample
        self.act_out = act()

    def forward(self, x):
        out = self.body(x)
        if self.downsample is not None: x = self.downsample(x)
        out += x
        out = self.act_out(out)

        return out
    
class DownSample(nn.Sequential):
    def __init__(
        self, in_channels, out_channels,
        stride=1, conv1x1=common.default_conv):

        m = [
            conv1x1(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        super(DownSample, self).__init__(*m)
        
class ResNet(nn.Module):
    def __init__(self, args,
                 conv3x3=common.default_conv, 
                 conv1x1=common.default_conv):
        super(ResNet, self).__init__()
               
        m = []
        block_config = {
            18: ([2, 2, 2, 2], ResBlock, 1),
            34: ([3, 4, 6, 3], ResBlock, 1),
            50: ([3, 4, 6, 3], BottleNeck, 4),
            101: ([3, 4, 23, 3], BottleNeck, 4),
            152: ([3, 8, 36, 3], BottleNeck, 4)
        }
        n_blocks, self.block, self.expansion = block_config[args.depth]
        self.in_channels = 64
        
        n_classes = 10
        if 'cifar' in args.dataset:
            n_classes = int(args.dataset[5:])
        kwargs = {
            'conv3x3': conv3x3,
            'conv1x1': conv1x1,
            'act': common.act_dict[args.activation]
        }
        m.append(BasicBlock(
            3, 64, 3, stride=1, conv3x3=conv3x3, bias=False
        ))
        m.append(self.make_layer(64, n_blocks[0], 3, **kwargs))
        m.append(self.make_layer(128, n_blocks[1], 3, stride=2, **kwargs))
        m.append(self.make_layer(256, n_blocks[2], 3, stride=2, **kwargs))
        m.append(self.make_layer(512, n_blocks[3], 3, stride=2, **kwargs))
        
        fc = nn.Linear(512 * self.expansion, n_classes)

        self.features= nn.Sequential(*m)
        self.classifier = fc

    def make_layer(
        self, planes, blocks, kernel_size, stride=1,
        conv3x3=common.default_conv,
        conv1x1=common.default_conv,
        norm=common.default_norm,
        act=common.default_act,
        bias=False):

        out_channels = planes * self.expansion
        if stride != 1 or self.in_channels != out_channels:
            downsample = DownSample(
                self.in_channels,
                out_channels,
                stride=stride,
                conv1x1=conv1x1
            )
        else:
            downsample = None

        kwargs = {
            'conv3x3': conv3x3,
            'conv1x1': conv1x1,
            'act': act
        }
        m = [self.block(
            self.in_channels, planes, kernel_size,
            stride=stride, downsample=downsample, **kwargs
        )]
        self.in_channels = out_channels

        for _ in range(blocks - 1):
            m.append(self.block(
                self.in_channels, planes, kernel_size, **kwargs
            ))

        return nn.Sequential(*m)

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool2d(x, 4)
        x = self.classifier(x.view(x.size(0), -1))

        return x
    
def make_model(args):
    return ResNet(args)