"""HyperStyle hypernetwork and W-encoder (inference only).

Adapted from https://github.com/yuval-alaluf/hyperstyle (MIT License, see licenses/LICENSE_hyperstyle).
Changes: device-agnostic parameter creation, no ImageNet download at construction time (weights
come from the HyperStyle checkpoint), the two hypernetwork variants share one class, cleaned imports.
Parameter names are unchanged so the official checkpoints load with ``strict=True``.
"""
import math

import numpy as np
import torch
from torch import nn
from torch.nn import BatchNorm2d, Conv2d, Linear, Module, PReLU, Sequential
from torchvision.models import resnet34

from ..models.encoders.helpers import bottleneck_IR, bottleneck_IR_SE, get_blocks
from ..models.stylegan2.model import EqualLinear

# StyleGAN2 (1024) layer index -> [kernel_size, in_channels, out_channels] of the modulated conv
PARAMETERS = {
    0: [3, 512, 512], 1: [1, 512, 3], 2: [3, 512, 512], 3: [3, 512, 512], 4: [1, 512, 3],
    5: [3, 512, 512], 6: [3, 512, 512], 7: [1, 512, 3], 8: [3, 512, 512], 9: [3, 512, 512],
    10: [1, 512, 3], 11: [3, 512, 512], 12: [3, 512, 512], 13: [1, 512, 3], 14: [3, 512, 256],
    15: [3, 256, 256], 16: [1, 256, 3], 17: [3, 256, 128], 18: [3, 128, 128], 19: [1, 128, 3],
    20: [3, 128, 64], 21: [3, 64, 64], 22: [1, 64, 3], 23: [3, 64, 32], 24: [3, 32, 32], 25: [1, 32, 3],
}
TO_RGB_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25]


class SeparableBlock(Module):
    def __init__(self, input_size, kernel_channels_in, kernel_channels_out, kernel_size):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.kernel_channels_in = kernel_channels_in
        self.kernel_channels_out = kernel_channels_out
        self.make_kernel_in = Linear(input_size, kernel_size * kernel_size * kernel_channels_in)
        self.make_kernel_out = Linear(input_size, kernel_size * kernel_size * kernel_channels_out)
        self.kernel_linear_in = Linear(kernel_channels_in, kernel_channels_in)
        self.kernel_linear_out = Linear(kernel_channels_out, kernel_channels_out)

    def forward(self, features):
        features = features.view(-1, self.input_size)
        kernel_in = self.make_kernel_in(features).view(-1, self.kernel_size, self.kernel_size, 1,
                                                       self.kernel_channels_in)
        kernel_out = self.make_kernel_out(features).view(-1, self.kernel_size, self.kernel_size,
                                                         self.kernel_channels_out, 1)
        kernel = torch.matmul(kernel_out, kernel_in)
        kernel = self.kernel_linear_in(kernel).permute(0, 1, 2, 4, 3)
        kernel = self.kernel_linear_out(kernel)
        return kernel.permute(0, 4, 3, 1, 2)


def _conv_stack(n_channels, inner_c, out_c, num_pools, first_stride):
    modules = [Conv2d(n_channels, inner_c, kernel_size=3, stride=first_stride, padding=1), nn.LeakyReLU()]
    for _ in range(num_pools - 1):
        modules += [Conv2d(inner_c, inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
    modules += [Conv2d(inner_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
    return nn.Sequential(*modules)


class RefinementBlock(Module):
    """Predicts the weight offsets of one (non-shared) StyleGAN layer."""

    def __init__(self, layer_idx, n_channels=512, inner_c=256, spatial=16, separable=False):
        super().__init__()
        self.layer_idx = layer_idx
        self.kernel_size, self.in_channels, self.out_channels = PARAMETERS[layer_idx]
        self.out_c = 512
        self.separable = separable
        num_pools = int(np.log2(spatial)) - 1
        if self.kernel_size == 3 and not separable:
            num_pools = num_pools - 1
        self.convs = _conv_stack(n_channels, inner_c, self.out_c, num_pools, first_stride=2)

        if layer_idx in TO_RGB_LAYERS:
            self.output = Sequential(Conv2d(self.out_c, self.in_channels * self.out_channels, kernel_size=1))
        elif separable:
            self.output = Sequential(SeparableBlock(self.out_c, self.in_channels, self.out_channels, self.kernel_size))
        else:
            self.output = Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                     Conv2d(self.out_c, self.in_channels * self.out_channels, kernel_size=1))

    def forward(self, x):
        x = self.output(self.convs(x))
        if self.layer_idx in TO_RGB_LAYERS:
            return x.view(-1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        if self.separable:
            return x
        x = x.view(-1, self.out_channels, self.in_channels)
        return x.unsqueeze(3).repeat(1, 1, 1, self.kernel_size).unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size)


class SharedWeightsHypernet(Module):
    def __init__(self, f_size=3, z_dim=512, out_size=512, in_size=512, mode=None):
        super().__init__()
        self.mode = mode
        self.z_dim = z_dim
        self.f_size = 1 if mode == 'delta_per_channel' else f_size
        self.out_size = out_size
        self.in_size = in_size
        n = self.out_size * self.f_size * self.f_size
        self.w1 = nn.Parameter(torch.fmod(torch.randn((self.z_dim, n)) / 40, 2))
        self.b1 = nn.Parameter(torch.fmod(torch.randn((n,)) / 40, 2))
        self.w2 = nn.Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size * self.z_dim)) / 40, 2))
        self.b2 = nn.Parameter(torch.fmod(torch.randn((self.in_size * self.z_dim,)) / 40, 2))

    def forward(self, z):
        batch_size = z.shape[0]
        h_in = (torch.matmul(z, self.w2) + self.b2).view(batch_size, self.in_size, self.z_dim)
        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(batch_size, self.out_size, self.in_size, self.f_size, self.f_size)
        if self.mode == 'delta_per_channel':
            kernel = kernel.repeat(1, 1, 1, 3, 3)
        return kernel


class HyperRefinementBlock(Module):
    """Refinement block of the layers that share the ``SharedWeightsHypernet``."""

    def __init__(self, hypernet, n_channels=512, inner_c=128, spatial=16):
        super().__init__()
        self.out_c = 512
        num_pools = int(np.log2(spatial))
        self.convs = _conv_stack(n_channels, inner_c, self.out_c, num_pools, first_stride=1)
        self.linear = EqualLinear(self.out_c, self.out_c, lr_mul=1)
        self.hypernet = hypernet

    def forward(self, features):
        code = self.convs(features).view(-1, self.out_c)
        return self.hypernet(self.linear(code))


class SharedWeightsHyperNetResNet(Module):
    SHARED_LAYERS = [0, 2, 3, 5, 6, 8, 9, 11, 12]

    def __init__(self, input_nc=6, n_outputs=26, layers_to_tune='', separable=False):
        super().__init__()
        self.conv1 = nn.Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)
        resnet = resnet34()  # ImageNet initialisation is only needed for training HyperStyle itself
        self.body = Sequential(*[b for layer in (resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
                                 for b in layer])

        if not layers_to_tune:
            self.layers_to_tune = list(range(n_outputs))
        else:
            self.layers_to_tune = [int(layer) for layer in str(layers_to_tune).split(',')]
        self.shared_weight_hypernet = SharedWeightsHypernet(in_size=512, out_size=512, mode=None)
        self.refinement_blocks = nn.ModuleList()
        self.n_outputs = n_outputs
        for layer_idx in range(n_outputs):
            if layer_idx not in self.layers_to_tune:
                block = None
            elif layer_idx in self.SHARED_LAYERS:
                block = HyperRefinementBlock(self.shared_weight_hypernet, n_channels=512, inner_c=128)
            else:
                block = RefinementBlock(layer_idx, n_channels=512, inner_c=256, separable=separable)
            self.refinement_blocks.append(block)

    def forward(self, x):
        x = self.body(self.relu(self.bn1(self.conv1(x))))
        return [block(x) if block is not None else None for block in self.refinement_blocks]


class WEncoder(Module):
    """e4e-style encoder that predicts a single code in W (the initial inversion of HyperStyle)."""

    def __init__(self, num_layers=50, mode='ir_se', output_size=1024):
        super().__init__()
        blocks = get_blocks(num_layers)
        unit_module = bottleneck_IR if mode == 'ir' else bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False), BatchNorm2d(64), PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        self.body = Sequential(*[unit_module(b.in_channel, b.depth, b.stride) for block in blocks for b in block])
        self.style_count = 2 * int(math.log(output_size, 2)) - 2

    def forward(self, x):
        x = self.body(self.input_layer(x))
        x = self.linear(self.output_pool(x).view(-1, 512))
        return x.repeat(self.style_count, 1, 1).permute(1, 0, 2)
