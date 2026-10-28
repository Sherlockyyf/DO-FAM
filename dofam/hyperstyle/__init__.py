"""Minimal HyperStyle inference used by DO-FAM to obtain latent codes and generator weight offsets.

    net = load_hyperstyle('pretrained_models/hyperstyle_ffhq.pt',
                          'pretrained_models/faces_w_encoder.pt', device='cuda')
    images, latents, deltas = run_inversion(x, net, n_iters=5)

Adapted from https://github.com/yuval-alaluf/hyperstyle (MIT License).
"""
import math

import torch
from torch import nn

from ..models.stylegan2 import Generator
from .networks import SharedWeightsHyperNetResNet, WEncoder
from ..io import torch_load


def _filter_keys(state_dict, prefix):
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    return {k[len(prefix) + 1:]: v for k, v in state_dict.items() if k.startswith(prefix + '.')}


class HyperStyle(nn.Module):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.output_size = opts.get('output_size', 1024)
        self.n_styles = int(math.log(self.output_size, 2)) * 2 - 2
        n_outputs = {1024: 26, 512: 23, 256: 20}[self.output_size]
        self.hypernet = SharedWeightsHyperNetResNet(
            input_nc=opts.get('input_nc', 6), n_outputs=n_outputs, layers_to_tune=opts.get('layers_to_tune', ''),
            separable=opts.get('encoder_type', 'SharedWeightsHyperNetResNet') == 'SharedWeightsHyperNetResNetSeparable')
        self.decoder = Generator(self.output_size, 512, 8, channel_multiplier=2)
        self.w_encoder = WEncoder(50, 'ir_se', self.output_size)
        self.face_pool = nn.AdaptiveAvgPool2d((256, 256))
        self.register_buffer('latent_avg', torch.zeros(self.n_styles, 512), persistent=False)

    @torch.no_grad()
    def initial_inversion(self, x):
        """W-encoder inversion: code in W (repeated n_styles times) and its 256x256 reconstruction."""
        codes = self.w_encoder(x) + self.latent_avg.unsqueeze(0)
        y_hat, _ = self.decoder([codes], input_is_latent=True, randomize_noise=False)
        return self.face_pool(y_hat), codes

    def forward(self, x, y_hat=None, codes=None, weights_deltas=None, resize=True):
        if y_hat is None:
            y_hat, codes = self.initial_inversion(x)
        # The hypernetwork sees the input image and the current reconstruction.
        hypernet_outputs = self.hypernet(torch.cat([x, y_hat], dim=1))
        if weights_deltas is None:
            weights_deltas = hypernet_outputs
        else:
            weights_deltas = [weights_deltas[i] + hypernet_outputs[i] if weights_deltas[i] is not None else None
                              for i in range(len(hypernet_outputs))]
        images, _ = self.decoder([codes], weights_deltas=weights_deltas, input_is_latent=True, randomize_noise=False)
        if resize:
            images = self.face_pool(images)
        return images, codes, weights_deltas


def load_hyperstyle(checkpoint_path, w_encoder_path=None, device='cuda'):
    """Load an official HyperStyle checkpoint. ``w_encoder_path`` defaults to the path stored in it."""
    ckpt = torch_load(checkpoint_path, map_location='cpu')
    opts = dict(ckpt.get('opts', {}))
    net = HyperStyle(opts)
    net.hypernet.load_state_dict(_filter_keys(ckpt, 'hypernet'), strict=True)
    net.decoder.load_state_dict(_filter_keys(ckpt, 'decoder'), strict=True)
    net.latent_avg.copy_(ckpt['latent_avg'].view(-1, 512).expand(net.n_styles, 512))

    w_encoder_path = w_encoder_path or opts.get('w_encoder_checkpoint_path')
    w_ckpt = torch_load(w_encoder_path, map_location='cpu')
    net.w_encoder.load_state_dict(_filter_keys(w_ckpt, 'encoder'), strict=True)
    return net.eval().to(device)


@torch.no_grad()
def run_inversion(x, net, n_iters=5):
    """Iterative refinement of HyperStyle (Alaluf et al.): ``n_iters`` forward passes.

    Returns the final 1024x1024 reconstructions, the W codes ([B, 18, 512]) and the per-layer weight
    offsets (list of 26 tensors of shape [B, C_out, C_in, k, k] or ``None``).
    """
    y_hat, codes, deltas = None, None, None
    images = None
    for i in range(n_iters):
        last = i == n_iters - 1
        images, codes, deltas = net(x, y_hat=y_hat, codes=codes, weights_deltas=deltas, resize=not last)
        y_hat = net.face_pool(images)
    return images, codes, deltas
