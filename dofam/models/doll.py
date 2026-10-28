"""DOLL: Disentangled nOn-Linear Latent transformation module (Sec. 2.1 of the DO-FAM paper).

Given a (flattened) W+ code ``w``:

    w_re = P(w)                      decomposer: attribute-related part          (Eq. 2)
    w_un = w - P(w)                  attribute-unrelated part                    (Eq. 2)
    w~_re = T(w_re)                  LTNet: non-linear navigation of w_re        (Eq. 3)
    w_m  = w_re + alpha (T(w_re) - w_re)
    w~   = w_un + w_m                edited code                                 (Eq. 4)

Both the decomposer P and the latent transformation network T are three fully-connected layers
on the 18 x 512 = 9216 dimensional code. Parameter names are kept identical to the original
implementation, so the released DOLL checkpoints load directly.
"""
import torch
from torch import nn


def _mlp(dim, hidden_dim, depth):
    layers = []
    in_dim = dim
    for i in range(depth):
        out_dim = dim if i == depth - 1 else hidden_dim
        layers.append(nn.Linear(in_dim, out_dim))
        if i < depth - 1:
            layers.append(nn.ReLU())
        in_dim = out_dim
    return nn.Sequential(*layers)


class DOLL(nn.Module):
    def __init__(self, style_dim=9216, hidden_dim=None, depth=3):
        super().__init__()
        hidden_dim = hidden_dim or style_dim
        self.style_dim = style_dim
        self.decomposer = _mlp(style_dim, hidden_dim, depth)
        self.transformer = _mlp(style_dim, hidden_dim, depth)

    def forward(self, w):
        """Returns ``(w_un, w_re, w_re_transformed)`` for a code of shape [B, 18, 512] or [B, 9216]."""
        w = w.flatten(1)
        w_re = self.decomposer(w)
        w_un = w - w_re
        return w_un, w_re, self.transformer(w_re)

    def edit(self, w, alpha):
        """Edited code ``w + alpha * (T(w_re) - w_re)`` with the same shape as ``w``."""
        shape = w.shape
        w_un, w_re, w_re_t = self(w)
        alpha = torch.as_tensor(alpha, dtype=w.dtype, device=w.device).reshape(-1, 1)
        return (w_un + w_re + alpha * (w_re_t - w_re)).view(shape)
