"""Loading DOLL checkpoints / HyperStyle weight offsets and editing codes with them."""
import glob
import os
import re
from collections import OrderedDict

import numpy as np
import torch

from .attributes import ALIASES, canonical_name
from .models import DOLL
from .io import torch_load


def load_doll(path, map_location='cpu'):
    ckpt = torch_load(path, map_location=map_location)
    state_dict = ckpt.get('state_dict', ckpt.get('DOLL_state_dict', ckpt))
    cfg = ckpt.get('model', {}) if 'state_dict' in ckpt else {}
    style_dim = state_dict['decomposer.0.weight'].shape[1]
    hidden_dim = state_dict['decomposer.0.weight'].shape[0]
    depth = len([k for k in state_dict if k.startswith('decomposer.') and k.endswith('.weight')])
    net = DOLL(style_dim=cfg.get('style_dim', style_dim), hidden_dim=cfg.get('hidden_dim', hidden_dim),
               depth=cfg.get('depth', depth))
    net.load_state_dict(state_dict)
    return net.eval()


def find_doll_checkpoint(model_dir, attr):
    """``DOLL_<Attribute>.pth`` (new), ``DOLL_<Attribute>.pth.tar`` / ``l2m_<Attribute>.pth.tar`` (released)."""
    name = canonical_name(attr)
    names = [name] + [alias for alias, target in ALIASES.items() if target == name]  # e.g. Male / Gender
    for n in names:
        for pattern in (f'DOLL_{n}.pth', f'DOLL_{n}.pth.tar', f'l2m_{n}.pth.tar', f'*{n}*.pth*'):
            matches = sorted(glob.glob(os.path.join(model_dir, pattern)))
            if matches:
                return matches[0]
    raise FileNotFoundError(f'No DOLL checkpoint for "{name}" in {model_dir}')


class DOLLEditor:
    """Edits W+ codes with per-attribute DOLL modules.

    A DOLL module has ~510M parameters (about 2 GB in fp32), so at most ``max_loaded`` of them are
    kept in memory at the same time. Only the first ``edit_layers`` style layers are modified; the
    fine layers are copied from the input code (as in the original implementation).
    """

    def __init__(self, model_dir, device='cpu', edit_layers=11, max_loaded=2):
        self.model_dir = model_dir
        self.device = device
        self.edit_layers = edit_layers
        self.max_loaded = max_loaded
        self._cache = OrderedDict()

    def doll(self, attr):
        name = canonical_name(attr)
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        while len(self._cache) >= self.max_loaded:
            self._cache.popitem(last=False)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._cache[name] = load_doll(find_doll_checkpoint(self.model_dir, name)).to(self.device)
        return self._cache[name]

    def _keep_fine_layers(self, w_edit, w):
        if self.edit_layers is None:
            return w_edit
        return torch.cat([w_edit[:, :self.edit_layers], w[:, self.edit_layers:]], dim=1)

    @torch.no_grad()
    def edit(self, w, attr, alpha):
        return self._keep_fine_layers(self.doll(attr).edit(w, alpha), w)

    @torch.no_grad()
    def sequential_edit(self, w, edits):
        w_edit = w
        for attr, alpha in edits:
            w_edit = self._keep_fine_layers(self.doll(attr).edit(w_edit, alpha), w)
        return w_edit


def parse_edits(spec):
    """``"Smiling=1.5,Eyeglasses=2"`` -> ``[("Smiling", 1.5), ("Eyeglasses", 2.0)]``."""
    edits = []
    for item in spec.split(','):
        if item.strip():
            name, value = item.split('=')
            edits.append((name.strip(), float(value)))
    return edits


# ----------------------------------------------------------------------------- latent / delta I/O
def save_weight_deltas(deltas, path, half=False):
    """Store the HyperStyle offsets of ONE image (list with tensors or None)."""
    out = [None if d is None else (d.half() if half else d.float()).cpu() for d in deltas]
    torch.save(out, path)


def load_weight_deltas(path, device='cpu'):
    """Load offsets written by :func:`save_weight_deltas` (``.pt``) or by the original code (``.npy``)."""
    if path.endswith('.npy'):
        deltas = np.load(path, allow_pickle=True)
        deltas = [None if d is None else torch.from_numpy(np.asarray(d, dtype=np.float32)) for d in deltas]
    else:
        deltas = torch_load(path, map_location='cpu')
    return [None if d is None else d.float().to(device) for d in deltas]


def _key(name):
    numbers = re.findall(r'\d+', name)
    return int(numbers[-1]) if numbers else name


def pair_latents_and_deltas(latent_dir, delta_dir=None):
    """Match ``<name>.pt`` latents with ``<name>.pt|.npy`` offsets. Falls back to the trailing number of
    the file name, so the released test data (``latent_code_000044.pt`` + ``000044.npy``) also works."""
    latents = sorted(f for f in os.listdir(latent_dir) if f.endswith(('.pt', '.npy')))
    deltas = {}
    if delta_dir:
        for f in os.listdir(delta_dir):
            if f.endswith(('.pt', '.npy')):
                deltas[os.path.splitext(f)[0]] = os.path.join(delta_dir, f)
                deltas.setdefault(_key(os.path.splitext(f)[0]), os.path.join(delta_dir, f))
    pairs = []
    for f in latents:
        stem = os.path.splitext(f)[0]
        delta = deltas.get(stem, deltas.get(_key(stem)))
        if delta_dir and delta is None:
            raise FileNotFoundError(f'No weight offsets found for {f} in {delta_dir}')
        pairs.append((stem, os.path.join(latent_dir, f), delta))
    return pairs
