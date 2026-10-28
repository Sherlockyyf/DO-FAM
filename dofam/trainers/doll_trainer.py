"""Training loop of DOLL (Sec. 2.1 / 2.3 of the DO-FAM paper).

The latent classifier is pre-trained and frozen; the decomposer and LTNet are trained jointly with

    L_DOLL = L_re + lambda_un * L_un + lambda_reg * L_reg + lambda_ort * L_ort          (Eq. 9)
"""
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..attributes import attribute_index, canonical_name
from ..losses import (attribute_related_loss, attribute_unrelated_loss, correlation_mask, editing_targets,
                      orthogonality_loss)
from ..models import DOLL
from ..utils import make_row

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


class DOLLTrainer:
    def __init__(self, cfg, attr, classifier, train_labels, exp_dir, device):
        self.cfg = cfg
        self.attr = canonical_name(attr)
        self.k = attribute_index(self.attr)
        self.exp_dir = exp_dir
        self.device = device
        self.classifier = classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)

        self.model_cfg = cfg['model']
        self.net = DOLL(**self.model_cfg).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg['lr'], betas=tuple(cfg['betas']),
                                          weight_decay=cfg['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg['step_size'],
                                                         gamma=cfg['gamma'])
        self.mask = correlation_mask(train_labels, self.k, cfg['corr_threshold']).to(device)
        self.writer = SummaryWriter(os.path.join(exp_dir, 'logs', self.attr)) if SummaryWriter else None

    def compute_loss(self, w):
        lam = self.cfg['lambda']
        w = w.flatten(1)
        logits0 = self.classifier(w)
        prob0 = torch.sigmoid(logits0[:, self.k])
        target, alpha = editing_targets(prob0)
        if not self.cfg.get('adaptive_alpha', True):
            alpha = 2 * target - 1

        w_un, w_re, w_re_t = self.net(w)
        w_edit = w_un + w_re + alpha.unsqueeze(1) * (w_re_t - w_re)   # Eq. (3)-(4)
        logits1 = self.classifier(w_edit)

        loss_re = attribute_related_loss(logits1[:, self.k], target)
        loss_un = attribute_unrelated_loss(logits1, logits0, self.mask)
        loss = lam['re'] * loss_re + lam['un'] * loss_un
        logs = {'loss_re': loss_re.item(), 'loss_un': loss_un.item()}
        if lam['ort'] > 0:
            loss_ort = orthogonality_loss(w_un, w_re, w_re_t, self.cfg.get('ort_mode', 'gram'))
            loss = loss + lam['ort'] * loss_ort
            logs['loss_ort'] = loss_ort.item()
        if lam['reg'] > 0:
            loss_reg = F.mse_loss(w_edit, w)                          # Eq. (8)
            loss = loss + lam['reg'] * loss_reg
            logs['loss_reg'] = loss_reg.item()
        logs['loss'] = loss.item()
        return loss, logs

    def train(self, dataset, preview=None):
        """``preview``: optional ``(generator, w, weight_deltas)`` used to save sample edits."""
        cfg = self.cfg
        loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                            num_workers=cfg.get('workers', 0))
        step, tic = 0, time.time()
        for epoch in range(cfg['epochs']):
            for w, _ in loader:
                loss, logs = self.compute_loss(w.to(self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if step % cfg['log_interval'] == 0:
                    msg = ', '.join(f'{k}={v:.4f}' for k, v in logs.items())
                    print(f'[{self.attr} | epoch {epoch} | step {step} | {time.time() - tic:.0f}s] {msg}', flush=True)
                    if self.writer:
                        for k, v in logs.items():
                            self.writer.add_scalar(k, v, step)
                if preview is not None and step % cfg['image_interval'] == 0:
                    self.save_preview(*preview, step=step)
                step += 1
            self.scheduler.step()
        return self.save()

    @torch.no_grad()
    def save_preview(self, generator, w, deltas, step, alphas=(-1.5, -0.5, 0.5, 1.5), edit_layers=11):
        self.net.eval()
        w = w.to(self.device)
        images = [generator([w], input_is_latent=True, randomize_noise=False, weights_deltas=deltas)[0][0]]
        for a in alphas:
            w_edit = self.net.edit(w, a)
            w_edit = torch.cat([w_edit[:, :edit_layers], w[:, edit_layers:]], dim=1)
            images.append(generator([w_edit], input_is_latent=True, randomize_noise=False, weights_deltas=deltas)[0][0])
        out_dir = os.path.join(self.exp_dir, 'images', self.attr)
        os.makedirs(out_dir, exist_ok=True)
        make_row(images).save(os.path.join(out_dir, f'step_{step:06d}.jpg'))
        self.net.train()

    def save(self):
        path = os.path.join(self.exp_dir, 'checkpoints', f'DOLL_{self.attr}.pth')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'state_dict': self.net.state_dict(), 'attr': self.attr, 'model': self.model_cfg}, path)
        print(f'Saved {path}')
        return path
