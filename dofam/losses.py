"""Loss functions of the DOLL module (Eq. 5-9 of the DO-FAM paper)."""
import numpy as np
import torch
import torch.nn.functional as F


def editing_targets(prob):
    """Flip every sample's attribute.

    Returns the target label (0/1) and the editing strength ``alpha``: ``-p`` when the attribute is
    present (p > 0.5) and ``1 - p`` otherwise, i.e. the distance to the opposite label.
    """
    present = prob > 0.5
    target = (~present).to(prob.dtype)
    alpha = torch.where(present, -prob, 1 - prob)
    return target, alpha


def attribute_related_loss(logits, target):
    """L_re, Eq. (6): class-balanced binary cross entropy of the edited attribute."""
    n = target.numel()
    pos_weight = n / (target.sum() + 1e-8)
    neg_weight = n / (n - target.sum() + 1e-8)
    weight = pos_weight * target + neg_weight * (1 - target)
    return (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * weight).mean()


def attribute_unrelated_loss(logits_edit, logits_orig, mask):
    """L_un, Eq. (7): the other attributes, weighted by (1 - |gamma_ik|), must not change."""
    return F.mse_loss(logits_edit * mask, logits_orig * mask)


def orthogonality_loss(w_un, w_re, w_re_t, mode='gram'):
    """L_ort, Eq. (5): attribute-related codes (before and after LTNet) orthogonal to the unrelated code.

    ``mode='gram'`` reproduces the released models: mean |<w_un^i, w_re^j>| over all pairs (i, j) of
    the batch. ``mode='elementwise'`` is the per-sample form written in the paper, ||w_re (.) w_un||_1.
    """
    if mode == 'gram':
        return (w_un @ w_re.t()).abs().mean() + (w_un @ w_re_t.t()).abs().mean()
    if mode == 'elementwise':
        return (w_re * w_un).abs().sum(1).mean() + (w_re_t * w_un).abs().sum(1).mean()
    raise ValueError(mode)


def correlation_mask(labels, attr_index, threshold=1.0):
    """``1 - |gamma_ik|`` from the attribute correlation matrix (Eq. 7)."""
    corr = np.corrcoef(np.asarray(labels, dtype=np.float64).T)
    corr[np.isnan(corr)] = 0
    rho = np.abs(corr[attr_index]).copy()
    rho[rho >= threshold] = 1
    return torch.from_numpy(1 - rho).float()
