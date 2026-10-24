from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd import Function

from einops import rearrange

# helpers

def _one_hot_mask(index, size):
    return torch.arange(size, device = index.device) == index

def _reduce(loss, reduction, denom = None):
    if reduction == 'none':
        return loss

    if reduction == 'sum':
        return loss.sum()

    return loss.mean() if denom is None else loss.sum() / denom

# stable softmax

def log_softmax(z):
    i_max = z.argmax(dim = -1, keepdim = True)
    z_max = z.gather(-1, i_max)
    is_max = _one_hot_mask(i_max, z.shape[-1])

    shifted = torch.where(is_max, 0., z - z_max)
    m = -torch.logsumexp(torch.where(is_max, -float('inf'), shifted), dim = -1, keepdim = True)

    return shifted - F.softplus(-m)

def softmax(z):
    return log_softmax(z).exp()

def _cross_entropy(z, target):
    target = rearrange(target.long(), '... -> ... 1')
    is_target = _one_hot_mask(target, z.shape[-1])
    z_target = z.gather(-1, target)

    m = -torch.logsumexp(torch.where(is_target, -float('inf'), z - z_target), dim = -1)

    return F.softplus(-m)

# soft targets

def _soft_target_terms(z_s, z_t):
    log_q = log_softmax(z_s)
    log_p = log_softmax(z_t)
    q = log_q.exp()
    p = log_p.exp()

    d = z_s - z_t
    d = d - (p * d).sum(dim = -1, keepdim = True)

    # expm1 for small d, log-space for large d

    term = torch.where(d > 1., torch.exp(log_p + d) - p, p * torch.expm1(d))
    sum_expm1 = term.sum(dim = -1, keepdim = True)
    s = torch.where(torch.isfinite(sum_expm1), d - torch.log1p(sum_expm1), log_q - log_p)

    u = torch.where(s.abs() < 1., p * torch.expm1(s), q - p)
    g = u - q * u.sum(dim = -1, keepdim = True)   # zero-sum projection

    return log_q, p, s, g

class _SoftTargetCrossEntropy(Function):
    @staticmethod
    def forward(ctx, z_s, z_t):
        log_q, p, _, g = _soft_target_terms(z_s.detach(), z_t.detach())
        ctx.save_for_backward(g)

        return -(p * log_q).sum(dim = -1)

    @staticmethod
    def backward(ctx, grad_output):
        g, = ctx.saved_tensors

        return rearrange(grad_output, '... -> ... 1') * g, None

def soft_target_cross_entropy(z_s, z_t):
    return _SoftTargetCrossEntropy.apply(z_s, z_t)

# kl

class _CategoricalKL(Function):
    @staticmethod
    def forward(ctx, z_p, z_q):
        need_grad_p, need_grad_q = ctx.needs_input_grad

        _, p, s, g = _soft_target_terms(z_q.detach(), z_p.detach())

        kl = torch.where(p > 0, -p * s, 0.).sum(dim = -1)

        grad_p = torch.where(p > 0, p * (-s - rearrange(kl, '... -> ... 1')), 0.) if need_grad_p else None

        ctx.save_for_backward(grad_p, g if need_grad_q else None)
        return kl

    @staticmethod
    def backward(ctx, grad_output):
        grad_p, grad_q = ctx.saved_tensors
        grad_output = rearrange(grad_output, '... -> ... 1')

        return (
            grad_output * grad_p if grad_p is not None else None,
            grad_output * grad_q if grad_q is not None else None
        )

def categorical_kl(z_p, z_q):
    return _CategoricalKL.apply(z_p, z_q)

def bernoulli_kl(z_p, z_q):
    logits_p = torch.stack((torch.zeros_like(z_p), z_p), dim = -1)
    logits_q = torch.stack((torch.zeros_like(z_q), z_q), dim = -1)

    return categorical_kl(logits_p, logits_q)

# nll

def cross_entropy(logits, target, ignore_index = -100, reduction = 'mean', class_dim = 1):
    assert logits.ndim > 1, 'logits must be batched'

    logits = torch.movedim(logits, class_dim, -1)

    target = target.long()
    valid = target != ignore_index

    target = torch.where(valid, target, 0)
    loss = torch.where(valid, _cross_entropy(logits, target), 0.)

    return _reduce(loss, reduction, valid.sum())

# binary cross entropy

def binary_cross_entropy_with_logits(logits, target, reduction = 'mean'):
    target = target.to(logits.dtype)

    # branch hard targets to avoid 0 * inf -> nan

    loss_pos = F.softplus(-logits)
    loss_neg = F.softplus(logits)

    loss = torch.where(
        target == 1,
        loss_pos,
        torch.where(target == 0, loss_neg, target * loss_pos + (1. - target) * loss_neg)
    )

    return _reduce(loss, reduction)

class _SoftTargetBinaryCrossEntropy(Function):
    @staticmethod
    def forward(ctx, z_s, z_t):
        p_pos = z_t.sigmoid()
        p_neg = (-z_t).sigmoid()

        loss = p_pos * F.softplus(-z_s) + p_neg * F.softplus(z_s)

        # sigmoid(z_s) - sigmoid(z_t), via expm1(-|delta|)

        delta = z_s - z_t
        g = torch.where(
            delta <= 0,
            (-z_s).sigmoid() * p_pos * torch.expm1(delta),
            -z_s.sigmoid() * p_neg * torch.expm1(-delta)
        )

        ctx.save_for_backward(g)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        g, = ctx.saved_tensors
        return grad_output * g, None

def soft_target_binary_cross_entropy(z_s, z_t, reduction = 'none'):
    return _reduce(_SoftTargetBinaryCrossEntropy.apply(z_s, z_t), reduction)

# Barron, Gnanapragasam & Tschannen. A Fix for Softmax and Cross-Entropy. 2026.
# https://jonbarron.info/softmaxfix/
