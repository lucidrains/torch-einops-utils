from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

# helpers

def _one_hot_mask(index, size):
    return F.one_hot(rearrange(index, '... 1 -> ...'), size).bool()

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

def cross_entropy(z, target):
    target = rearrange(target.long(), '... -> ... 1')
    is_target = _one_hot_mask(target, z.shape[-1])
    z_target = z.gather(-1, target)

    m = -torch.logsumexp(torch.where(is_target, -float('inf'), z - z_target), dim = -1)

    return F.softplus(-m)

# soft targets

def _soft_target_terms(z_s, z_t):
    log_q = log_softmax(z_s)
    q = log_q.exp()
    p = softmax(z_t)

    d = z_s - z_t
    d = d - (p * d).sum(dim = -1, keepdim = True)

    weighted = torch.where(p > 0, p * torch.expm1(d), 0.)
    s = d - torch.log1p(weighted.sum(dim = -1, keepdim = True).clamp_min(-1.))
    u = torch.where(s.abs() < 1., p * torch.expm1(s), q - p)

    return log_q, p, s, u

class _SoftTargetCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_s, z_t):
        log_q, p, _, u = _soft_target_terms(z_s.detach(), z_t.detach())
        ctx.save_for_backward(u)

        return -(p * log_q).sum(dim = -1)

    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors

        return rearrange(grad_output, '... -> ... 1') * u, None

def soft_target_cross_entropy(z_s, z_t):
    return _SoftTargetCrossEntropy.apply(z_s, z_t)

# kl

class _CategoricalKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z_p, z_q):
        _, p, s, u = _soft_target_terms(z_q.detach(), z_p.detach())

        kl = torch.where(p > 0, -p * s, 0.).sum(dim = -1)
        grad_p = torch.where(p > 0, p * (-s - rearrange(kl, '... -> ... 1')), 0.)

        ctx.save_for_backward(grad_p, u)
        return kl

    @staticmethod
    def backward(ctx, grad_output):
        grad_p, grad_q = ctx.saved_tensors
        grad_output = rearrange(grad_output, '... -> ... 1')

        return grad_output * grad_p, grad_output * grad_q

def categorical_kl(z_p, z_q):
    return _CategoricalKL.apply(z_p, z_q)

def bernoulli_kl(z_p, z_q):
    logits_p = torch.stack((torch.zeros_like(z_p), z_p), dim = -1)
    logits_q = torch.stack((torch.zeros_like(z_q), z_q), dim = -1)

    return categorical_kl(logits_p, logits_q)

# Barron, Gnanapragasam & Tschannen. A Fix for Softmax and Cross-Entropy. 2026.
# https://jonbarron.info/softmaxfix/
