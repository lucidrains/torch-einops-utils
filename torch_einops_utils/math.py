from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange

from torch_einops_utils.torch_einops_utils import default

# helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

# projections of v onto u
# v = parallel + orthogonal

def parallel_component(v, u, dim = -1, eps = None):
    eps = default(eps, torch.finfo(u.dtype).tiny)
    v, u = v.movedim(dim, -1), u.movedim(dim, -1)
    denom = (u * u).sum(dim = -1, keepdim = True).clamp_min(eps)
    coeff = (v * u).sum(dim = -1, keepdim = True) / denom
    return (coeff * u).movedim(-1, dim)

def orthogonal_component(v, u, dim = -1, eps = None):
    return v - parallel_component(v, u, dim = dim, eps = eps)

def components(v, u, dim = -1, eps = None):
    parallel = parallel_component(v, u, dim = dim, eps = eps)
    return parallel, v - parallel

# spherical linear interpolation

def slerp(
    v0,
    v1,
    t,
    eps = None,
    normalize = True
):
    dtype = v0.dtype
    device = v0.device
    eps = default(eps, torch.finfo(dtype).eps)

    assert v0.shape == v1.shape, 'v0 and v1 must have the same shape'

    *leading_shape, d = v0.shape

    t = torch.as_tensor(t, dtype = dtype, device = device)

    # align t with leading dims

    while len(t.shape) > len(leading_shape) and t.shape[-1] == 1:
        t = rearrange(t, '... 1 -> ...')

    batch_shape = torch.broadcast_shapes(leading_shape, t.shape)

    t = rearrange(t, '... -> ... 1').expand(*batch_shape, 1)

    if normalize:
        v0, v1 = l2norm(v0), l2norm(v1)

    v0 = v0.expand(*batch_shape, d)
    v1 = v1.expand(*batch_shape, d)

    dot = (v0 * v1).sum(dim = -1, keepdim = True)
    omega = dot.clamp(-1. + eps, 1. - eps).acos()
    sin_omega = omega.sin()

    return (((1. - t) * omega).sin() * v0 + (t * omega).sin() * v1) / sin_omega
