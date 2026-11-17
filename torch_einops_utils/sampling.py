from __future__ import annotations

import torch
from torch import Tensor

# helpers

def _as_tensor(v, device = None, dtype = None):
    if torch.is_tensor(v):
        if device is not None or dtype is not None:
            v = v.to(device = device, dtype = dtype)
        return v

    return torch.as_tensor(v, device = device, dtype = dtype)

# functions

def uniform(
    shape,
    low = None,
    high = None,
    *,
    device = None,
    dtype = None,
    generator = None
):
    # single bound -> symmetric half-width, [-high, high]

    low = None if low is None else _as_tensor(low, device, dtype)
    high = None if high is None else _as_tensor(high, device, dtype)

    if low is None and high is None:
        low, high = _as_tensor(-1., device, dtype), _as_tensor(1., device, dtype)
    elif low is None:
        low = -high
    elif high is None:
        high = low.abs()
        low = -high

    rand = torch.rand(shape, device = device, dtype = dtype, generator = generator)
    return low + (high - low) * rand

def uniform_like(
    t: Tensor,
    low = None,
    high = None,
    *,
    generator = None
):
    return uniform(t.shape, low, high, device = t.device, dtype = t.dtype, generator = generator)
