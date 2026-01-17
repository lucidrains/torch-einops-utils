from __future__ import annotations
from functools import wraps

import torch
from torch import tensor, is_tensor, cat, stack, arange
import torch.nn.functional as F

from torch.utils._pytree import tree_flatten, tree_unflatten

from einops import rearrange, repeat, reduce, pack, unpack

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t, *args, **kwargs):
    return t

def first(arr):
    return arr[0]

def maybe(fn):

    if not exists(fn):
        return identity

    @wraps(fn)
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None

        return fn(t, *args, **kwargs)

    return inner

# exported functions

# dimensions

def pad_ndim(t, ndims: tuple[int, int]):
    shape = t.shape
    left, right = ndims

    ones = (1,)
    ones_left = ones * left
    ones_right = ones * right
    return t.reshape(*ones_left, *shape, *ones_right)

def pad_left_ndim(t, ndims: int):
    return pad_ndim(t, (ndims, 0))

def pad_right_ndim(t, ndims: int):
    return pad_ndim(t, (0, ndims))

def align_dims_left(
    tensors,
    *,
    ndim = None
):
    if not exists(ndim):
        ndim = max([t.ndim for t in tensors])

    return tuple(pad_right_ndim(t, ndim - t.ndim) for t in tensors)

# masking

def lens_to_mask(lens, max_len = None):
    device = lens.device

    if not exists(max_len):
        max_len = lens.amax().item()

    seq = arange(max_len, device = device)
    lens = rearrange(lens, '... -> ... 1')
    return seq < lens

def reduce_masks(masks, op):
    masks = [*filter(exists, masks)]

    if len(masks) == 0:
        return None
    elif len(masks) == 1:
        return first(masks)

    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask = op(mask, rest_mask)

    return mask

def and_masks(masks):
    return reduce_masks(masks, torch.logical_and)

def or_masks(masks):
    return reduce_masks(masks, torch.logical_or)

# padding

def pad_at_dim(
    t,
    pad: tuple[int, int],
    *,
    dim = -1,
    value = 0.
):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pad_left_at_dim(t, pad: int, **kwargs):
    return pad_at_dim(t, (pad, 0), **kwargs)

def pad_right_at_dim(t, pad: int, **kwargs):
    return pad_at_dim(t, (0, pad), **kwargs)

# better pad sequence

def pad_sequence(
    tensors,
    *,
    dim = -1,
    value = 0.,
    left = False,
    dim_stack = 0,
    return_lens = False
):
    if len(tensors) == 0:
        return None

    device = first(tensors).device

    lens = tensor([t.shape[dim] for t in tensors], device = device)
    max_len = lens.amax().item()

    pad_fn = pad_left_at_dim if left else pad_right_at_dim
    padded_tensors = [pad_fn(t, max_len - t_len, dim = dim, value = value) for t, t_len in zip(tensors, lens)]

    stacked = stack(padded_tensors, dim = dim_stack)

    if not return_lens:
        return stacked

    return stacked, lens

# tree flatten with inverse

def tree_flatten_with_inverse(tree):
    flattened, spec = tree_flatten(tree)

    def inverse(out):
        return tree_unflatten(out, spec)

    return flattened, inverse

# einops pack

def pack_with_inverse(t, pattern):
    is_one = is_tensor(t)

    if is_one:
        t = [t]

    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        out = unpack(out, packed_shape, inv_pattern)

        if is_one:
            out = first(out)

        return out

    return packed, inverse
