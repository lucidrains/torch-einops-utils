from __future__ import annotations

from functools import wraps

from torch import is_tensor
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from einops import pack, unpack
from torch_einops_utils import default, exists, first, identity, pad_right_ndim


def masked_mean(
    t,
    mask=None,
    dim=None,
    eps=1e-5,
):
    if not exists(mask):
        return t.mean(dim=dim) if exists(dim) else t.mean()

    if mask.ndim < t.ndim:
        mask = pad_right_ndim(mask, t.ndim - mask.ndim)

    mask = mask.expand_as(t)

    if not exists(dim):
        return t[mask].mean() if mask.any() else t[mask].sum()

    num = (t * mask).sum(dim=dim)
    den = mask.sum(dim=dim)

    return num / den.clamp(min=eps)


# tree flatten with inverse


def tree_map_tensor(fn, tree):
    return tree_map(lambda t: fn(t) if is_tensor(t) else t, tree)


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

    def inverse(out, inv_pattern=None):
        inv_pattern = default(inv_pattern, pattern)
        out = unpack(out, packed_shape, inv_pattern)

        if is_one:
            out = first(out)

        return out

    return packed, inverse


def maybe(fn):

    if not exists(fn):
        return identity

    @wraps(fn)
    def inner(t, *args, **kwargs):
        if not exists(t):
            return None

        return fn(t, *args, **kwargs)

    return inner
