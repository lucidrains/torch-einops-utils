from __future__ import annotations

from torch_einops_utils import exists


def pad_ndim(t, ndims: tuple[int, int]):
    shape = t.shape
    left, right = ndims
    assert left >= 0 and right >= 0

    ones = (1,)
    ones_left = ones * left
    ones_right = ones * right
    return t.reshape(*ones_left, *shape, *ones_right)


def pad_left_ndim(t, ndims: int):
    return pad_ndim(t, (ndims, 0))


def pad_right_ndim(t, ndims: int):
    return pad_ndim(t, (0, ndims))


def pad_right_ndim_to(t, ndims: int):
    if t.ndim >= ndims:
        return t

    return pad_right_ndim(t, ndims - t.ndim)


def pad_left_ndim_to(t, ndims: int):
    if t.ndim >= ndims:
        return t

    return pad_left_ndim(t, ndims - t.ndim)


def align_dims_left(
    tensors,
    *,
    ndim=None,
):
    if not exists(ndim):
        ndim = max([t.ndim for t in tensors])

    return tuple(pad_right_ndim(t, ndim - t.ndim) for t in tensors)
