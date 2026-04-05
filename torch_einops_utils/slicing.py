from __future__ import annotations

import torch

from torch_einops_utils import exists


def slice_at_dim(t, slc, dim = -1):
    dims = t.ndim
    dim = (dim + dims) if dim < 0 else dim

    full_slice = [slice(None)] * dims
    full_slice[dim] = slc

    return t[tuple(full_slice)]


def slice_left_at_dim(t, length, dim = -1):
    if length == 0:
        return slice_at_dim(t, slice(0, 0), dim = dim)

    return slice_at_dim(t, slice(None, length), dim = dim)


def slice_right_at_dim(t, length, dim = -1):
    if length == 0:
        return slice_at_dim(t, slice(0, 0), dim = dim)

    return slice_at_dim(t, slice(-length, None), dim = dim)


def shape_with_replace(
    t,
    replace_dict: dict[int, int] | None = None,
):
    shape = t.shape

    if not exists(replace_dict):
        return shape

    shape_list = list(shape)

    for index, value in replace_dict.items():
        assert index < len(shape_list)
        shape_list[index] = value

    return torch.Size(shape_list)
