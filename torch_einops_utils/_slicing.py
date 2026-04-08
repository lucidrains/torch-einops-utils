from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from torch_einops_utils import exists


def slice_at_dim(t: Tensor, slc: slice, dim: int = -1) -> Tensor:
    dims: int = t.ndim
    dim = (dim + dims) if dim < 0 else dim

    full_slice: list[slice] = [slice(None)] * dims
    full_slice[dim] = slc

    return t[tuple(full_slice)]


def slice_left_at_dim(t: Tensor, length: int, dim: int = -1) -> Tensor:
    if length == 0:
        return slice_at_dim(t, slice(0, 0), dim=dim)

    return slice_at_dim(t, slice(None, length), dim=dim)


def slice_right_at_dim(t: Tensor, length: int, dim: int = -1) -> Tensor:
    if length == 0:
        return slice_at_dim(t, slice(0, 0), dim=dim)

    return slice_at_dim(t, slice(-length, None), dim=dim)


def shape_with_replace(
    t: Tensor,
    replace_dict: Mapping[int, int] | None = None,
) -> torch.Size:
    shape: torch.Size = t.shape

    if not exists(replace_dict):
        return shape

    shape_list: list[int] = list(shape)

    for index, value in replace_dict.items():
        if index >= len(shape_list):
            message: str = f"I received `{index = }`, but I need `index` to be less than `{len(shape_list) = }`."
            raise ValueError(message)
        shape_list[index] = value

    return torch.Size(shape_list)
