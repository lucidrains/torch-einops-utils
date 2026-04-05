from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor, arange

from einops import rearrange
from torch_einops_utils import exists, safe


def lens_to_mask(lens, max_len=None):
    device = lens.device

    if not exists(max_len):
        max_len = lens.amax().item()

    seq = arange(max_len, device=device)
    lens = rearrange(lens, "... -> ... 1")
    return seq < lens


@safe
def reduce_masks(masks: Sequence[Tensor], op: Callable[[Tensor, Tensor], Tensor]) -> Tensor:
    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask = op(mask, rest_mask)

    return mask


def and_masks(masks: Sequence[Tensor | None]) -> Tensor | None:
    return reduce_masks(masks, torch.logical_and)


def or_masks(masks: Sequence[Tensor | None]) -> Tensor | None:
    return reduce_masks(masks, torch.logical_or)
