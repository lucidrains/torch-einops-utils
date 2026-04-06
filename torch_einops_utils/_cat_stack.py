from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, cat, stack

from torch_einops_utils import safe


@safe
def safe_stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
    return stack(tensors, dim=dim)  # type: ignore https://github.com/pytorch/pytorch/issues/179391


@safe
def safe_cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
    return cat(tensors, dim=dim)  # type: ignore https://github.com/pytorch/pytorch/issues/179391
