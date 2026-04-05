from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, cat, stack

from torch_einops_utils import compact, safe


# TODO If @safe didn't return on len==1, this could use @safe.
def safe_stack(tensors: Sequence[Tensor | None], dim: int = 0) -> Tensor | None:
    output: list[Tensor] = compact(tensors)

    if len(output) == 0:
        return None

    return stack(output, dim=dim)


@safe
def safe_cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
    return cat(tensors, dim=dim)  # type: ignore https://github.com/pytorch/pytorch/issues/179391
