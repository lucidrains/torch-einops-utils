from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Literal, TypedDict, overload

import torch.nn.functional as F
from torch import Tensor, cat, device, stack, tensor

from torch_einops_utils import first

from hunterMakesPy import decreasing, raiseIfNone
from typing_extensions import Unpack


class _DimAndValue(TypedDict, total=False):
    dim: int
    value: float


class _DimValueLeft(_DimAndValue, total=False):
    left: bool


def pad_at_dim(
    t: Tensor,
    pad: tuple[int, int],
    *,
    dim: int = -1,
    value: float = 0.0,
) -> Tensor:
    dims_from_right: int = (decreasing * dim + decreasing + t.ndim) % t.ndim
    zeros: tuple[Literal[0], ...] = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def pad_left_at_dim(t: Tensor, pad: int, **kwargs: Unpack[_DimAndValue]) -> Tensor:
    return pad_at_dim(t, (pad, 0), **kwargs)


def pad_right_at_dim(t: Tensor, pad: int, **kwargs: Unpack[_DimAndValue]) -> Tensor:
    return pad_at_dim(t, (0, pad), **kwargs)


def pad_left_at_dim_to(t: Tensor, length: int, dim: int = -1, **kwargs: float) -> Tensor:
    curr_len: int = t.shape[dim]
    if curr_len >= length:
        return t

    return pad_left_at_dim(t, length - curr_len, dim=dim, **kwargs)


def pad_right_at_dim_to(t: Tensor, length: int, dim: int = -1, **kwargs: float) -> Tensor:
    curr_len: int = t.shape[dim]
    if curr_len >= length:
        return t

    return pad_right_at_dim(t, length - curr_len, dim=dim, **kwargs)


@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.0,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[True] = True,
    return_lens: Literal[False] = False,
    pad_lens: bool = False,
) -> Tensor | None: ...
@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.0,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[True] = True,
    return_lens: Literal[True],
    pad_lens: bool = False,
) -> tuple[Tensor, Tensor] | None: ...
@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.0,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[False],
    return_lens: Literal[False] = False,
    pad_lens: bool = False,
) -> list[Tensor] | None: ...
@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.0,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[False],
    return_lens: Literal[True],
    pad_lens: bool = False,
) -> tuple[list[Tensor], Tensor] | None: ...
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.0,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: bool = True,
    return_lens: bool = False,
    pad_lens: bool = False,  # returns padding lengths instead of dimension lengths
) -> Tensor | list[Tensor] | tuple[Tensor | list[Tensor], Tensor] | None:
    output: Tensor | list[Tensor] | tuple[Tensor | list[Tensor], Tensor] | None = None

    if len(tensors) > 0:
        lens: list[int] | Tensor = [t.shape[dim] for t in tensors]
        max_len: int = max(lens)

        pad_fn: Callable[..., Tensor] = pad_left_at_dim if left else pad_right_at_dim
        padded_tensors: list[Tensor] = [
            pad_fn(t, max_len - t_len, dim=dim, value=value) for t, t_len in zip(tensors, lens)
        ]

        output = stack(padded_tensors, dim=dim_stack) if return_stacked else padded_tensors

        if return_lens:
            device: device = first(tensors).device
            lens = tensor(lens, device=device)

            if pad_lens:
                lens = max_len - lens

            output = (output, lens)

    return output


"""Alternative flow with multiple returns:
    if len(tensors) == 0:
        return None

    pad_fn: Callable[..., Tensor] = pad_left_at_dim if left else pad_right_at_dim

    lens: list[int] | Tensor = [t.shape[dim] for t in tensors]
    max_len: int = max(lens)

    padded_tensors: list[Tensor] = [
        pad_fn(t, max_len - t_len, dim=dim, value=value) for t, t_len in zip(tensors, lens)
    ]

    output: Tensor | list[Tensor] = (
        stack(padded_tensors, dim=dim_stack) if return_stacked else padded_tensors
    )

    if not return_lens:
        return output

    device: device = first(tensors).device
    lens = tensor(lens, device=device)

    if pad_lens:
        lens = max_len - lens

    return output, lens
"""

"""pad_sequence_and_cat
1. If len(tensors) == 0?
- return None
- issue a warning and return None
- let torch.cat raise the Exception: `TypeError: cat(): argument 'tensors' (position 1) must be tuple
    of Tensors, not NoneType`
- raise an Exception with a custom message
2. `assert`, generally: use something else. https://docs.astral.sh/ruff/rules/assert/, jit might
    remove the assert.
3. Function signature
- **kwargs: Any. The user has to figure it out.
- **kwargs: Unpack[...]. Some IDEs tell the user the names and types but not the default values.
- **kwargs. If you pass **kwargs to `pad_sequence`, you have to protect yourself from "bad" kwargs.
- Explicitly list the parameters. It creates duplicate code -- especially the default values. But it
    also allows you to put the parameters in the same order as the other functions in this file.

"""


def pad_sequence_and_cat(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.0,
    left: bool = False,
    dim_cat: int = 0,
) -> Tensor | None:

    padded: Tensor | list[Tensor] | None = pad_sequence(
        tensors, dim=dim, value=value, left=left, return_stacked=False, return_lens=False
    )
    if padded is not None:
        padded = cat(padded, dim=dim_cat)
    return padded


def pad_sequence_and_catUNPACK(  # noqa: N802
    tensors: Sequence[Tensor],
    *,
    dim_cat: int = 0,
    **kwargs: Unpack[_DimValueLeft],
) -> Tensor | None:
    if len(tensors) == 0:
        return None

    kwargs.pop("return_stacked", None)
    kwargs.pop("return_lens", None)

    padded = pad_sequence(tensors, return_stacked=False, return_lens=False, **kwargs)
    return cat(padded, dim=dim_cat)


def pad_sequence_and_catRAISES(  # noqa: N802
    tensors: Sequence[Tensor],
    *,
    dim_cat: int = 0,
    **kwargs: Unpack[_DimValueLeft],
) -> Tensor:
    kwargs.pop("return_stacked", None)
    kwargs.pop("return_lens", None)

    padded: list[Tensor] = raiseIfNone(
        pad_sequence(tensors, return_stacked=False, return_lens=False, **kwargs),
        "`pad_sequence_and_cat` requires at least one `Tensor` to pad and concatenate",
    )
    return cat(padded, dim=dim_cat)
