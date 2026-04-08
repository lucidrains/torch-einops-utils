from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, overload

import torch.nn.functional as F
from torch import Tensor, cat, device, stack, tensor

from torch_einops_utils import DimAndValue, first

from typing_extensions import Unpack


def pad_at_dim(
    t: Tensor,
    pad: tuple[int, int],
    *,
    dim: int = -1,
    value: float = 0.0,
) -> Tensor:
    """Pad `t` along `dim`, inserting `value` at each end by the amounts in `pad`.

    You can use `pad_at_dim` to add fill elements to either side of any dimension of `t`. Negative
    values in `pad` trim elements from the corresponding end rather than adding them. `pad_at_dim`
    delegates to `torch.nn.functional.pad` [1] after constructing the flat padding tuple.

    Parameters
    ----------
    t : Tensor
        The input `Tensor` to pad.
    pad : tuple[int, int]
        A pair `(left, right)` specifying how many elements to insert before and after the existing
        content along `dim`. Negative values trim elements from the corresponding end.
    dim : int = -1
        The dimension along which to apply the padding.
    value : float = 0.0
        The scalar fill value used for inserted elements.

    Returns
    -------
    padded : Tensor
        The padded `Tensor`. The shape along `dim` is `t.shape[dim] + pad[0] + pad[1]`.

    See Also
    --------
    pad_left_at_dim : Pad only the left side of a dimension.
    pad_right_at_dim : Pad only the right side of a dimension.

    Examples
    --------
    From the test suite [2]:

        >>> import torch
        >>> from torch_einops_utils import pad_at_dim
        >>> t = torch.randn(3, 6, 1)
        >>> pad_at_dim(t, (0, 1), dim=1).shape
        torch.Size([3, 7, 1])

    From dreamer4 [3], inserting a leading zero token to shift action tokens forward by one position
    in the time dimension:

        ```python
        action_tokens = pad_at_dim(action_tokens[:, :-1], (1, 0), value=0.0, dim=1)
        ```

    From locoformer [4], creating a one-step-delayed copy of an action sequence:

        ```python
        past_action = pad_at_dim(action, (1, -1), dim=-2)
        ```

    References
    ----------
    [1] torch.nn.functional.pad - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html

    [2] tests/test_padding.py

    [3] lucidrains/dreamer4
        https://github.com/lucidrains/dreamer4

    [4] lucidrains/locoformer
        https://github.com/lucidrains/locoformer
    """
    dims_from_right: int = ((-1 * dim) - 1 + t.ndim) % t.ndim
    zeros: tuple[Literal[0], ...] = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value=value)


def pad_left_at_dim(t: Tensor, pad: int, **kwargs: Unpack[DimAndValue]) -> Tensor:
    """Pad `t` on the left side of a dimension by a fixed number of fill elements.

    You can use `pad_left_at_dim` to prepend fill elements along any dimension of `t`.
    `pad_left_at_dim` calls `pad_at_dim` [1] with `pad=(pad, 0)`. The keyword arguments `dim` and
    `value` are forwarded via `DimAndValue` [2].

    Parameters
    ----------
    t : Tensor
        The input `Tensor` to pad.
    pad : int
        The number of fill elements to insert before the existing content.
    **kwargs : Unpack[DimAndValue]
        Keyword arguments forwarded to `pad_at_dim`. Accepted keys are `dim` (`int`, default `-1`)
        and `value` (`float`, default `0.0`).

    Returns
    -------
    padded : Tensor
        The padded `Tensor`. The shape along the target dimension is `t.shape[dim] + pad`.

    See Also
    --------
    pad_at_dim : Pad with independent left and right amounts.
    pad_left_at_dim_to : Pad the left side to reach a target length.

    Examples
    --------
    From pi-zero-pytorch [3], prepending a start token marker to a sequence of discrete
    action identifiers:

        ```python
        discrete_action_ids_with_start = pad_left_at_dim(discrete_action_ids + 1, 1)
        ```

    References
    ----------
    [1] torch_einops_utils._padding.pad_at_dim

    [2] torch_einops_utils._types.DimAndValue

    [3] lucidrains/pi-zero-pytorch
        https://github.com/lucidrains/pi-zero-pytorch
    """
    return pad_at_dim(t, (pad, 0), **kwargs)


def pad_right_at_dim(t: Tensor, pad: int, **kwargs: Unpack[DimAndValue]) -> Tensor:
    """Pad `t` on the right side of a dimension by a fixed number of fill elements.

    You can use `pad_right_at_dim` to append fill elements along any dimension of `t`.
    `pad_right_at_dim` calls `pad_at_dim` [1] with `pad=(0, pad)`. The keyword arguments `dim` and
    `value` are forwarded via `DimAndValue` [2].

    Parameters
    ----------
    t : Tensor
        The input `Tensor` to pad.
    pad : int
        The number of fill elements to append after the existing content.
    **kwargs : Unpack[DimAndValue]
        Keyword arguments forwarded to `pad_at_dim`. Accepted keys are `dim` (`int`, default `-1`)
        and `value` (`float`, default `0.0`).

    Returns
    -------
    padded : Tensor
        The padded `Tensor`. The shape along the target dimension is `t.shape[dim] + pad`.

    See Also
    --------
    pad_at_dim : Pad with independent left and right amounts.
    pad_right_at_dim_to : Pad the right side to reach a target length.

    Examples
    --------
    From the test suite [3], verifying that `pad_right_at_dim` appends one element along a dimension:

        ```python
        import torch
        from torch_einops_utils import pad_right_at_dim

        t = torch.randn(3, 6, 1)
        assert pad_right_at_dim(t, 1, dim=1).shape == torch.Size([3, 7, 1])
        ```

    References
    ----------
    [1] torch_einops_utils._padding.pad_at_dim

    [2] torch_einops_utils._types.DimAndValue

    [3] tests/test_padding.py
    """
    return pad_at_dim(t, (0, pad), **kwargs)


def pad_left_at_dim_to(t: Tensor, length: int, dim: int = -1, **kwargs: float) -> Tensor:
    """Pad `t` on the left side of `dim` until `dim` reaches `length`.

    You can use `pad_left_at_dim_to` to ensure the size of `t` along `dim` is at least `length` by
    left-padding with fill values. If `t.shape[dim]` is already greater than or equal to `length`,
    `pad_left_at_dim_to` returns `t` unchanged. Otherwise, `pad_left_at_dim_to` calls
    `pad_left_at_dim` [1] with `pad = length - t.shape[dim]`.

    Parameters
    ----------
    t : Tensor
        The input `Tensor` to conditionally pad.
    length : int
        The minimum target size along `dim`.
    dim : int = -1
        The dimension to pad.
    **kwargs : float
        Keyword arguments forwarded to `pad_left_at_dim`. The accepted key is `value` (`float`,
        default `0.0`), the fill value for inserted elements.

    Returns
    -------
    padded : Tensor
        The padded `Tensor` with `padded.shape[dim] == max(t.shape[dim], length)`. Returns `t`
        unchanged when `t.shape[dim] >= length`.

    See Also
    --------
    pad_left_at_dim : Pad the left side by an explicit count.
    pad_right_at_dim_to : Pad the right side to a target length.

    Examples
    --------
    From the test suite [2]:

    >>> import torch
    >>> from torch_einops_utils import pad_left_at_dim_to
    >>> t = torch.randn(3, 6, 1)
    >>> pad_left_at_dim_to(t, 7, dim=1).shape
    torch.Size([3, 7, 1])
    >>> pad_left_at_dim_to(t, 6, dim=1) is t  # already at length, returned unchanged
    True

    References
    ----------
    [1] torch_einops_utils._padding.pad_left_at_dim

    [2] tests/test_utils.py
    """
    curr_len: int = t.shape[dim]
    if curr_len >= length:
        return t

    return pad_left_at_dim(t, length - curr_len, dim=dim, **kwargs)


def pad_right_at_dim_to(t: Tensor, length: int, dim: int = -1, **kwargs: float) -> Tensor:
    """Pad `t` on the right side of `dim` until `dim` reaches `length`.

    You can use `pad_right_at_dim_to` to ensure the size of `t` along `dim` is at least `length` by
    right-padding with fill values. If `t.shape[dim]` is already greater than or equal to `length`,
    `pad_right_at_dim_to` returns `t` unchanged. Otherwise, `pad_right_at_dim_to` calls
    `pad_right_at_dim` [1] with `pad = length - t.shape[dim]`.

    Parameters
    ----------
    t : Tensor
        The input `Tensor` to conditionally pad.
    length : int
        The minimum target size along `dim`.
    dim : int = -1
        The dimension to pad.
    **kwargs : float
        Keyword arguments forwarded to `pad_right_at_dim`. The accepted key is `value` (`float`,
        default `0.0`), the fill value for inserted elements.

    Returns
    -------
    padded : Tensor
        The padded `Tensor` with `padded.shape[dim] == max(t.shape[dim], length)`. Returns `t`
        unchanged when `t.shape[dim] >= length`.

    See Also
    --------
    pad_right_at_dim : Pad the right side by an explicit count.
    pad_left_at_dim_to : Pad the left side to a target length.

    Examples
    --------
    From dreamer4 [2], bringing variable-length action sequences to a uniform time length before
    batching:

        ```python
        tensors = [pad_right_at_dim_to(t, max_time, dim=dim) for t in tensors]
        ```

    References
    ----------
    [1] torch_einops_utils._padding.pad_right_at_dim

    [2] lucidrains/dreamer4
        https://github.com/lucidrains/dreamer4
    """
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
    pad_lens: bool = False,
) -> Tensor | list[Tensor] | tuple[Tensor | list[Tensor], Tensor] | None:
    """Pad a sequence of `Tensor` values to a shared length along `dim` and optionally stack them.

    You can use `pad_sequence` to align a heterogeneous-length sequence of `Tensor` values so that
    all `Tensor` values in `tensors` share the same size along `dim`. Each `Tensor` is padded to the
    maximum size of `dim` across `tensors` using `pad_left_at_dim` [1] (when `left=True`) or
    `pad_right_at_dim` [2] (when `left=False`). If `tensors` is empty, `pad_sequence` returns `None`.

    The return type depends on the combination of `return_stacked` and `return_lens`. When
    `return_stacked=True`, the padded list is combined into a single `Tensor` along `dim_stack` using
    `torch.stack` [3]. When `return_lens=True`, a second `Tensor` of original (or padding) lengths is
    appended to the return value as a tuple.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        A sequence of `Tensor` values to pad. Each `Tensor` must have the same shape in every
        dimension except `dim`.
    dim : int = -1
        The dimension along which lengths vary and padding is applied.
    value : float = 0.0
        The scalar fill value for inserted padding elements.
    left : bool = False
        When `True`, pad at the beginning of `dim` (left-pad). When `False`, pad at the end
        (right-pad).
    dim_stack : int = 0
        The dimension along which to stack the padded `Tensor` values. Used only when
        `return_stacked=True`.
    return_stacked : bool = True
        When `True`, return a single stacked `Tensor`. When `False`, return a `list` of individually
        padded `Tensor` values.
    return_lens : bool = False
        When `True`, include a 1-D `Tensor` of per-element lengths (or padding widths, if
        `pad_lens=True`) as the second element of a returned tuple.
    pad_lens : bool = False
        When `True` and `return_lens=True`, the returned lengths `Tensor` contains padding widths
        (`max_len - original_len`) rather than original dimension lengths.

    Returns
    -------
    output : Tensor | list[Tensor] | tuple[Tensor | list[Tensor], Tensor] | None
        When `return_stacked=True` and `return_lens=False`, a stacked `Tensor` or `None`. When
        `return_stacked=True` and `return_lens=True`, a tuple of a stacked `Tensor` and a 1-D lengths
        `Tensor`, or `None`. When `return_stacked=False` and `return_lens=False`, a `list` of padded
        `Tensor` values or `None`. When `return_stacked=False` and `return_lens=True`, a tuple of a
        `list` of padded `Tensor` values and a 1-D lengths `Tensor`, or `None`. Returns `None` if
        `tensors` is empty.

    See Also
    --------
    pad_sequence_and_cat : Pad and concatenate along a separate dimension.

    Examples
    --------
    From the test suite [4], padding three tensors of different sequence lengths and verifying the
    packed shape and recovered lengths:

        >>> import torch
        >>> from torch_einops_utils import pad_sequence, lens_to_mask
        >>> x, y, z = torch.randn(2, 4, 5), torch.randn(2, 3, 5), torch.randn(2, 1, 5)
        >>> packed, lens = pad_sequence([x, y, z], dim=1, return_lens=True)
        >>> packed.shape
        torch.Size([3, 2, 4, 5])
        >>> lens.tolist()
        [4, 3, 1]

    From sdft-pytorch [5], left-padding variable-length prompt token sequences and retrieving
    per-sample padding widths for start-position tracking:

        ```python
        student_prompt_ids, student_seq_start_pos = pad_sequence(student_prompt_ids, return_lens=True, left=True, pad_lens=True)
        ```

    From pi-zero-pytorch [6], right-padding a list of discrete action id tensors with a sentinel fill
    value of `-1`:

        ```python
        discrete_action_ids = pad_sequence([tensor(ids) for ids in discrete_action_ids], value=-1)
        ```

    References
    ----------
    [1] torch_einops_utils._padding.pad_left_at_dim

    [2] torch_einops_utils._padding.pad_right_at_dim

    [3] torch.stack - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.stack.html
    [4] tests/test_utils.py

    [5] lucidrains/sdft-pytorch
        https://github.com/lucidrains/sdft-pytorch
    [6] lucidrains/pi-zero-pytorch
        https://github.com/lucidrains/pi-zero-pytorch
    """
    output: Tensor | list[Tensor] | tuple[Tensor | list[Tensor], Tensor] | None = None

    if len(tensors) > 0:
        lens: list[int] | Tensor = [t.shape[dim] for t in tensors]
        max_len: int = max(lens)

        pad_fn: Callable[..., Tensor] = pad_left_at_dim if left else pad_right_at_dim
        padded_tensors: list[Tensor] = [pad_fn(t, max_len - t_len, dim=dim, value=value) for t, t_len in zip(tensors, lens, strict=True)]

        output = stack(padded_tensors, dim=dim_stack) if return_stacked else padded_tensors

        if return_lens:
            device: device = first(tensors).device
            lens = tensor(lens, device=device)

            if pad_lens:
                lens = max_len - lens

            output = (output, lens)

    return output


def pad_sequence_and_cat(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.0,
    left: bool = False,
    dim_cat: int = 0,
) -> Tensor | None:
    """Pad `tensors` to a shared length along `dim` and concatenate along `dim_cat`.

    You can use `pad_sequence_and_cat` to align and merge a heterogeneous-length sequence of `Tensor`
    values into a single `Tensor`. `pad_sequence_and_cat` first calls `pad_sequence` [1] with
    `return_stacked=False` to obtain a list of padded `Tensor` values, then concatenates the list
    along `dim_cat` using `torch.cat` [2]. If `tensors` is empty, `pad_sequence_and_cat` returns
    `None`.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        A sequence of `Tensor` values to pad and concatenate. Each `Tensor` must have the same shape
        in every dimension except `dim`.
    dim : int = -1
        The dimension along which lengths vary and padding is applied.
    value : float = 0.0
        The scalar fill value for inserted padding elements.
    left : bool = False
        When `True`, pad at the beginning of `dim`. When `False`, pad at the end.
    dim_cat : int = 0
        The dimension along which to concatenate the padded `Tensor` values.

    Returns
    -------
    concatenated : Tensor | None
        A single `Tensor` produced by concatenating all padded `Tensor` values along `dim_cat`.
        Returns `None` if `tensors` is empty.

    See Also
    --------
    pad_sequence : Pad and stack along a new or existing dimension.

    Examples
    --------
    From the test suite [3], padding images to a common height and concatenating along the batch
    dimension to produce a single batch `Tensor`:

    >>> import torch
    >>> from torch_einops_utils import pad_sequence, pad_sequence_and_cat
    >>> images = [torch.randn(3, 16, 17), torch.randn(3, 15, 18), torch.randn(3, 17, 16)]
    >>> padded_height = pad_sequence(images, dim=-2, return_stacked=False)
    >>> stacked = pad_sequence_and_cat(padded_height, dim_cat=0)
    >>> stacked.shape
    torch.Size([9, 17, 18])

    References
    ----------
    [1] torch_einops_utils._padding.pad_sequence

    [2] torch.cat - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.cat.html
    [3] tests/test_utils.py
    """

    padded: Tensor | list[Tensor] | None = pad_sequence(tensors, dim=dim, value=value, left=left, return_stacked=False, return_lens=False)
    if padded is not None:
        padded = cat(padded, dim=dim_cat)
    return padded
