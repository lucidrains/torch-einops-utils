from __future__ import annotations

from functools import wraps

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Concatenate, Literal, TypeGuard, overload
from typing_extensions import Unpack
from torch.types import Number

import torch
from torch import tensor, is_tensor, cat, stack, arange, Tensor
import torch.nn.functional as F

from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map, PyTree

from einops import rearrange, pack, unpack

from torch_einops_utils import decreasing, DimAndValue, IdentityCallable, PSpec, RVar, SupportsIntIndex, T_co, TVar, zeroIndexed

# helper functions

def exists(v: TVar | None) -> TypeGuard[TVar]:
    """Test whether `v` is not `None`.

    You can use `exists` as a `None`-guard throughout this package. `exists` returns `True` for any
    value that is not `None`, including falsy values such as `0`, `False`, and empty collections. The
    return type is annotated as `TypeGuard[TVar]` [1] so that static analyzers narrow the type of `v`
    to `TVar` in branches guarded by `exists`.

    Parameters
    ----------
    v : TVar | None
        The value to test.

    Returns
    -------
    result : bool
        `True` when `v is not None`, otherwise `False`.

    See Also
    --------
    default : Return a fallback value when `v` is `None`.
    compact : Filter `None` values from an iterable.

    Examples
    --------
    From `torch_einops_utils.lens_to_mask` [2], guarding optional parameter `max_len` before use:

        ```python
        if not exists(max_len):
            max_len = lens.amax().item()
        ```

    References
    ----------
    [1] TypeGuard - Python typing documentation
        https://docs.python.org/3/library/typing.html#typing.TypeGuard
    [2] torch_einops_utils.lens_to_mask
    """
    return v is not None

def default(v: TVar | None, d: TVar) -> TVar:
    """Return `v` when `v` is not `None`, or `d` when `v` is `None`.

    You can use `default` to supply a fallback value for optional parameters and accumulators.
    `default` calls `exists` [1] to test `v`. When `exists(v)` is `True`, `default` returns `v`
    unchanged. When `v` is `None`, `default` returns `d`. Two overloads ensure the return type
    narrows correctly under static analysis.

    Parameters
    ----------
    v : TVar | None
        The primary value to test.
    d : TVar
        The fallback value returned when `v` is `None`.

    Returns
    -------
    result : TVar
        `v` when `v` is not `None`, otherwise `d`.

    See Also
    --------
    exists : Test whether a value is not `None`.

    References
    ----------
    [1] torch_einops_utils.exists
    """
    return v if exists(v) else d

def identity(t: TVar, *_args: object, **_kwargs: object) -> TVar:
    """Return `t` unchanged, ignoring all other arguments.

    You can use `identity` as a no-op callable in contexts that require a function but no
    transformation. `identity` accepts and discards any additional positional or keyword arguments,
    making `identity` a drop-in substitute for any unary or multi-argument callable.
    `torch_einops_utils.maybe` [1] returns `identity` when its `fn` argument is `None`.

    Parameters
    ----------
    t : TVar
        The value to return unchanged.
    *args : Any
        Additional positional arguments. Accepted and discarded.
    **kwargs : Any
        Additional keyword arguments. Accepted and discarded.

    Returns
    -------
    result : TVar
        `t`, the same object that was passed as the first argument.

    See Also
    --------
    maybe : Return `identity` when `fn` is `None`, otherwise wrap `fn` to skip `None` inputs.

    References
    ----------
    [1] torch_einops_utils.identity
    """
    return t

def first(arr: SupportsIntIndex[TVar]) -> TVar:
    """Return the element at index `0` of `arr`.

    You can use `first` to retrieve the first element of any sequence that supports integer indexing
    via `SupportsIntIndex` [1]. `first` delegates to `arr[0]`.

    Parameters
    ----------
    arr : SupportsIntIndex[TVar]
        A sequence that supports integer indexing. Access with index `0` must be valid.

    Returns
    -------
    element : TVar
        The element at position `0` in `arr`.

    References
    ----------
    [1] torch_einops_utils.SupportsIntIndex
    """
    return arr[0]

def compact(arr: Iterable[T_co | None]) -> list[T_co]:
    """Filter `None` values from `arr` and return the remaining elements as a `list`.

    You can use `compact` to remove all `None` values from an iterable, producing a typed `list` of
    non-`None` elements. `compact` applies `exists` [1] as the predicate for `filter` [2]. The `safe`
    [3] decorator uses `compact` to strip `None` values from a `Sequence` of `Tensor` before passing
    the result to the decorated function.

    Parameters
    ----------
    arr : Iterable[T_co | None]
        An iterable that may contain a mix of non-`None` values and `None` values to discard.

    Returns
    -------
    compacted : list[T_co]
        A `list` containing only the non-`None` elements of `arr`, in iteration order.

    See Also
    --------
    exists : Test whether a value is not `None`.
    safe : Decorator that applies `compact` to filter `None` values before calling the wrapped function.

    References
    ----------
    [1] torch_einops_utils.exists

    [2] filter - Python documentation
        https://docs.python.org/3/library/functions.html#filter
    [3] torch_einops_utils.safe
    """
    return [*filter(exists, arr)]

@overload
def maybe(fn: Callable[Concatenate[TVar, PSpec], RVar]) -> Callable[Concatenate[TVar | None, PSpec], RVar | None]: ...
@overload
def maybe(fn: None) -> IdentityCallable: ...
def maybe(
    fn: Callable[Concatenate[TVar, PSpec], RVar] | None,
) -> Callable[Concatenate[TVar | None, PSpec], RVar | None] | IdentityCallable:
    """Wrap `fn` so that the wrapped function returns `None` when the first argument is `None`.

    You can use this function to conditionally apply `fn` without adding an explicit `if`/`else`
    guard at every call site. The returned callable passes all positional and keyword arguments to
    `fn` unchanged when the first argument is not `None`, and returns `None` immediately when the
    first argument is `None`. When `fn` is `None`, the function returns `identity` [1], which passes
    its first argument through without modification.

    Parameters
    ----------
    fn : Callable[Concatenate[TVar, PSpec], RVar] | None
        The callable to wrap, or `None`. The first positional parameter of `fn` is the value that may
        be `None`. Pass `None` to receive an identity function.

    Returns
    -------
    wrapped : Callable[Concatenate[TVar | None, PSpec], RVar | None] | Callable[..., TVar]
        A wrapped version of `fn` that short-circuits to `None` when the first argument is `None`, or
        `identity` when `fn` is `None`.

    See Also
    --------
    identity : Return the first argument unchanged.

    Examples
    --------
    Skip the function when the first argument is `None` [2]:

        ```python
        from torch_einops_utils import maybe

        result = maybe(lambda t: t + 1)(None)
        # result is None
        ```

    Pass `None` as `fn` to receive an identity function [2]:

        ```python
        result = maybe(None)(1)
        # result == 1
        ```

    Conditionally convert episode lengths to a mask [3]:

        ```python
        from torch_einops_utils import maybe, lens_to_mask

        mask = maybe(lens_to_mask)(episode_lens, seq_len)
        # mask is None when episode_lens is None,
        # otherwise mask == lens_to_mask(episode_lens, seq_len)
        ```

    References
    ----------
    [1] torch_einops_utils.identity

    [2] tests.test_utils.test_maybe

    [3] metacontroller.metacontroller.ratio_loss
        https://context7.com/lucidrains/metacontroller
    """
    if not exists(fn):
        return identity

    @wraps(fn)
    def inner(t: TVar | None, *args: PSpec.args, **kwargs: PSpec.kwargs) -> RVar | None:
        if not exists(t):
            return None

        return fn(t, *args, **kwargs)

    return inner

def safe(
    fn: Callable[Concatenate[Sequence[Tensor], PSpec], Tensor | None],
) -> Callable[Concatenate[Sequence[Tensor | None], PSpec], Tensor | None]:
    """Wrap `fn` so that `None` values are filtered from the first argument before the call.

    You can use `safe` as a decorator to make a function that accepts a `Sequence[Tensor]` tolerate
    `None` values in the sequence. The decorated function accepts a `Sequence[Tensor | None]`. Before
    calling `fn`, `safe` compacts [1] the sequence to remove all `None` values. When the compacted
    sequence is empty, `safe` returns `None` without calling `fn`. When at least one non-`None`
    `Tensor` remains, `safe` passes the compacted sequence to `fn`.

    `safe` is applied as a decorator to `safe_stack` [2] and `safe_cat` [3] to produce null-safe
    stacking and concatenation. `safe` is also applied to `reduce_masks` [4] to produce null-safe
    mask reduction.

    Parameters
    ----------
    fn : Callable[Concatenate[Sequence[Tensor], PSpec], Tensor | None]
        A callable whose first argument is a `Sequence[Tensor]` with at least one element. `fn` must
        handle a compacted sequence of any length ≥ 1.

    Returns
    -------
    wrapped : Callable[Concatenate[Sequence[Tensor | None], PSpec], Tensor | None]
        A wrapped version of `fn` that accepts `None` values in the first argument and returns `None`
        when no non-`None` `Tensor` values are present.

    See Also
    --------
    compact : Filter `None` values from an iterable.
    safe_cat : Concatenate tensors along an existing dimension, skipping `None` values.
    safe_stack : Stack tensors along a new dimension, skipping `None` values.

    References
    ----------
    [1] torch_einops_utils.compact

    [2] torch_einops_utils.safe_stack

    [3] torch_einops_utils.safe_cat

    [4] torch_einops_utils.reduce_masks
    """

    @wraps(fn)
    def inner(tensors: Sequence[Tensor | None], *args: PSpec.args, **kwargs: PSpec.kwargs) -> Tensor | None:
        safe_tensors: list[Tensor] = compact(tensors)
        if len(safe_tensors) == 0:
            return None
        return fn(safe_tensors, *args, **kwargs)

    return inner

# exported functions

def masked_mean(
    t: Tensor,
    mask: Tensor | None = None,
    dim: int | None = None,
    eps: float = 1e-5,
) -> Tensor:
    """Compute the mean of `t` over positions selected by `mask`.

    You can use this function to average only the elements of `t` where `mask` is `True`, ignoring
    masked-out positions. When `mask` is `None`, the function falls back to the standard
    `torch.Tensor.mean` [1]. When `mask` has fewer dimensions than `t`, the function right-pads
    `mask` with singleton dimensions using `pad_right_ndim` [2] before broadcasting. When all
    positions in `mask` are `False` and `dim` is `None`, the function returns zero by summing over
    the empty selection.

    Parameters
    ----------
    t : Tensor
        The input tensor to be averaged.
    mask : Tensor | None = None
        A boolean tensor selecting which positions contribute to the mean. When `mask` has fewer
        dimensions than `t`, singleton dimensions are appended on the right before broadcasting. Pass
        `None` to compute an unmasked mean.
    dim : int | None = None
        The dimension along which to compute the mean. Pass `None` to reduce over all dimensions.
    eps : float = 1e-5
        A small value added to the denominator to prevent division by zero when computing the masked
        mean along a dimension.

    Returns
    -------
    result : Tensor
        The masked mean of `t`. The shape matches `t` with the reduced dimension removed when `dim`
        is specified, or a scalar tensor when `dim` is `None`.

    See Also
    --------
    pad_right_ndim : Pad singleton dimensions on the right of a tensor to reach a target number of dimensions.

    Examples
    --------
    Compute the mean of all elements with no mask [3]:

        ```python
        from torch import tensor
        from torch_einops_utils import masked_mean

        t = tensor([1.0, 2.0, 3.0, 4.0])
        result = masked_mean(t)
        # result == tensor(2.5)
        ```

    Select only the `True` positions using a boolean mask [3]:

        ```python
        mask = tensor([True, False, True, False])
        result = masked_mean(t, mask=mask)
        # result == tensor(2.0)
        ```

    Average along a specific dimension [3]:

        ```python
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        mask = tensor([[True, False], [True, True]])
        result = masked_mean(t, mask=mask, dim=1)
        # result == tensor([1.0, 3.5])
        ```

    References
    ----------
    [1] torch.Tensor.mean - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.Tensor.mean.html
    [2] torch_einops_utils.pad_right_ndim

    [3] tests.test_utils.test_masked_mean

    """
    if not exists(mask):
        return t.mean(dim = dim) if exists(dim) else t.mean()

    if mask.ndim < t.ndim:
        mask = pad_right_ndim(mask, t.ndim - mask.ndim)

    mask = mask.expand_as(t)

    if not exists(dim):
        return t[mask].mean() if mask.any() else t[mask].sum()

    num: Tensor = (t * mask).sum(dim = dim)
    den: Tensor = mask.sum(dim = dim)

    return num / den.clamp(min = eps)

# shapes

def shape_with_replace(
    t: Tensor,
    replace_dict: dict[int, int] | None = None
) -> torch.Size:
    """Return the shape of a tensor with selected dimension sizes replaced by new values.

    You can use this function to compute a target shape derived from an existing tensor, substituting
    one or more dimension sizes without modifying the tensor itself. A common use case is
    constructing a zero-padding tensor whose shape matches a source tensor except along the time or
    sequence dimension, as in `dreamer4` [1] and `mimic-video` [2].

    When `replace_dict` is `None` or empty, this function returns `t.shape` unchanged.

    Parameters
    ----------
    t : Tensor
        The reference tensor whose shape is used as the base.
    replace_dict : dict[int, int] | None = None
        A mapping from dimension index to the replacement size for that dimension. All keys must be
        non-negative integers less than `t.ndim`. Negative dimension indices are not supported; pass
        a non-negative index instead.

    Returns
    -------
    shape : torch.Size
        The shape of `t` with each dimension listed in `replace_dict` substituted with the
        corresponding value.

    Raises
    ------
    ValueError
        Raised when any key in `replace_dict` is greater than or equal to `t.ndim`.

    See Also
    --------
    slice_at_dim : Apply an arbitrary slice to one dimension.

    Examples
    --------
    From `dreamer4.trainers` [1] and the test suite:

        ```python
        import torch
        from torch_einops_utils import shape_with_replace

        t = torch.randn(3, 4, 5)

        # Replace the size of dimension 1 with 2
        assert shape_with_replace(t, {1: 2}) == (3, 2, 5)

        # Build a zero-padding tensor matching a video tensor's shape
        # except along the time dimension (dim 2) — from dreamer4.trainers
        pad_shape = shape_with_replace(generated_video, {2: real_len - gen_len})
        padding = generated_video.new_zeros(pad_shape)
        generated_video = torch.cat((generated_video, padding), dim=2)

        # Allocate future-frame noise with the latent shape — from mimic-video
        pred_shape = shape_with_replace(latents, {2: predict_num_future_latents})
        future_noise = torch.randn(pred_shape, device=latents.device)
        ```

    References
    ----------
    [1] dreamer4.trainers.BehaviorCloneTrainer
        https://github.com/lucidrains/dreamer4
    [2] mimic_video.cosmos_predict.CosmosPredictWrapper
        https://github.com/lucidrains/mimic-video
    [3] torch_einops_utils.exists

    """
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

# slicing

def slice_at_dim(t: Tensor, slc: slice, dim: int = -1) -> Tensor:
    """Apply a slice to a single dimension of a tensor while preserving all other dimensions.

    You can use this function to extract a contiguous or strided sub-range along one axis without
    affecting any other axis. The function normalizes negative `dim` values before constructing the
    full index tuple, so both positive and negative dimension indices produce the same result.

    Parameters
    ----------
    t : Tensor
        The input tensor to slice.
    slc : slice
        The slice object describing the range to extract along `dim`. Any valid Python `slice` is
        accepted, including open-ended slices such as `slice(None, length)` or `slice(-length,
        None)`.
    dim : int = -1
        The dimension along which to apply `slc`. Negative values are converted to their positive
        equivalents before indexing.

    Returns
    -------
    sliced : Tensor
        A tensor with the same number of dimensions as `t`, where the size of `dim` equals the length
        selected by `slc` and all other dimensions are unchanged.

    See Also
    --------
    slice_left_at_dim : Select a prefix of a given length along one dimension.
    slice_right_at_dim : Select a suffix of a given length along one dimension.

    Examples
    --------
    From the test suite and external usage in `alphafold3_pytorch` [1] and `rotary_embedding_torch`
    [2]:

        ```python
        import torch
        from torch_einops_utils import slice_at_dim

        t = torch.randn(3, 4, 5)

        # Slice the last dimension (default dim=-1)
        res = slice_at_dim(t, slice(1, 3))
        assert res.shape == (3, 4, 2)

        # Slice the first two elements of dimension 1
        res = slice_at_dim(t, slice(None, 2), dim=1)
        assert res.shape == (3, 2, 5)

        # Slice from position 2 onward along dim -2
        res = slice_at_dim(t, slice(2, None), dim=-2)
        assert res.shape == (3, 2, 5)

        # Trim positional frequencies to match a query length (rotary embeddings)
        freqs = slice_at_dim(freqs, slice(-seq_len, None), dim=freqs_seq_dim)

        # Shift-and-concatenate for windowed attention (alphafold3)
        left = slice_at_dim(t, slice(None, -1), dim=dim_seq)
        right = slice_at_dim(t, slice(1, None), dim=dim_seq)
        ```

    References
    ----------
    [1] alphafold3_pytorch.attention.slice_at_dim
        https://github.com/lucidrains/alphafold3-pytorch
    [2] rotary_embedding_torch.rotary_embedding_torch.slice_at_dim
        https://github.com/lucidrains/rotary-embedding-torch
    """
    dims: int = t.ndim
    dim = (dim + dims) if dim < 0 else dim

    full_slice: list[slice] = [slice(None)] * dims
    full_slice[dim] = slc

    return t[tuple(full_slice)]

def slice_left_at_dim(t: Tensor, length: int, dim: int = -1) -> Tensor:
    """Select a prefix of a given length from one dimension of a tensor.

    You can use this function to keep only the first `length` elements along `dim`, discarding the
    remainder. When `length` is zero, the function returns an empty tensor with size zero along `dim`
    rather than the full tensor.

    Parameters
    ----------
    t : Tensor
        The input tensor to slice.
    length : int
        The number of elements to retain from the start of `dim`. When `length` is zero, the returned
        tensor has size zero along `dim`.
    dim : int = -1
        The dimension along which to take the prefix.

    Returns
    -------
    sliced : Tensor
        A tensor whose size along `dim` is `min(length, t.shape[dim])`, with all other dimensions
        unchanged.

    See Also
    --------
    slice_at_dim : Apply an arbitrary slice to one dimension.
    slice_right_at_dim : Select a suffix of a given length along one dimension.

    Examples
    --------
    From the test suite:

        ```python
        import torch
        from torch_einops_utils import slice_left_at_dim

        t = torch.randn(3, 4, 5)

        res = slice_left_at_dim(t, 2, dim=1)
        assert res.shape == (3, 2, 5)
        ```

    References
    ----------
    [1] torch_einops_utils.slice_at_dim

    """
    if length == 0:
        return slice_at_dim(t, slice(0, 0), dim = dim)

    return slice_at_dim(t, slice(None, length), dim = dim)

def slice_right_at_dim(t: Tensor, length: int, dim: int = -1) -> Tensor:
    """Select a suffix of a given length from one dimension of a tensor.

    You can use this function to keep only the last `length` elements along `dim`, discarding the
    earlier elements. A common use case is aligning positional frequency tensors to a shorter query
    length during inference, as in `PoPE` [1]. When `length` is zero, the function returns an empty
    tensor with size zero along `dim` rather than the full tensor.

    Parameters
    ----------
    t : Tensor
        The input tensor to slice.
    length : int
        The number of elements to retain from the end of `dim`. When `length` is zero, the returned
        tensor has size zero along `dim`.
    dim : int = -1
        The dimension along which to take the suffix.

    Returns
    -------
    sliced : Tensor
        A tensor whose size along `dim` is `min(length, t.shape[dim])`, with all other dimensions
        unchanged.

    See Also
    --------
    slice_at_dim : Apply an arbitrary slice to one dimension.
    slice_left_at_dim : Select a prefix of a given length along one dimension.

    Examples
    --------
    From the test suite and `PoPE_pytorch` [1]:

        ```python
        import torch
        from torch_einops_utils import slice_right_at_dim

        t = torch.randn(3, 4, 5)

        # Keep the last two elements of dimension 1
        res = slice_right_at_dim(t, 2, dim=1)
        assert res.shape == (3, 2, 5)

        # Trim precomputed positional frequencies to the query length (PoPE)
        freqs = slice_right_at_dim(freqs, q_len, dim=-2)
        ```

    References
    ----------
    [1] PoPE_pytorch.pope.apply_pope_to_qk
        https://github.com/lucidrains/PoPE-pytorch
    [2] torch_einops_utils.slice_at_dim

    """
    if length == 0:
        return slice_at_dim(t, slice(0, 0), dim = dim)

    return slice_at_dim(t, slice(-length, None), dim = dim)

# dimensions

def pad_ndim(t: Tensor, ndims: tuple[int, int]) -> Tensor:
    """Reshape a tensor by inserting singleton dimensions on both sides.

    You can use this function to insert leading and trailing singleton dimensions into a tensor,
    increasing its rank for broadcasting with higher-rank tensors. Set `ndims` to `(left, right)` to
    control how many singleton dimensions appear before and after the existing shape. The function
    raises `ValueError` if either count is negative.

    Parameters
    ----------
    t : Tensor
        The input tensor to reshape.
    ndims : tuple[int, int]
        The count of singleton dimensions to add on each side. The first element specifies dimensions
        prepended to the shape; the second element specifies dimensions appended. Both values must be
        ≥ 0.

    Returns
    -------
    padded : Tensor
        A view of `t` with shape `(1,) * left + t.shape + (1,) * right`, where `left, right = ndims`.

    Raises
    ------
    ValueError
        When either element of `ndims` is negative.

    See Also
    --------
    pad_left_ndim : Insert singleton dimensions only at the leading side.
    pad_right_ndim : Insert singleton dimensions only at the trailing side.
    pad_left_ndim_to : Pad the leading side until the tensor has a target rank.
    pad_right_ndim_to : Pad the trailing side until the tensor has a target rank.

    torch
    -----
    This function uses `torch.Tensor.reshape` [1] to produce a view with the requested singleton
    axes. In einops pattern notation [2], adding `left=1` leading and `right=1` trailing dimensions
    transforms a pattern `'b n d'` to `'1 b n d 1'`, without copying data.

    References
    ----------
    [1] torch.Tensor.reshape - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
    [2] einops - Context7
        https://context7.com/arogozhnikov/einops
    """
    shape: tuple[int, ...] = t.shape
    left, right = ndims
    if left < 0 or right < 0:
        message: str = f"I received `{left = }` and `{right = }`, but I need both values to be greater than or equal to `0`."
        raise ValueError(message)

    ones: tuple[int] = (1,)
    ones_left: tuple[int, ...] = ones * left
    ones_right: tuple[int, ...] = ones * right
    return t.reshape(*ones_left, *shape, *ones_right)

def pad_left_ndim(t: Tensor, ndims: int) -> Tensor:
    """Reshape a tensor by inserting singleton dimensions at the leading side.

    You can use this function to prepend `ndims` singleton dimensions to a tensor, increasing its
    rank without copying data.

    Parameters
    ----------
    t : Tensor
        The input tensor to reshape.
    ndims : int
        The number of singleton dimensions to prepend to the tensor's shape. Must be ≥ 0.

    Returns
    -------
    padded : Tensor
        A view of `t` with `ndims` leading singleton dimensions added.

    See Also
    --------
    pad_ndim : Insert singleton dimensions on both sides of the shape.
    pad_right_ndim : Insert singleton dimensions at the trailing side only.
    pad_left_ndim_to : Pad the leading side until the tensor reaches a target rank.

    torch
    -----
    This function delegates to `pad_ndim` [1], which uses `torch.Tensor.reshape` [2] to produce a
    view. In einops pattern notation [3], prepending two singleton dimensions transforms a pattern
    `'b n d'` to `'1 1 b n d'`, making the tensor suitable for broadcasting against higher-rank
    tensors without copying data.

    References
    ----------
    [1] torch_einops_utils.pad_ndim

    [2] torch.Tensor.reshape - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
    [3] einops - Context7
        https://context7.com/arogozhnikov/einops
    """
    return pad_ndim(t, (ndims, 0))

def pad_right_ndim(t: Tensor, ndims: int) -> Tensor:
    """Reshape a tensor by inserting singleton dimensions at the trailing side.

    You can use this function to append `ndims` singleton dimensions to a tensor, increasing its rank
    without copying data.

    Parameters
    ----------
    t : Tensor
        The input tensor to reshape.
    ndims : int
        The number of singleton dimensions to append to the tensor's shape. Must be ≥ 0.

    Returns
    -------
    padded : Tensor
        A view of `t` with `ndims` trailing singleton dimensions added.

    See Also
    --------
    pad_ndim : Insert singleton dimensions on both sides of the shape.
    pad_left_ndim : Insert singleton dimensions at the leading side only.
    pad_right_ndim_to : Pad the trailing side until the tensor reaches a target rank.
    align_dims_left : Pad all tensors in a sequence to the same rank.

    torch
    -----
    This function delegates to `pad_ndim` [1], which uses `torch.Tensor.reshape` [2] to produce a
    view. In einops pattern notation [3], appending two singleton dimensions transforms a pattern `'b
    n d'` to `'b n d 1 1'`, enabling scalar or lower-rank values to broadcast element-wise against a
    higher-rank tensor.

    References
    ----------
    [1] torch_einops_utils.pad_ndim

    [2] torch.Tensor.reshape - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
    [3] einops - Context7
        https://context7.com/arogozhnikov/einops
    """
    return pad_ndim(t, (0, ndims))

def pad_right_ndim_to(t: Tensor, ndims: int) -> Tensor:
    """Reshape a tensor by padding trailing singleton dimensions up to a target rank.

    You can use this function to ensure a tensor has at least `ndims` dimensions by appending
    singleton dimensions at the trailing side. The function returns `t` unchanged when `t.ndim` is
    already ≥ `ndims`.

    Parameters
    ----------
    t : Tensor
        The input tensor to reshape.
    ndims : int
        The target total number of dimensions. Dimensions are added only when `t.ndim < ndims`.

    Returns
    -------
    padded : Tensor
        A view of `t` with at least `ndims` dimensions.

    See Also
    --------
    pad_right_ndim : Append an exact count of trailing singleton dimensions.
    pad_left_ndim_to : Pad the leading side up to a target rank.
    align_dims_left : Pad all tensors in a sequence to the same rank.

    torch
    -----
    This function uses `pad_right_ndim` [1], which calls `torch.Tensor.reshape` [2]. In einops
    pattern notation [3], the operation appends as many `1` axes as needed to align the tensor rank
    before element-wise or `einsum` [3] operations with a higher-rank tensor.

    Examples
    --------
    Broadcast a scalar time value against a video tensor of shape `(b, c, t, h, w)`
    for flow interpolation:

        ```python
        from torch_einops_utils import pad_right_ndim_to

        # dreamer4: align time '(b,)' with video '(b, c, t, h, w)'
        padded_time = pad_right_ndim_to(time[None], video.ndim)
        pred_flow = (pred_video - video) / (1.0 - padded_time)
        ```

    Scale a flow prediction using a denominator with lower rank than the prediction:

        ```python
        from torch_einops_utils import pad_right_ndim_to

        # mimic_video: convert model output to flow space
        pred_flow = (pred - actions) / pad_right_ndim_to(1.0 - action_time, pred.ndim).clamp_min(eps)
        ```

    References
    ----------
    [1] torch_einops_utils.pad_right_ndim

    [2] torch.Tensor.reshape - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
    [3] einops - Context7
        https://context7.com/arogozhnikov/einops
    """
    if t.ndim >= ndims:
        return t

    return pad_right_ndim(t, ndims - t.ndim)

def pad_left_ndim_to(t: Tensor, ndims: int) -> Tensor:
    """Reshape a tensor by padding leading singleton dimensions up to a target rank.

    You can use this function to ensure a tensor has at least `ndims` dimensions by prepending
    singleton dimensions at the leading side. The function returns `t` unchanged when `t.ndim` is
    already ≥ `ndims`.

    Parameters
    ----------
    t : Tensor
        The input tensor to reshape.
    ndims : int
        The target total number of dimensions. Dimensions are added only when `t.ndim < ndims`.

    Returns
    -------
    padded : Tensor
        A view of `t` with at least `ndims` dimensions.

    See Also
    --------
    pad_left_ndim : Prepend an exact count of leading singleton dimensions.
    pad_right_ndim_to : Pad the trailing side up to a target rank.
    align_dims_left : Pad all tensors in a sequence to the same rank.

    torch
    -----
    This function uses `pad_left_ndim` [1], which calls `torch.Tensor.reshape` [2]. In einops pattern
    notation [3], the operation prepends as many `1` axes as needed to align the tensor rank, placing
    existing dimensions on the right so they correspond to the innermost axes of a higher-rank
    reference tensor.

    References
    ----------
    [1] torch_einops_utils.pad_left_ndim

    [2] torch.Tensor.reshape - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
    [3] einops - Context7
        https://context7.com/arogozhnikov/einops
    """
    if t.ndim >= ndims:
        return t

    return pad_left_ndim(t, ndims - t.ndim)

def align_dims_left(
    tensors: Sequence[Tensor],
    *,
    ndim: int | None = None,
) -> tuple[Tensor, ...]:
    """Pad all tensors in a sequence with trailing singleton dimensions to a common rank.

    You can use this function to align a heterogeneous sequence of tensors to the same number of
    dimensions, enabling broadcasting in element-wise operations over tensors with different ranks.
    When `ndim` is `None`, the target rank is the maximum rank across all input tensors.

    Parameters
    ----------
    tensors : Sequence[Tensor]
        The tensors to align. Each tensor may have a different rank.
    ndim : int | None = None
        The target number of dimensions. When `None`, the largest rank among the input tensors is
        used as the target.

    Returns
    -------
    aligned : tuple[Tensor, ...]
        A tuple of tensor views, each having exactly `ndim` dimensions, with trailing singleton
        dimensions appended as needed.

    See Also
    --------
    pad_right_ndim : Append an exact count of trailing singleton dimensions.
    pad_right_ndim_to : Pad the trailing side of a single tensor up to a target rank.

    torch
    -----
    This function applies `pad_right_ndim` [1] to each tensor using `torch.Tensor.reshape` [2]. In
    einops usage [3], aligning tensor ranks before an `einsum` [3] or element-wise multiply ensures a
    scalar weight of shape `'b'` and a per-token loss of shape `'b n d'` broadcast correctly without
    explicit `rearrange` calls.

    Examples
    --------
    Align a PPO advantage tensor `(b, n)` with a log-probability ratio tensor `(b, n, d)` for
    element-wise multiplication:

        ```python
        from torch_einops_utils import align_dims_left

        # metacontroller: align ratio and advantages before the PPO surrogate loss
        ratio, advantages = align_dims_left((ratio, advantages))
        surr1 = ratio * advantages
        ```

    Align a noise schedule `(b,)` with a latent tensor `(b, n, d)` for linear interpolation:

        ```python
        from torch_einops_utils import align_dims_left

        # dreamer4: align time with latents before noising
        aligned_times, _ = align_dims_left((times, latents))
        noised_latents = noise.lerp(latents, aligned_times)
        ```

    Align a 1-D time value with an action tensor before flow-matching noise interpolation:

        ```python
        from torch_einops_utils import align_dims_left

        # mimic_video: align time with actions for noise interpolation
        actions, left_aligned_time = align_dims_left((actions, time))
        noised = noise.lerp(actions, left_aligned_time)
        ```

    References
    ----------
    [1] torch_einops_utils.pad_right_ndim

    [2] torch.Tensor.reshape - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
    [3] einops - Context7
        https://context7.com/arogozhnikov/einops
    """
    if not exists(ndim):
        ndim = max([t.ndim for t in tensors])

    return tuple(pad_right_ndim(t, ndim - t.ndim) for t in tensors)

# cat and stack

@safe
def safe_stack(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
    """Stack tensors from `tensors` along a new dimension, skipping `None` values.

    You can use `safe_stack` to stack a mixed sequence of `Tensor` and `None` values. The `safe` [1]
    decorator filters out `None` values from `tensors` before passing the remaining `Tensor` values
    to `torch.stack` [2]. If `tensors` contains no non-`None` values, `safe_stack` returns `None`.

    Parameters
    ----------
    tensors : Sequence[Tensor | None]
        A `Sequence` of `Tensor` or `None` values. `None` values are filtered out before stacking.
        All non-`None` `Tensor` values must have the same shape.
    dim : int = 0
        The dimension along which to stack. The result has one more dimension than each input
        `Tensor`.

    Returns
    -------
    stacked : Tensor | None
        The stacked `Tensor`, or `None` if `tensors` contains no non-`None` values.

    See Also
    --------
    safe_cat : Concatenate tensors along an existing dimension, skipping `None` values.

    Examples
    --------
    From the test suite [3]:

        >>> import torch
        >>> from torch_einops_utils import safe_stack
        >>> t1, t2 = torch.randn(2, 3), torch.randn(2, 3)
        >>> safe_stack([]) is None
        True
        >>> safe_stack([None]) is None
        True
        >>> safe_stack([t1]).shape
        torch.Size([1, 2, 3])
        >>> safe_stack([t1, None]).shape  # None is skipped; only t1 is stacked
        torch.Size([1, 2, 3])
        >>> safe_stack([t1, t2]).shape
        torch.Size([2, 2, 3])

    From dreamer4 [4], collecting optional per-layer intermediates where some layers
    may not produce output (returning `None`):

        ```python
            intermediates = TransformerIntermediates(
                stack(time_attn_kv_caches),
                safe_stack(normed_time_attn_inputs),
                safe_stack(normed_space_attn_inputs),
                safe_stack(rnn_hiddens),
                hiddens,
        )

    References
    ----------
    [1] torch_einops_utils.safe

    [2] torch.stack - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.stack.html
    [3] tests/test_utils.py

    [4] lucidrains/dreamer4
        https://github.com/lucidrains/dreamer4
    """
    return stack(tensors, dim = dim)  # pyright: ignore[reportArgumentType] https://github.com/pytorch/pytorch/issues/179391

@safe
def safe_cat(tensors: Sequence[Tensor], dim: int = 0) -> Tensor | None:
    """Concatenate tensors from `tensors` along an existing dimension, skipping `None` values.

    You can use `safe_cat` to concatenate a mixed sequence of `Tensor` and `None` values. The `safe`
    [1] decorator filters out `None` values from `tensors` before passing the remaining `Tensor`
    values to `torch.cat` [2]. If `tensors` contains no non-`None` values, `safe_cat` returns `None`.

    A common pattern is iterative accumulation where the accumulator starts as `None`. On the first
    iteration, `safe_cat` receives one non-`None` `Tensor` and returns it unchanged. On subsequent
    iterations, `safe_cat` concatenates the accumulator with the new `Tensor`.

    Parameters
    ----------
    tensors : Sequence[Tensor | None]
        A sequence of `Tensor` or `None` values. `None` values are filtered out before concatenation.
        All non-`None` `Tensor` values must have the same shape in every dimension except `dim`.
    dim : int = 0
        The dimension along which to concatenate.

    Returns
    -------
    concatenated : Tensor | None
        The concatenated `Tensor`, or `None` if `tensors` contains no non-`None` values.

    See Also
    --------
    safe_stack : Stack tensors along a new dimension, skipping `None` values.

    Examples
    --------
    From the test suite [3]:

        >>> import torch
        >>> from torch_einops_utils import safe_cat
        >>> t1, t2 = torch.randn(2, 3), torch.randn(2, 3)
        >>> safe_cat([]) is None
        True
        >>> safe_cat([None]) is None
        True
        >>> safe_cat([t1, None]).shape  # None is skipped; only t1 is returned
        torch.Size([2, 3])
        >>> safe_cat([t1, t2]).shape
        torch.Size([4, 3])

    From sdft_pytorch [4], accumulating per-step token losses across a generation loop where
    `token_kl_div_losses` is initialized to `None` before the loop:

        ```python
        token_kl_div_losses = safe_cat((token_kl_div_losses, token_kl_div), dim=1)
        ```

    References
    ----------
    [1] torch_einops_utils.safe

    [2] torch.cat - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.cat.html
    [3] tests/test_utils.py

    [4] lucidrains/sdft-pytorch
        https://github.com/lucidrains/sdft-pytorch
    """
    return cat(tensors, dim = dim)  # pyright: ignore[reportUnknownVariableType, reportCallIssue, reportArgumentType] https://github.com/pytorch/pytorch/issues/179391

# masking

def lens_to_mask(lens: Tensor, max_len: Number | None = None) -> Tensor:
    """Convert a sequence of length values into a boolean mask `Tensor`.

    You can use `lens_to_mask` to create sequence masks from integer length values. For each scalar
    in `lens`, `lens_to_mask` produces a row of `True` values for positions less than that length
    value and `False` values for all positions equal to or greater than it. The output `Tensor` has
    one more dimension than `lens`, appended at the last axis, with length `max_len`.

    Parameters
    ----------
    lens : Tensor
        A `Tensor` of non-negative integers representing sequence lengths. `lens` may have any shape;
        the output shape is `(*lens.shape, max_len)`.
    max_len : int | None = None
        The size of the last dimension of the output `Tensor`. If `None`, `max_len` is set to
        `int(lens.amax().item())`.

    Returns
    -------
    mask : Tensor
        A boolean `Tensor` of shape `(*lens.shape, max_len)`. Position `i` along the last axis is
        `True` if `i < lens[...]` for the corresponding element of `lens`.

    Examples
    --------
    From the test suite [1], verifying that `lens_to_mask` sets exactly `length` leading `True`
    values per row:

        >>> import torch
        >>> from torch_einops_utils import lens_to_mask
        >>> lens = torch.tensor([4, 3, 1])
        >>> mask = lens_to_mask(lens)
        >>> mask.shape
        torch.Size([3, 4])
        >>> (mask.sum(dim=-1) == lens).all()
        tensor(True)

    Passing an explicit `max_len` produces a wider mask than the maximum length in `lens`:

        >>> lens_to_mask(lens, max_len=6).shape
        torch.Size([3, 6])

    From dreamer4 [2], masking padded time steps in variable-length rollouts:

        ```python
        mask_for_gae = lens_to_mask(experience.lens, time)
        ```

    References
    ----------
    [1] tests/test_masking.py

    [2] lucidrains/dreamer4
        https://github.com/lucidrains/dreamer4
    """
    device: torch.device = lens.device

    if not exists(max_len):
        max_len = lens.amax().item()

    seq: Tensor = arange(max_len, device = device)
    lens = rearrange(lens, '... -> ... 1')
    return seq < lens

@safe
def reduce_masks(masks: Sequence[Tensor], op: Callable[[Tensor, Tensor], Tensor]) -> Tensor | None:
    """Reduce a sequence of boolean mask `Tensor` values to a single mask using a binary operator.

    You can use `reduce_masks` to apply any binary element-wise callable reduction over a sequence of
    boolean masks. The `safe` [1] decorator filters out `None` values from `masks` before `op` is
    applied. If `masks` contains no non-`None` values, `reduce_masks` returns `None`. Reduction
    proceeds left-to-right over the non-`None` elements of `masks`.

    Parameters
    ----------
    masks : Sequence[Tensor | None]
        A `Sequence` of boolean `Tensor` or `None` values. `None` values are filtered out before `op`
        is applied. All non-`None` `Tensor` values must have the same shape.
    op : Callable[[Tensor, Tensor], Tensor]
        A binary callable that accepts two `Tensor` arguments and returns a `Tensor`. Common choices
        are `torch.logical_and` [2] and `torch.logical_or` [3].

    Returns
    -------
    mask : Tensor | None
        The result of applying `op` cumulatively, left-to-right, over the non-`None` members of
        `masks`. Returns `None` if no non-`None` values remain after filtering.

    See Also
    --------
    and_masks : Reduce masks using element-wise logical AND.
    or_masks : Reduce masks using element-wise logical OR.

    Examples
    --------
    From the test suite [4]:

    >>> import torch
    >>> from torch_einops_utils import reduce_masks
    >>> mask1 = torch.tensor([True, True])
    >>> mask2 = torch.tensor([True, False])
    >>> reduce_masks([None, None], torch.logical_and) is None
    True
    >>> reduce_masks([mask1, None, mask2], torch.logical_and)
    tensor([ True, False])
    >>> reduce_masks([mask1, None, mask2], torch.logical_or)
    tensor([ True,  True])

    References
    ----------
    [1] torch_einops_utils.safe

    [2] torch.logical_and - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.logical_and.html
    [3] torch.logical_or - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.logical_or.html
    [4] tests/test_masking.py

    """
    mask, *rest_masks = masks

    for rest_mask in rest_masks:
        mask: Tensor = op(mask, rest_mask)

    return mask

def and_masks(masks: Sequence[Tensor | None]) -> Tensor | None:
    """Reduce a `Sequence` of boolean mask `Tensor` values to a single mask using element-wise logical AND.

    You can use `and_masks` to combine multiple boolean masks so that the result is `True` only where
    all non-`None` input masks are `True`. `and_masks` calls `reduce_masks` [1] with
    `torch.logical_and` [2]. `None` values in `masks` are filtered out before reduction. If all
    values in `masks` are `None`, `and_masks` returns `None`.

    Parameters
    ----------
    masks : Sequence[Tensor | None]
        A `Sequence` of boolean `Tensor` or `None` values. `None` values are treated as absent and
        filtered out before reduction. All non-`None` `Tensor` values must have the same shape.

    Returns
    -------
    mask : Tensor | None
        A boolean `Tensor` that is `True` only at positions where every non-`None` input mask
        is `True`. Returns `None` if `masks` contains no non-`None` values.

    See Also
    --------
    or_masks : Reduce masks using element-wise logical OR.
    reduce_masks : Reduce masks using a caller-supplied binary operator.

    Examples
    --------
    From the test suite [3]:

        >>> from torch import tensor
        >>> from torch_einops_utils import and_masks
        >>> and_masks([None]) is None
        True
        >>> mask1 = tensor([True, True])
        >>> mask2 = tensor([True, False])
        >>> and_masks([mask1, None, mask2])
        tensor([ True, False])

    From sdft_pytorch [4], intersecting an end-of-sequence mask with an initial-token mask to exclude
    padding and masked prefix positions from the loss calculation:

        ```python
        mask = and_masks([eos_mask, init_tokens_mask])
        ```

    References
    ----------
    [1] torch_einops_utils.reduce_masks

    [2] torch.logical_and - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.logical_and.html
    [3] tests/test_utils.py

    [4] lucidrains/sdft-pytorch
        https://github.com/lucidrains/sdft-pytorch
    """
    return reduce_masks(masks, torch.logical_and)

def or_masks(masks: Sequence[Tensor | None]) -> Tensor | None:
    """Reduce a sequence of boolean mask `Tensor` values to a single mask using element-wise logical OR.

    You can use `or_masks` to combine multiple boolean masks so that the result is `True` wherever at
    least one non-`None` input mask is `True`. `or_masks` calls `reduce_masks` [1] with
    `torch.logical_or` [2]. `None` values in `masks` are filtered out before reduction. If all values
    in `masks` are `None`, `or_masks` returns `None`.

    Parameters
    ----------
    masks : Sequence[Tensor | None]
        A sequence of boolean `Tensor` or `None` values. `None` values are treated as absent and
        filtered out before reduction. All non-`None` `Tensor` values must have the same shape.

    Returns
    -------
    mask : Tensor | None
        A boolean `Tensor` that is `True` at any position where at least one non-`None` input mask is
        `True`. Returns `None` if `masks` contains no non-`None` values.

    See Also
    --------
    and_masks : Reduce masks using element-wise logical AND.
    reduce_masks : Reduce masks using a caller-supplied binary operator.

    References
    ----------
    [1] torch_einops_utils.reduce_masks

    [2] torch.logical_or - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.logical_or.html
    """
    return reduce_masks(masks, torch.logical_or)

# padding

def pad_at_dim(
    t: Tensor,
    pad: tuple[int, int],
    *,
    dim: int = -1,
    value: float = 0.
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

    [2] tests.test_padding.py

    [3] lucidrains/dreamer4
        https://github.com/lucidrains/dreamer4

    [4] lucidrains/locoformer
        https://github.com/lucidrains/locoformer
    """
    dims_from_right: int = ((decreasing * dim) - zeroIndexed + t.ndim) % t.ndim
    zeros: tuple[Literal[0], ...] = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), value = value)

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
    [1] torch_einops_utils.pad_at_dim

    [2] torch_einops_utils.DimAndValue

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
    [1] torch_einops_utils.pad_at_dim

    [2] torch_einops_utils.DimAndValue

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
    [1] torch_einops_utils.pad_left_at_dim

    [2] tests/test_utils.py
    """
    curr_len: int = t.shape[dim]
    if curr_len >= length:
        return t

    return pad_left_at_dim(t, length - curr_len, dim = dim, **kwargs)

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
    [1] torch_einops_utils.pad_right_at_dim

    [2] lucidrains/dreamer4
        https://github.com/lucidrains/dreamer4
    """
    curr_len: int = t.shape[dim]
    if curr_len >= length:
        return t

    return pad_right_at_dim(t, length - curr_len, dim = dim, **kwargs)

# better pad sequence

@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[True] = True,
    return_lens: Literal[False] = False,
    pad_lens: bool = False
) -> Tensor | None: ...
@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[True] = True,
    return_lens: Literal[True],
    pad_lens: bool = False
) -> tuple[Tensor, Tensor] | None: ...
@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[False],
    return_lens: Literal[False] = False,
    pad_lens: bool = False
) -> list[Tensor] | None: ...
@overload
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: Literal[False],
    return_lens: Literal[True],
    pad_lens: bool = False
) -> tuple[list[Tensor], Tensor] | None: ...
def pad_sequence(
    tensors: Sequence[Tensor],
    *,
    dim: int = -1,
    value: float = 0.,
    left: bool = False,
    dim_stack: int = 0,
    return_stacked: bool = True,
    return_lens: bool = False,
    pad_lens: bool = False
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
    [1] torch_einops_utils.pad_left_at_dim

    [2] torch_einops_utils.pad_right_at_dim

    [3] torch.stack - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.stack.html
    [4] tests/test_utils.py

    [5] lucidrains/sdft-pytorch
        https://github.com/lucidrains/sdft-pytorch
    [6] lucidrains/pi-zero-pytorch
        https://github.com/lucidrains/pi-zero-pytorch
    """
    output: Tensor | list[Tensor] | tuple[Tensor | list[Tensor], Tensor] | None = None

    if 0 < len(tensors):
        lens: list[int] | Tensor = [t.shape[dim] for t in tensors]
        max_len: int = max(lens)

        pad_fn: Callable[..., Tensor] = pad_left_at_dim if left else pad_right_at_dim
        padded_tensors: list[Tensor] = [pad_fn(t, max_len - t_len, dim = dim, value = value) for t, t_len in zip(tensors, lens, strict=True)]

        if return_stacked:
            output = stack(padded_tensors, dim = dim_stack)
        else:
            output = padded_tensors

        if return_lens:
            device: torch.device = first(tensors).device
            lens = tensor(lens, device=device)

            if pad_lens:
                lens = max_len - lens

            output = (output, lens)

    return output

def pad_sequence_and_cat(
    tensors: Sequence[Tensor],
    *,
    dim_cat: int = 0,
    dim: int = -1,
    value: float = 0.,
    left: bool = False
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
    [1] torch_einops_utils.pad_sequence

    [2] torch.cat - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.cat.html
    [3] tests/test_utils.py
    """

    padded: Tensor | list[Tensor] | None = pad_sequence(tensors, dim = dim, value = value, left = left, return_stacked = False, return_lens = False)
    if padded is not None:
        return cat(padded, dim = dim_cat)
    return padded

# tree flatten with inverse

def tree_map_tensor(fn: Callable[[Tensor], Tensor], tree: PyTree) -> PyTree:
    """Apply `fn` to every `torch.Tensor` leaf in `tree`, leaving non-tensor leaves unchanged.

    You can use this function to transform only the tensor leaves of a PyTree [1] structure without
    disturbing the non-tensor leaves. The function wraps `fn` with an identity pass-through for
    non-tensor values and delegates structural traversal to `torch.utils._pytree.tree_map` [2].

    Parameters
    ----------
    fn : Callable[[Tensor], Tensor]
        A function to apply to each `torch.Tensor` leaf in `tree`.
    tree : PyTree
        A nested Python structure such as a tuple, list, or dictionary containing a mix of
        `torch.Tensor` values and other Python objects.

    Returns
    -------
    mappedTree : PyTree
        A PyTree with the same structure as `tree`, where each `torch.Tensor` leaf has been replaced
        by the result of `fn(leaf)` and all non-tensor leaves are unchanged.

    See Also
    --------
    tree_flatten_with_inverse : Flatten a PyTree into a list and return an inverse function.

    Examples
    --------
    Increment only the tensor leaf while preserving non-tensor leaves [3]:

        ```python
        from torch import tensor
        from torch_einops_utils import tree_map_tensor

        tree = (1, tensor(2), 3)
        result = tree_map_tensor(lambda t: t + 1, tree)
        # result[0] == 1
        # result[1] == tensor(3)
        # result[2] == 3
        ```

    Detach all tensors nested inside a state container [4]:

        ```python
        from torch_einops_utils import tree_map_tensor

        nextMemory = tree_map_tensor(lambda t: t.detach(), nextMemory)
        ```

    References
    ----------
    [1] PyTree - PyTorch documentation
        https://pytorch.org/docs/stable/pytree.html
    [2] torch.utils._pytree.tree_map - PyTorch documentation
        https://pytorch.org/docs/stable/pytree.html#torch.utils._pytree.tree_map
    [3] tests.test_utils.test_tree_map_tensor

    [4] fast_weight_attention.chunk_manager.ChunkManager.forward
        https://context7.com/lucidrains/fast-weight-attention
    """
    def func(t: object) -> object:
        if is_tensor(t):
            return fn(t)
        return t

    return tree_map(func, tree)

def tree_flatten_with_inverse(tree: PyTree) -> tuple[list[Any], Callable[[Iterable[Any]], PyTree]]:
    """Flatten `tree` into a list of leaves and return a paired inverse function.

    You can use this function to decompose a nested PyTree [1] structure into a flat list of leaves
    and to recover the original nested structure from a modified list. The paired inverse function
    calls `torch.utils._pytree.tree_unflatten` [2] with the `TreeSpec` captured at flatten time, so
    the structure can be reconstructed even after the leaves have been modified.

    Parameters
    ----------
    tree : PyTree
        A nested Python structure such as a tuple, list, or dictionary to flatten.

    Returns
    -------
    flattened : list[Any]
        A flat list of all leaves in `tree` in left-to-right traversal order.
    inverse : Callable[[Iterable[Any]], PyTree]
        A function that accepts an iterable of leaves and reconstructs a PyTree with the same
        structure as the original `tree`.

    See Also
    --------
    tree_map_tensor : Apply a function to every tensor leaf in a PyTree.

    Examples
    --------
    Modify a single leaf and reconstruct the original nested structure [3]:

        ```python
        from torch_einops_utils import tree_flatten_with_inverse

        tree = (1, (2, 3), 4)
        (first, *rest), inverse = tree_flatten_with_inverse(tree)
        result = inverse((first + 1, *rest))
        # result == (2, (2, 3), 4)
        ```

    References
    ----------
    [1] PyTree - PyTorch documentation
        https://pytorch.org/docs/stable/pytree.html
    [2] torch.utils._pytree.tree_unflatten - PyTorch documentation
        https://pytorch.org/docs/stable/pytree.html#torch.utils._pytree.tree_unflatten
    [3] tests.test_utils.test_tree_flatten_with_inverse

    """
    flattened, spec = tree_flatten(tree)

    def inverse(out: Iterable[Any]) -> PyTree:
        return tree_unflatten(out, spec)

    return flattened, inverse

# einops pack

@overload
def pack_with_inverse(t: Tensor, pattern: str) -> tuple[Tensor, Callable[[Tensor, str | None], Tensor]]: ...
@overload
def pack_with_inverse(t: list[Tensor], pattern: str) -> tuple[Tensor, Callable[[Tensor, str | None], list[Tensor]]]: ...
def pack_with_inverse(t: Tensor | list[Tensor], pattern: str) -> tuple[Tensor, Callable[[Tensor, str | None], Tensor | list[Tensor]]]:
    """Pack `t` with `pattern` using einops and return a paired inverse unpacking function.

    You can use this function to merge one or more tensors into a single packed tensor using an
    einops `pack` pattern [1] and to later restore the original shapes. When `t` is a single
    `torch.Tensor`, the function wraps `t` in a list before packing and unwraps the result inside the
    inverse function. When `t` is a list of tensors, the inverse function returns a list of tensors.
    The inverse function accepts an optional `inv_pattern` argument to override the pattern used for
    unpacking; when `inv_pattern` is `None`, the original `pattern` is reused.

    Parameters
    ----------
    t : Tensor | list[Tensor]
        A single tensor or a list of tensors to pack.
    pattern : str
        An einops pack pattern string such as `'b * d'`, where `*` collects the packed dimensions.

    Returns
    -------
    packed : Tensor
        The packed tensor produced by `einops.pack` [1].
    inverse : Callable[[Tensor, str | None], Tensor | list[Tensor]]
        A function that accepts the packed (or transformed) tensor and an optional override pattern
        and returns the unpacked tensor or list of tensors.

    See Also
    --------
    tree_flatten_with_inverse : Flatten a PyTree and return an inverse reconstruction function.

    Examples
    --------
    Pack a single tensor and recover the original shape [2]:

        ```python
        import torch
        from torch_einops_utils import pack_with_inverse

        t = torch.randn(3, 12, 2, 2)
        packed, inverse = pack_with_inverse(t, "b * d")
        # packed.shape == (3, 24, 2)
        recovered = inverse(packed)
        # recovered.shape == (3, 12, 2, 2)
        ```

    Pack a list of tensors and unpack with an overriding pattern [2]:

        ```python
        t = torch.randn(3, 12, 2)
        u = torch.randn(3, 4, 2)
        packed, inverse = pack_with_inverse([t, u], "b * d")
        # packed.shape == (3, 28, 2)

        reduced = packed.sum(dim=-1)
        t_out, u_out = inverse(reduced, "b *")
        # t_out.shape == (3, 12)
        # u_out.shape == (3, 4)
        ```

    References
    ----------
    [1] einops.pack - einops documentation
        https://einops.rocks/api/pack/
    [2] tests.test_utils.test_pack_with_inverse

    """
    is_one: bool = is_tensor(t)

    if is_one:
        t = [t]

    packed, packed_shape = pack(t, pattern)

    def inverse(out: Tensor, inv_pattern: str | None = None) -> Tensor | list[Tensor]:
        inv_pattern = default(inv_pattern, pattern)
        unpacked: list[Tensor] = unpack(out, packed_shape, inv_pattern)

        if is_one:
            return first(unpacked)

        return unpacked

    return packed, inverse
