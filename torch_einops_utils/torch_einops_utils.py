"""Provide PyTree manipulation and masked tensor utilities for torch and einops workflows.

You can use this module to compute masked means, apply functions to tensor leaves in PyTree
structures, flatten and reconstruct nested data, and pack tensors with recoverable inverses.

Contents
--------
Functions
    masked_mean
        Compute the mean of a tensor over positions selected by a boolean mask.
    pack_with_inverse
        Pack a tensor or list of tensors with an einops pattern and return a paired inverse.
    tree_flatten_with_inverse
        Flatten a PyTree into a list of leaves and return a paired inverse function.
    tree_map_tensor
        Apply a function to every tensor leaf in a PyTree, leaving non-tensor leaves unchanged.
"""

# ruff: noqa: PLC0414
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, overload

from torch import Tensor, is_tensor
from torch.utils._pytree import PyTree, tree_flatten, tree_map, tree_unflatten

from einops import pack, unpack
from torch_einops_utils import (
    align_dims_left as align_dims_left,
    and_masks as and_masks,
    default,
    exists,
    first,
    lens_to_mask as lens_to_mask,
    maybe as maybe,
    or_masks as or_masks,
    pad_at_dim as pad_at_dim,
    pad_left_at_dim as pad_left_at_dim,
    pad_left_at_dim_to as pad_left_at_dim_to,
    pad_left_ndim as pad_left_ndim,
    pad_left_ndim_to as pad_left_ndim_to,
    pad_ndim as pad_ndim,
    pad_right_at_dim as pad_right_at_dim,
    pad_right_at_dim_to as pad_right_at_dim_to,
    pad_right_ndim as pad_right_ndim,
    pad_right_ndim_to as pad_right_ndim_to,
    pad_sequence as pad_sequence,
    pad_sequence_and_cat as pad_sequence_and_cat,
    safe_cat as safe_cat,
    safe_stack as safe_stack,
    shape_with_replace as shape_with_replace,
    slice_at_dim as slice_at_dim,
    slice_left_at_dim as slice_left_at_dim,
    slice_right_at_dim as slice_right_at_dim
)


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
    [2] torch_einops_utils._padding.pad_right_ndim

    [3] tests.test_utils.test_masked_mean

    """
    if not exists(mask):
        return t.mean(dim=dim) if exists(dim) else t.mean()

    if mask.ndim < t.ndim:
        mask = pad_right_ndim(mask, t.ndim - mask.ndim)

    mask = mask.expand_as(t)

    if not exists(dim):
        return t[mask].mean() if mask.any() else t[mask].sum()

    num: Tensor = (t * mask).sum(dim=dim)
    den: Tensor = mask.sum(dim=dim)

    return num / den.clamp(min=eps)


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

    """

    def func(t: Any) -> Any:  # noqa: ANN401
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
        unpack_pattern = default(inv_pattern, pattern)
        unpacked: list[Tensor] = unpack(out, packed_shape, unpack_pattern)

        if is_one:
            return first(unpacked)

        return unpacked

    return packed, inverse
