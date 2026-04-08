from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from torch_einops_utils import exists


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
    shape: tuple[int, ...] = tuple(t.shape)
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
    [1] torch_einops_utils._dimensions.pad_ndim

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
    [1] torch_einops_utils._dimensions.pad_ndim

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
    [1] torch_einops_utils._dimensions.pad_right_ndim

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
    [1] torch_einops_utils._dimensions.pad_left_ndim

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
    [1] torch_einops_utils._dimensions.pad_right_ndim

    [2] torch.Tensor.reshape - PyTorch documentation
        https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html
    [3] einops - Context7
        https://context7.com/arogozhnikov/einops
    """
    if not exists(ndim):
        ndim = max([t.ndim for t in tensors])

    return tuple(pad_right_ndim(t, ndim - t.ndim) for t in tensors)
