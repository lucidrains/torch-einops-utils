from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from torch_einops_utils import exists


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
    [1] torch_einops_utils._slicing.slice_at_dim

    """
    if length == 0:
        return slice_at_dim(t, slice(0, 0), dim=dim)

    return slice_at_dim(t, slice(None, length), dim=dim)


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
    [2] torch_einops_utils._slicing.slice_at_dim

    """
    if length == 0:
        return slice_at_dim(t, slice(0, 0), dim=dim)

    return slice_at_dim(t, slice(-length, None), dim=dim)


def shape_with_replace(
    t: Tensor,
    replace_dict: Mapping[int, int] | None = None,
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
    replace_dict : Mapping[int, int] | None = None
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
    [3] torch_einops_utils._helpers.exists

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
