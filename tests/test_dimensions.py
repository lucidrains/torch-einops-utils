from __future__ import annotations

import torch
from torch import Tensor

from torch_einops_utils import (
    align_dims_left,
    pad_left_ndim,
    pad_left_ndim_to,
    pad_ndim,
    pad_right_ndim,
    pad_right_ndim_to
)

import pytest


@pytest.mark.parametrize("ndim", [pytest.param(None, id="infer-ndim"), pytest.param(1, id="ndim-one"), pytest.param(3, id="ndim-three")])
def test_align_dims_left(t: Tensor, ndim: int | None) -> None:
    tensor_sequence = (t,)

    if ndim is not None and ndim < t.ndim:
        with pytest.raises(ValueError, match="greater than or equal to `0`"):
            align_dims_left(tensor_sequence, ndim=ndim)
        return

    aligned = align_dims_left(tensor_sequence, ndim=ndim)
    expected_ndim = t.ndim if ndim is None else max(ndim, t.ndim)
    expected_shape = (*t.shape, *(1,) * (expected_ndim - t.ndim))

    assert len(aligned) == 1, f"align_dims_left returned {len(aligned)} tensors, expected 1 for {ndim=} and {tuple(t.shape)=}."
    assert tuple(aligned[0].shape) == expected_shape, (
        f"align_dims_left returned shape {tuple(aligned[0].shape)}, expected {expected_shape} for {ndim=} and {tuple(t.shape)=}."
    )
    assert torch.equal(aligned[0].reshape(t.shape), t), (
        f"align_dims_left changed tensor values for {ndim=} and {tuple(t.shape)=}."
    )


@pytest.mark.parametrize(
    ("explicit_ndim", "expected_length", "expected_exception"),
    [pytest.param(None, None, ValueError, id="empty-inferred-ndim"), pytest.param(3, 0, None, id="empty-explicit-ndim")],
)
def test_align_dims_left_empty_sequence(
    explicit_ndim: int | None, expected_length: int | None, expected_exception: type[Exception] | None
) -> None:
    empty: tuple[Tensor, ...] = ()

    if expected_exception is not None:
        with pytest.raises(expected_exception):
            align_dims_left(empty, ndim=explicit_ndim)
    else:
        result = align_dims_left(empty, ndim=explicit_ndim)
        assert len(result) == expected_length, (
            f"align_dims_left returned {len(result)} tensors for empty input, expected {expected_length} with {explicit_ndim=}."
        )


@pytest.mark.parametrize(
    ("tensors", "explicit_ndim"),
    [
        pytest.param(
            (torch.tensor([127.0, 131.0]), torch.tensor([[137.0, 139.0], [149.0, 151.0]])),
            1,
            id="explicit-ndim-smaller-than-matrix-rank",
        ),
    ],
)
def test_align_dims_left_raises_when_explicit_ndim_too_small(tensors: tuple[Tensor, ...], explicit_ndim: int) -> None:
    with pytest.raises(ValueError, match="greater than or equal to `0`"):
        align_dims_left(tensors, ndim=explicit_ndim)


@pytest.mark.parametrize("ndims", [pytest.param(2, id="two-dims"), pytest.param(3, id="three-dims")])
def test_pad_left_ndim(t: Tensor, ndims: int) -> None:
    expected_shape = (*(1,) * ndims, *t.shape)
    padded = pad_left_ndim(t, ndims)

    assert tuple(padded.shape) == expected_shape, (
        f"pad_left_ndim returned shape {tuple(padded.shape)}, expected {expected_shape} for {ndims=} and {tuple(t.shape)=}."
    )
    assert torch.equal(padded.reshape(t.shape), t), (
        f"pad_left_ndim changed tensor values for {ndims=} and {tuple(t.shape)=}."
    )


@pytest.mark.parametrize(
    "target_ndim_delta",
    [
        pytest.param(-1, id="target-below-current"),
        pytest.param(0, id="target-equals-current"),
        pytest.param(2, id="target-exceeds-by-two"),
        pytest.param(5, id="target-exceeds-by-five"),
    ],
)
def test_pad_left_ndim_to(t: Tensor, target_ndim_delta: int) -> None:
    target_ndim = t.ndim + target_ndim_delta
    added = max(0, target_ndim_delta)
    expected_shape = (*(1,) * added, *t.shape)
    result = pad_left_ndim_to(t, target_ndim)

    assert tuple(result.shape) == expected_shape, (
        f"pad_left_ndim_to returned shape {tuple(result.shape)}, expected {expected_shape} "
        f"for {target_ndim=} and {tuple(t.shape)=}."
    )
    assert torch.equal(result.reshape(t.shape), t), (
        f"pad_left_ndim_to changed tensor values for {target_ndim=} and {tuple(t.shape)=}."
    )


@pytest.mark.parametrize(
    ("left_padding", "right_padding"),
    [pytest.param(0, 0, id="no-padding"), pytest.param(2, 1, id="two-left-one-right"), pytest.param(1, 3, id="one-left-three-right")],
)
def test_pad_ndim(t: Tensor, left_padding: int, right_padding: int) -> None:
    expected_shape = (*(1,) * left_padding, *t.shape, *(1,) * right_padding)
    padded = pad_ndim(t, (left_padding, right_padding))

    assert tuple(padded.shape) == expected_shape, (
        f"pad_ndim returned shape {tuple(padded.shape)}, expected {expected_shape} "
        f"for {left_padding=}, {right_padding=}, and {tuple(t.shape)=}."
    )
    assert torch.equal(padded.reshape(t.shape), t), (
        f"pad_ndim changed tensor values for {left_padding=}, {right_padding=}, and {tuple(t.shape)=}."
    )


def test_pad_ndim_raises_for_negative_padding(tensor_malformed_padding: tuple[Tensor, int, int]) -> None:
    tensor_value, left_padding, right_padding = tensor_malformed_padding
    with pytest.raises(ValueError, match="greater than or equal to `0`"):
        pad_ndim(tensor_value, (left_padding, right_padding))


@pytest.mark.parametrize("ndims", [pytest.param(2, id="two-dims"), pytest.param(3, id="three-dims")])
def test_pad_right_ndim(t: Tensor, ndims: int) -> None:
    expected_shape = (*t.shape, *(1,) * ndims)
    padded = pad_right_ndim(t, ndims)

    assert tuple(padded.shape) == expected_shape, (
        f"pad_right_ndim returned shape {tuple(padded.shape)}, expected {expected_shape} for {ndims=} and {tuple(t.shape)=}."
    )
    assert torch.equal(padded.reshape(t.shape), t), (
        f"pad_right_ndim changed tensor values for {ndims=} and {tuple(t.shape)=}."
    )


@pytest.mark.parametrize(
    "target_ndim_delta",
    [
        pytest.param(-1, id="target-below-current"),
        pytest.param(0, id="target-equals-current"),
        pytest.param(2, id="target-exceeds-by-two"),
        pytest.param(5, id="target-exceeds-by-five"),
    ],
)
def test_pad_right_ndim_to(t: Tensor, target_ndim_delta: int) -> None:
    target_ndim = t.ndim + target_ndim_delta
    added = max(0, target_ndim_delta)
    expected_shape = (*t.shape, *(1,) * added)
    result = pad_right_ndim_to(t, target_ndim)

    assert tuple(result.shape) == expected_shape, (
        f"pad_right_ndim_to returned shape {tuple(result.shape)}, expected {expected_shape} "
        f"for {target_ndim=} and {tuple(t.shape)=}."
    )
    assert torch.equal(result.reshape(t.shape), t), (
        f"pad_right_ndim_to changed tensor values for {target_ndim=} and {tuple(t.shape)=}."
    )
