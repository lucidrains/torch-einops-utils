from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor

from torch_einops_utils import (
    shape_with_replace,
    slice_at_dim,
    slice_left_at_dim,
    slice_right_at_dim
)

import pytest


@pytest.mark.parametrize(
    ("slc", "dim"),
    [
        pytest.param(slice(None, 2), -1, id="prefix-two-last-dim"),
        pytest.param(slice(1, None), -1, id="skip-first-last-dim"),
        pytest.param(slice(None), -1, id="full-last-dim"),
        pytest.param(slice(None, 1), 0, id="prefix-one-first-dim"),
    ],
)
def test_slice_at_dim(t: Tensor, slc: slice, dim: int) -> None:
    result = slice_at_dim(t, slc, dim=dim)

    normalized_dim = dim if dim >= 0 else t.ndim + dim
    expected_size = len(range(*slc.indices(t.shape[normalized_dim])))

    assert result.ndim == t.ndim, f"slice_at_dim changed ndim from {t.ndim} to {result.ndim} for {slc=}, {dim=}, shape={tuple(t.shape)}."
    assert result.shape[normalized_dim] == expected_size, (
        f"slice_at_dim returned size {result.shape[normalized_dim]} at dim {dim}, "
        f"expected {expected_size} for {slc=}, shape={tuple(t.shape)}."
    )
    for d in range(t.ndim):
        if d != normalized_dim:
            assert result.shape[d] == t.shape[d], (
                f"slice_at_dim changed shape at non-sliced dim {d} from {t.shape[d]} to {result.shape[d]} "
                f"for {slc=}, {dim=}, shape={tuple(t.shape)}."
            )

    full_index = [slice(None)] * t.ndim
    full_index[normalized_dim] = slc
    assert torch.equal(result, t[tuple(full_index)]), f"slice_at_dim returned incorrect values for {slc=}, {dim=}, shape={tuple(t.shape)}."


@pytest.mark.parametrize(
    ("length", "dim"),
    [
        pytest.param(0, -1, id="length-zero-last-dim"),
        pytest.param(2, -1, id="length-two-last-dim"),
        pytest.param(1, 0, id="length-one-first-dim"),
    ],
)
def test_slice_left_at_dim(t: Tensor, length: int, dim: int) -> None:
    result = slice_left_at_dim(t, length, dim=dim)

    normalized_dim = dim if dim >= 0 else t.ndim + dim
    expected_size = 0 if length == 0 else min(length, t.shape[normalized_dim])

    assert result.ndim == t.ndim, (
        f"slice_left_at_dim changed ndim from {t.ndim} to {result.ndim} for {length=}, {dim=}, shape={tuple(t.shape)}."
    )
    assert result.shape[normalized_dim] == expected_size, (
        f"slice_left_at_dim returned size {result.shape[normalized_dim]} at dim {dim}, "
        f"expected {expected_size} for {length=}, shape={tuple(t.shape)}."
    )
    for d in range(t.ndim):
        if d != normalized_dim:
            assert result.shape[d] == t.shape[d], (
                f"slice_left_at_dim changed shape at non-sliced dim {d} from {t.shape[d]} to {result.shape[d]} "
                f"for {length=}, {dim=}, shape={tuple(t.shape)}."
            )

    corresponding_slice = slice(0, 0) if length == 0 else slice(None, length)
    full_index = [slice(None)] * t.ndim
    full_index[normalized_dim] = corresponding_slice
    assert torch.equal(result, t[tuple(full_index)]), (
        f"slice_left_at_dim returned incorrect values for {length=}, {dim=}, shape={tuple(t.shape)}."
    )


@pytest.mark.parametrize(
    ("length", "dim"),
    [
        pytest.param(0, -1, id="length-zero-last-dim"),
        pytest.param(2, -1, id="length-two-last-dim"),
        pytest.param(1, 0, id="length-one-first-dim"),
    ],
)
def test_slice_right_at_dim(t: Tensor, length: int, dim: int) -> None:
    result = slice_right_at_dim(t, length, dim=dim)

    normalized_dim = dim if dim >= 0 else t.ndim + dim
    expected_size = 0 if length == 0 else min(length, t.shape[normalized_dim])

    assert result.ndim == t.ndim, (
        f"slice_right_at_dim changed ndim from {t.ndim} to {result.ndim} for {length=}, {dim=}, shape={tuple(t.shape)}."
    )
    assert result.shape[normalized_dim] == expected_size, (
        f"slice_right_at_dim returned size {result.shape[normalized_dim]} at dim {dim}, "
        f"expected {expected_size} for {length=}, shape={tuple(t.shape)}."
    )
    for d in range(t.ndim):
        if d != normalized_dim:
            assert result.shape[d] == t.shape[d], (
                f"slice_right_at_dim changed shape at non-sliced dim {d} from {t.shape[d]} to {result.shape[d]} "
                f"for {length=}, {dim=}, shape={tuple(t.shape)}."
            )

    corresponding_slice = slice(0, 0) if length == 0 else slice(-length, None)
    full_index = [slice(None)] * t.ndim
    full_index[normalized_dim] = corresponding_slice
    assert torch.equal(result, t[tuple(full_index)]), (
        f"slice_right_at_dim returned incorrect values for {length=}, {dim=}, shape={tuple(t.shape)}."
    )


@pytest.mark.parametrize(
    ("replace_dict", "trigger_out_of_range"),
    [
        pytest.param(None, False, id="none-replace-dict"),
        pytest.param({0: 17}, False, id="replace-first-dim"),
        pytest.param(None, True, id="out-of-range-index"),
    ],
)
def test_shape_with_replace(t: Tensor, replace_dict: Mapping[int, int] | None, trigger_out_of_range: bool) -> None:
    if trigger_out_of_range:
        with pytest.raises(ValueError, match="less than"):
            shape_with_replace(t, {t.ndim: 17})
        return

    result = shape_with_replace(t, replace_dict)  # pyright: ignore[reportArgumentType]

    if replace_dict is None:
        assert result == t.shape, (
            f"shape_with_replace returned {result}, expected {t.shape} for None replace_dict and shape={tuple(t.shape)}."
        )
    else:
        replaced_dims = {idx if idx >= 0 else t.ndim + idx for idx in replace_dict}
        for index, value in replace_dict.items():
            normalized = index if index >= 0 else t.ndim + index
            assert result[normalized] == value, (
                f"shape_with_replace returned {result[normalized]} at dim {index}, "
                f"expected {value} for {replace_dict=} and shape={tuple(t.shape)}."
            )
        for d in range(t.ndim):
            if d not in replaced_dims:
                assert result[d] == t.shape[d], (
                    f"shape_with_replace changed dim {d} from {t.shape[d]} to {result[d]} for {replace_dict=} and shape={tuple(t.shape)}."
                )
