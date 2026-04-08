from __future__ import annotations

import torch
from torch import Tensor

from torch_einops_utils import safe_cat, safe_stack

import pytest


def test_safe_stack(
    sequence_tensors: list[Tensor],
    empty_optional_tensor_sequence: list[Tensor | None],
) -> None:
    list_tensors = sequence_tensors

    empty_result = safe_stack(empty_optional_tensor_sequence)
    assert empty_result is None, "safe_stack must return None for empty input tensors."

    all_none_sequence: list[Tensor | None] = [None for _ in list_tensors]
    all_none_result = safe_stack(all_none_sequence)
    assert all_none_result is None, "safe_stack must return None when every entry is None."

    for tensor_index, tensor_value in enumerate(list_tensors):
        optional_sequence: list[Tensor | None] = [None, tensor_value, None]

        for stack_dimension in [0, -1]:
            expected = torch.stack([tensor_value], dim=stack_dimension)
            result = safe_stack(optional_sequence, dim=stack_dimension)

            assert result is not None, f"safe_stack returned None for single active tensor with {tensor_index=}, {stack_dimension=}."
            assert torch.equal(result, expected), (
                f"safe_stack returned incorrect values for single active tensor with {tensor_index=}, {stack_dimension=}."
            )

    grouped_tensors: dict[tuple[torch.dtype, tuple[int, ...]], list[Tensor]] = {}
    for tensor_value in list_tensors:
        grouping_key = (tensor_value.dtype, tuple(tensor_value.shape))
        grouped_tensors.setdefault(grouping_key, []).append(tensor_value)

    repeated_group_count = 0
    for grouping_key, grouped_tensor_values in grouped_tensors.items():
        if len(grouped_tensor_values) < 2:  # noqa: PLR2004
            continue

        repeated_group_count += 1
        optional_sequence: list[Tensor | None] = [grouped_tensor_values[0], None, *grouped_tensor_values[1:]]

        for stack_dimension in [0, -1]:
            expected = torch.stack(grouped_tensor_values, dim=stack_dimension)
            result = safe_stack(optional_sequence, dim=stack_dimension)

            assert result is not None, f"safe_stack returned None for repeated-shape group {grouping_key} with {stack_dimension=}."
            assert torch.equal(result, expected), (
                f"safe_stack returned incorrect values for repeated-shape group {grouping_key} with {stack_dimension=}."
            )

    assert repeated_group_count > 0, "safe_stack test requires at least one repeated shape group for multi-tensor checks."

    mismatched_pair: tuple[Tensor, Tensor] | None = None
    for left_tensor in list_tensors:
        for right_tensor in list_tensors:
            if left_tensor is right_tensor:
                continue
            if (
                left_tensor.ndim == right_tensor.ndim
                and left_tensor.dtype == right_tensor.dtype
                and tuple(left_tensor.shape) != tuple(right_tensor.shape)
            ):
                mismatched_pair = (left_tensor, right_tensor)
                break
        if mismatched_pair is not None:
            break

    assert mismatched_pair is not None, "safe_stack mismatch subcheck could not find tensors with same dtype/ndim and different shapes."

    with pytest.raises(RuntimeError):
        safe_stack([mismatched_pair[0], mismatched_pair[1]], dim=0)


def test_safe_cat(
    sequence_tensors: list[Tensor],
    empty_optional_tensor_sequence: list[Tensor | None],
) -> None:
    list_tensors = sequence_tensors

    empty_result = safe_cat(empty_optional_tensor_sequence)
    assert empty_result is None, "safe_cat must return None for empty input tensors."

    all_none_sequence: list[Tensor | None] = [None for _ in list_tensors]
    all_none_result = safe_cat(all_none_sequence)
    assert all_none_result is None, "safe_cat must return None when every entry is None."

    for tensor_index, tensor_value in enumerate(list_tensors):
        optional_sequence: list[Tensor | None] = [None, tensor_value, None]

        for cat_dimension in [0, -1]:
            expected = torch.cat([tensor_value], dim=cat_dimension)
            result = safe_cat(optional_sequence, dim=cat_dimension)

            assert result is not None, f"safe_cat returned None for single active tensor with {tensor_index=}, {cat_dimension=}."
            assert torch.equal(result, expected), (
                f"safe_cat returned incorrect values for single active tensor with {tensor_index=}, {cat_dimension=}."
            )

    grouped_tensors: dict[tuple[torch.dtype, tuple[int, ...]], list[Tensor]] = {}
    for tensor_value in list_tensors:
        grouping_key = (tensor_value.dtype, tuple(tensor_value.shape))
        grouped_tensors.setdefault(grouping_key, []).append(tensor_value)

    repeated_group_count = 0
    for grouping_key, grouped_tensor_values in grouped_tensors.items():
        if len(grouped_tensor_values) < 2:  # noqa: PLR2004
            continue

        repeated_group_count += 1
        optional_sequence: list[Tensor | None] = [grouped_tensor_values[0], None, *grouped_tensor_values[1:]]

        for cat_dimension in [0, -1]:
            expected = torch.cat(grouped_tensor_values, dim=cat_dimension)
            result = safe_cat(optional_sequence, dim=cat_dimension)

            assert result is not None, f"safe_cat returned None for repeated-shape group {grouping_key} with {cat_dimension=}."
            assert torch.equal(result, expected), (
                f"safe_cat returned incorrect values for repeated-shape group {grouping_key} with {cat_dimension=}."
            )

    assert repeated_group_count > 0, "safe_cat test requires at least one repeated shape group for multi-tensor checks."

    mismatched_pair_for_last_dimension: tuple[Tensor, Tensor] | None = None
    for left_tensor in list_tensors:
        for right_tensor in list_tensors:
            if left_tensor is right_tensor:
                continue
            if left_tensor.ndim <= 1 or left_tensor.ndim != right_tensor.ndim or left_tensor.dtype != right_tensor.dtype:
                continue
            if tuple(left_tensor.shape[:-1]) != tuple(right_tensor.shape[:-1]):
                mismatched_pair_for_last_dimension = (left_tensor, right_tensor)
                break
        if mismatched_pair_for_last_dimension is not None:
            break

    assert mismatched_pair_for_last_dimension is not None, (
        "safe_cat mismatch subcheck could not find same-dtype tensors with non-matching non-cat dimensions for dim=-1."
    )

    with pytest.raises(RuntimeError):
        safe_cat([mismatched_pair_for_last_dimension[0], mismatched_pair_for_last_dimension[1]], dim=-1)
