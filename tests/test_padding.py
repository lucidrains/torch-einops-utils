from __future__ import annotations

import torch
from torch import Tensor

from torch_einops_utils import (
    pad_at_dim,
    pad_left_at_dim,
    pad_left_at_dim_to,
    pad_right_at_dim,
    pad_right_at_dim_to,
    pad_sequence,
    pad_sequence_and_cat
)

import pytest

LEFT_PAD_WIDTH = 2
RIGHT_PAD_WIDTH = 3
LENGTH_STEP = 3
ROW_COUNT_TWO = 2
WIDTH_THREE = 3

PAD_FILL_FLOAT_73 = 73.0
PAD_FILL_INT_73 = 73
PAD_FILL_FLOAT_79 = 79.0
PAD_FILL_INT_79 = 79
PAD_FILL_FLOAT_83 = 83.0
PAD_FILL_INT_83 = 83
PAD_FILL_FLOAT_89 = 89.0
PAD_FILL_INT_89 = 89
PAD_FILL_FLOAT_97 = 97.0
PAD_FILL_INT_97 = 97
PAD_FILL_FLOAT_101 = 101.0
PAD_FILL_FLOAT_103 = 103.0


def _dimension_inputs_for_tensor(tensor_value: Tensor) -> list[int]:
    dimension_inputs = [-1, 0]
    if tensor_value.ndim > 1:
        dimension_inputs.extend([1, -2])
    return dimension_inputs


def _normalize_dimension_index(tensor_value: Tensor, dimension_index: int) -> int:
    return dimension_index if dimension_index >= 0 else tensor_value.ndim + dimension_index


def _fill_value_for_tensor(
    tensor_value: Tensor,
    *,
    float_fill_value: float,
    int_fill_value: int,
) -> float | int:
    return float_fill_value if tensor_value.is_floating_point() else int_fill_value


def _float_row_two_tensors(list_tensors: list[Tensor]) -> list[Tensor]:
    return [
        tensor_value
        for tensor_value in list_tensors
        if tensor_value.ndim == ROW_COUNT_TWO and tensor_value.shape[0] == ROW_COUNT_TWO and tensor_value.is_floating_point()
    ]


def _float_width_three_tensors(list_tensors: list[Tensor]) -> list[Tensor]:
    return [
        tensor_value
        for tensor_value in list_tensors
        if tensor_value.ndim == ROW_COUNT_TWO and tensor_value.shape[1] == WIDTH_THREE and tensor_value.is_floating_point()
    ]


def _assert_right_padded_sequence_matches_inputs(
    list_tensors: list[Tensor],
    padded_tensors: list[Tensor],
    *,
    max_last_dimension_length: int,
    fill_value: float,
) -> None:
    assert len(padded_tensors) == len(list_tensors), f"pad_sequence returned {len(padded_tensors)} tensors, expected {len(list_tensors)}."

    for tensor_index, (tensor_value, padded_tensor) in enumerate(zip(list_tensors, padded_tensors, strict=True)):
        expected_fill_value = fill_value if tensor_value.is_floating_point() else int(fill_value)
        expected_padding_width = max_last_dimension_length - tensor_value.shape[-1]

        assert padded_tensor.shape[-1] == max_last_dimension_length, (
            f"pad_sequence wrong padded length for {tensor_index=}; got {padded_tensor.shape[-1]}, expected {max_last_dimension_length}."
        )

        center_slice = padded_tensor.narrow(-1, 0, tensor_value.shape[-1])
        right_padding_slice = padded_tensor.narrow(-1, tensor_value.shape[-1], expected_padding_width)

        assert torch.equal(center_slice, tensor_value), f"pad_sequence changed right-padded center values for {tensor_index=}."
        assert torch.all(right_padding_slice == expected_fill_value), (
            f"pad_sequence generated incorrect right padding values for {tensor_index=}."
        )


def _assert_left_padded_sequence_matches_inputs(
    list_tensors: list[Tensor],
    padded_tensors: list[Tensor],
    *,
    max_last_dimension_length: int,
    fill_value: float,
) -> None:
    assert len(padded_tensors) == len(list_tensors), f"pad_sequence returned {len(padded_tensors)} tensors, expected {len(list_tensors)}."

    for tensor_index, (tensor_value, padded_tensor) in enumerate(zip(list_tensors, padded_tensors, strict=True)):
        expected_fill_value = fill_value if tensor_value.is_floating_point() else int(fill_value)
        expected_padding_width = max_last_dimension_length - tensor_value.shape[-1]

        center_slice = padded_tensor.narrow(-1, expected_padding_width, tensor_value.shape[-1])
        left_padding_slice = padded_tensor.narrow(-1, 0, expected_padding_width)

        assert torch.equal(center_slice, tensor_value), f"pad_sequence changed left-padded center values for {tensor_index=}."
        assert torch.all(left_padding_slice == expected_fill_value), (
            f"pad_sequence generated incorrect left padding values for {tensor_index=}."
        )


@pytest.mark.parametrize(
    ("left_pad_width", "right_pad_width"),
    [pytest.param(LEFT_PAD_WIDTH, RIGHT_PAD_WIDTH, id="asymmetric-padding-widths")],
)
def test_pad_at_dim(
    t: Tensor,
    left_pad_width: int,
    right_pad_width: int,
) -> None:
    fill_value = _fill_value_for_tensor(
        t,
        float_fill_value=PAD_FILL_FLOAT_73,
        int_fill_value=PAD_FILL_INT_73,
    )

    for dimension_index in _dimension_inputs_for_tensor(t):
        normalized_dimension_index = _normalize_dimension_index(t, dimension_index)
        original_length = t.shape[normalized_dimension_index]

        result = pad_at_dim(
            t,
            (left_pad_width, right_pad_width),
            dim=dimension_index,
            value=float(fill_value),
        )

        expected_shape = list(t.shape)
        expected_shape[normalized_dimension_index] = original_length + left_pad_width + right_pad_width

        assert tuple(result.shape) == tuple(expected_shape), (
            f"pad_at_dim returned shape {tuple(result.shape)}, expected {tuple(expected_shape)} for "
            f"{dimension_index=} and {tuple(t.shape)=}."
        )

        center_slice = result.narrow(normalized_dimension_index, left_pad_width, original_length)
        assert torch.equal(center_slice, t), (
            f"pad_at_dim changed center values for {dimension_index=} and {tuple(t.shape)=}; original tensor must be preserved."
        )

        left_padding_slice = result.narrow(normalized_dimension_index, 0, left_pad_width)
        right_padding_slice = result.narrow(
            normalized_dimension_index,
            result.shape[normalized_dimension_index] - right_pad_width,
            right_pad_width,
        )

        assert torch.all(left_padding_slice == fill_value), (
            f"pad_at_dim left padding values are incorrect for {dimension_index=} and {tuple(t.shape)=}."
        )
        assert torch.all(right_padding_slice == fill_value), (
            f"pad_at_dim right padding values are incorrect for {dimension_index=} and {tuple(t.shape)=}."
        )


@pytest.mark.parametrize("left_pad_width", [pytest.param(LEFT_PAD_WIDTH, id="left-padding-width-two")])
def test_pad_left_at_dim(
    t: Tensor,
    left_pad_width: int,
) -> None:
    fill_value = _fill_value_for_tensor(
        t,
        float_fill_value=PAD_FILL_FLOAT_79,
        int_fill_value=PAD_FILL_INT_79,
    )

    for dimension_index in _dimension_inputs_for_tensor(t):
        expected = pad_at_dim(
            t,
            (left_pad_width, 0),
            dim=dimension_index,
            value=float(fill_value),
        )
        result = pad_left_at_dim(
            t,
            left_pad_width,
            dim=dimension_index,
            value=float(fill_value),
        )

        assert torch.equal(result, expected), f"pad_left_at_dim does not match pad_at_dim for {dimension_index=} and {tuple(t.shape)=}."


@pytest.mark.parametrize("right_pad_width", [pytest.param(LEFT_PAD_WIDTH, id="right-padding-width-two")])
def test_pad_right_at_dim(
    t: Tensor,
    right_pad_width: int,
) -> None:
    fill_value = _fill_value_for_tensor(
        t,
        float_fill_value=PAD_FILL_FLOAT_83,
        int_fill_value=PAD_FILL_INT_83,
    )

    for dimension_index in _dimension_inputs_for_tensor(t):
        expected = pad_at_dim(
            t,
            (0, right_pad_width),
            dim=dimension_index,
            value=float(fill_value),
        )
        result = pad_right_at_dim(
            t,
            right_pad_width,
            dim=dimension_index,
            value=float(fill_value),
        )

        assert torch.equal(result, expected), f"pad_right_at_dim does not match pad_at_dim for {dimension_index=} and {tuple(t.shape)=}."


@pytest.mark.parametrize("length_step", [pytest.param(LENGTH_STEP, id="length-step-three")])
def test_pad_left_at_dim_to(
    t: Tensor,
    length_step: int,
) -> None:
    fill_value = _fill_value_for_tensor(
        t,
        float_fill_value=PAD_FILL_FLOAT_89,
        int_fill_value=PAD_FILL_INT_89,
    )

    for dimension_index in _dimension_inputs_for_tensor(t):
        normalized_dimension_index = _normalize_dimension_index(t, dimension_index)
        current_length = t.shape[normalized_dimension_index]
        target_lengths = sorted({max(current_length - 2, 0), current_length, current_length + 1, current_length + length_step})

        for target_length in target_lengths:
            result = pad_left_at_dim_to(
                t,
                target_length,
                dim=dimension_index,
                value=float(fill_value),
            )

            if target_length <= current_length:
                assert result is t, (
                    f"pad_left_at_dim_to returned a new tensor for non-expanding case with "
                    f"{dimension_index=}, {target_length=}, and {tuple(t.shape)=}."
                )
                continue

            left_pad_width = target_length - current_length
            expected_shape = list(t.shape)
            expected_shape[normalized_dimension_index] = target_length

            assert tuple(result.shape) == tuple(expected_shape), (
                f"pad_left_at_dim_to returned shape {tuple(result.shape)}, expected {tuple(expected_shape)} for "
                f"{dimension_index=}, {target_length=}, and {tuple(t.shape)=}."
            )

            center_slice = result.narrow(normalized_dimension_index, left_pad_width, current_length)
            left_padding_slice = result.narrow(normalized_dimension_index, 0, left_pad_width)

            assert torch.equal(center_slice, t), (
                f"pad_left_at_dim_to changed center values for {dimension_index=}, {target_length=}, and {tuple(t.shape)=}."
            )
            assert torch.all(left_padding_slice == fill_value), (
                f"pad_left_at_dim_to left padding values are incorrect for {dimension_index=}, {target_length=}, and {tuple(t.shape)=}."
            )


@pytest.mark.parametrize("length_step", [pytest.param(LENGTH_STEP, id="length-step-three")])
def test_pad_right_at_dim_to(
    t: Tensor,
    length_step: int,
) -> None:
    fill_value = _fill_value_for_tensor(
        t,
        float_fill_value=PAD_FILL_FLOAT_97,
        int_fill_value=PAD_FILL_INT_97,
    )

    for dimension_index in _dimension_inputs_for_tensor(t):
        normalized_dimension_index = _normalize_dimension_index(t, dimension_index)
        current_length = t.shape[normalized_dimension_index]
        target_lengths = sorted({max(current_length - 2, 0), current_length, current_length + 1, current_length + length_step})

        for target_length in target_lengths:
            result = pad_right_at_dim_to(
                t,
                target_length,
                dim=dimension_index,
                value=float(fill_value),
            )

            if target_length <= current_length:
                assert result is t, (
                    f"pad_right_at_dim_to returned a new tensor for non-expanding case with "
                    f"{dimension_index=}, {target_length=}, and {tuple(t.shape)=}."
                )
                continue

            right_pad_width = target_length - current_length
            expected_shape = list(t.shape)
            expected_shape[normalized_dimension_index] = target_length

            assert tuple(result.shape) == tuple(expected_shape), (
                f"pad_right_at_dim_to returned shape {tuple(result.shape)}, expected {tuple(expected_shape)} for "
                f"{dimension_index=}, {target_length=}, and {tuple(t.shape)=}."
            )

            center_slice = result.narrow(normalized_dimension_index, 0, current_length)
            right_padding_slice = result.narrow(normalized_dimension_index, current_length, right_pad_width)

            assert torch.equal(center_slice, t), (
                f"pad_right_at_dim_to changed center values for {dimension_index=}, {target_length=}, and {tuple(t.shape)=}."
            )
            assert torch.all(right_padding_slice == fill_value), (
                f"pad_right_at_dim_to right padding values are incorrect for {dimension_index=}, {target_length=}, and {tuple(t.shape)=}."
            )


def test_pad_sequence(
    sequence_tensors: list[Tensor],
    empty_tensor_sequence: list[Tensor],
) -> None:
    list_tensors = sequence_tensors

    empty_result = pad_sequence(empty_tensor_sequence)
    assert empty_result is None, "pad_sequence must return None for empty input tensors."

    max_last_dimension_length = max(tensor_value.shape[-1] for tensor_value in list_tensors)

    right_padded_output = pad_sequence(
        list_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_101,
        left=False,
        return_stacked=False,
        return_lens=False,
    )
    assert right_padded_output is not None, "pad_sequence returned None for non-empty input with return_stacked=False."
    _assert_right_padded_sequence_matches_inputs(
        list_tensors,
        right_padded_output,
        max_last_dimension_length=max_last_dimension_length,
        fill_value=PAD_FILL_FLOAT_101,
    )

    left_padded_output = pad_sequence(
        list_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_101,
        left=True,
        return_stacked=False,
        return_lens=False,
    )
    assert left_padded_output is not None, "pad_sequence returned None for non-empty input with left=True."
    _assert_left_padded_sequence_matches_inputs(
        list_tensors,
        left_padded_output,
        max_last_dimension_length=max_last_dimension_length,
        fill_value=PAD_FILL_FLOAT_101,
    )

    lengths_output = pad_sequence(
        list_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_101,
        left=False,
        return_stacked=False,
        return_lens=True,
        pad_lens=False,
    )
    assert lengths_output is not None, "pad_sequence returned None for non-empty input with return_lens=True."
    _padded_tensors_for_lengths, dimension_lengths = lengths_output
    expected_lengths = torch.tensor([tensor_value.shape[-1] for tensor_value in list_tensors], device=dimension_lengths.device)
    assert torch.equal(dimension_lengths, expected_lengths), "pad_sequence returned incorrect dimension lengths."

    pad_lengths_output = pad_sequence(
        list_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_101,
        left=False,
        return_stacked=False,
        return_lens=True,
        pad_lens=True,
    )
    assert pad_lengths_output is not None, "pad_sequence returned None for non-empty input with pad_lens=True."
    _padded_tensors_for_pad_lengths, pad_lengths = pad_lengths_output
    expected_pad_lengths = max_last_dimension_length - expected_lengths
    assert torch.equal(pad_lengths, expected_pad_lengths), "pad_sequence returned incorrect pad lengths."

    row_two_tensors = _float_row_two_tensors(list_tensors)
    assert len(row_two_tensors) > 1, "Need at least two float tensors with shape (2, n) for stacked checks."

    stacked_two_dimensional_output = pad_sequence(
        row_two_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_101,
        left=False,
        return_stacked=True,
        return_lens=True,
    )
    assert stacked_two_dimensional_output is not None, "pad_sequence returned None for stacked two-dimensional input."
    stacked_two_dimensional_tensors, stacked_lengths = stacked_two_dimensional_output

    max_two_dimensional_length = max(tensor_value.shape[-1] for tensor_value in row_two_tensors)
    assert tuple(stacked_two_dimensional_tensors.shape) == (
        len(row_two_tensors),
        ROW_COUNT_TWO,
        max_two_dimensional_length,
    ), "pad_sequence returned incorrect stacked two-dimensional shape."

    expected_stacked_lengths = torch.tensor(
        [tensor_value.shape[-1] for tensor_value in row_two_tensors],
        device=stacked_lengths.device,
    )
    assert torch.equal(stacked_lengths, expected_stacked_lengths), "pad_sequence returned incorrect stacked lengths."

    for tensor_index, tensor_value in enumerate(row_two_tensors):
        expected_padding_width = max_two_dimensional_length - tensor_value.shape[-1]
        center_slice = stacked_two_dimensional_tensors[tensor_index].narrow(-1, 0, tensor_value.shape[-1])
        right_padding_slice = stacked_two_dimensional_tensors[tensor_index].narrow(-1, tensor_value.shape[-1], expected_padding_width)

        assert torch.equal(center_slice, tensor_value), (
            f"pad_sequence changed stacked center values for row-two tensor with {tensor_index=}."
        )
        assert torch.all(right_padding_slice == PAD_FILL_FLOAT_101), (
            f"pad_sequence generated incorrect stacked right padding values for row-two tensor with {tensor_index=}."
        )

    width_three_tensors = _float_width_three_tensors(list_tensors)
    assert len(width_three_tensors) > 1, "Need at least two float tensors with width 3 for dim=0 stacked checks."

    stacked_dim_zero_output = pad_sequence(
        width_three_tensors,
        dim=0,
        value=PAD_FILL_FLOAT_101,
        left=False,
        return_stacked=True,
        return_lens=True,
    )
    assert stacked_dim_zero_output is not None, "pad_sequence returned None for dim=0 stacked checks."
    stacked_dim_zero_tensors, stacked_dim_zero_lengths = stacked_dim_zero_output

    max_dim_zero_length = max(tensor_value.shape[0] for tensor_value in width_three_tensors)
    assert tuple(stacked_dim_zero_tensors.shape) == (
        len(width_three_tensors),
        max_dim_zero_length,
        WIDTH_THREE,
    ), "pad_sequence returned incorrect dim=0 stacked shape."

    expected_dim_zero_lengths = torch.tensor(
        [tensor_value.shape[0] for tensor_value in width_three_tensors],
        device=stacked_dim_zero_lengths.device,
    )
    assert torch.equal(stacked_dim_zero_lengths, expected_dim_zero_lengths), "pad_sequence returned incorrect dim=0 lengths."


def test_pad_sequence_and_cat(
    sequence_tensors: list[Tensor],
    empty_tensor_sequence: list[Tensor],
) -> None:
    list_tensors = sequence_tensors

    empty_result = pad_sequence_and_cat(empty_tensor_sequence)
    assert empty_result is None, "pad_sequence_and_cat must return None for empty input tensors."

    one_dimensional_float_tensors = [
        tensor_value for tensor_value in list_tensors if tensor_value.ndim == 1 and tensor_value.is_floating_point()
    ]
    assert len(one_dimensional_float_tensors) > 1, "Need at least two one-dimensional float tensors for cat checks."

    manual_right_output = pad_sequence(
        one_dimensional_float_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_103,
        left=False,
        return_stacked=False,
        return_lens=False,
    )
    assert manual_right_output is not None, "pad_sequence returned None while preparing manual right-cat expectation."
    manual_right_expected = torch.cat(manual_right_output, dim=0)

    right_result = pad_sequence_and_cat(
        one_dimensional_float_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_103,
        left=False,
        dim_cat=0,
    )
    assert right_result is not None, "pad_sequence_and_cat returned None for one-dimensional float tensors."
    assert torch.equal(right_result, manual_right_expected), (
        "pad_sequence_and_cat right-padding output differs from manual pad_sequence + cat for one-dimensional float tensors."
    )

    manual_left_output = pad_sequence(
        one_dimensional_float_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_103,
        left=True,
        return_stacked=False,
        return_lens=False,
    )
    assert manual_left_output is not None, "pad_sequence returned None while preparing manual left-cat expectation."
    manual_left_expected = torch.cat(manual_left_output, dim=0)

    left_result = pad_sequence_and_cat(
        one_dimensional_float_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_103,
        left=True,
        dim_cat=0,
    )
    assert left_result is not None, "pad_sequence_and_cat returned None for one-dimensional float tensors with left=True."
    assert torch.equal(left_result, manual_left_expected), (
        "pad_sequence_and_cat left-padding output differs from manual pad_sequence + cat for one-dimensional float tensors."
    )

    row_two_tensors = _float_row_two_tensors(list_tensors)
    assert len(row_two_tensors) > 1, "Need at least two float tensors with shape (2, n) for two-dimensional cat checks."

    manual_two_dimensional_output = pad_sequence(
        row_two_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_103,
        left=False,
        return_stacked=False,
        return_lens=False,
    )
    assert manual_two_dimensional_output is not None, "pad_sequence returned None while preparing two-dimensional manual expectation."
    manual_two_dimensional_expected = torch.cat(manual_two_dimensional_output, dim=0)

    two_dimensional_result = pad_sequence_and_cat(
        row_two_tensors,
        dim=-1,
        value=PAD_FILL_FLOAT_103,
        left=False,
        dim_cat=0,
    )
    assert two_dimensional_result is not None, "pad_sequence_and_cat returned None for two-dimensional row-two tensors."
    assert torch.equal(two_dimensional_result, manual_two_dimensional_expected), (
        "pad_sequence_and_cat two-dimensional output differs from manual pad_sequence + cat result."
    )

    width_three_tensors = _float_width_three_tensors(list_tensors)
    assert len(width_three_tensors) > 1, "Need at least two float tensors with width 3 for dim=0 + dim=-1 checks."

    manual_dim_zero_output = pad_sequence(
        width_three_tensors,
        dim=0,
        value=PAD_FILL_FLOAT_103,
        left=False,
        return_stacked=False,
        return_lens=False,
    )
    assert manual_dim_zero_output is not None, "pad_sequence returned None while preparing dim=0 manual expectation."
    manual_dim_zero_expected = torch.cat(manual_dim_zero_output, dim=-1)

    dim_zero_result = pad_sequence_and_cat(
        width_three_tensors,
        dim=0,
        value=PAD_FILL_FLOAT_103,
        left=False,
        dim_cat=-1,
    )
    assert dim_zero_result is not None, "pad_sequence_and_cat returned None for dim=0 padding and dim=-1 concatenation."
    assert torch.equal(dim_zero_result, manual_dim_zero_expected), (
        "pad_sequence_and_cat dim=0 padding + dim=-1 cat output differs from manual pad_sequence + cat result."
    )
