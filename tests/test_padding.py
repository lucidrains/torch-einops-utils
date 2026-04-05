from __future__ import annotations

from collections.abc import Callable
from typing import Optional, assert_type

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


@pytest.mark.parametrize(
    ("pad_tuple", "dimension", "padding_value", "expected_tensor"),
    [
        pytest.param(
            (2, 3),
            -1,
            89.0,
            torch.tensor(
                [
                    [89.0, 89.0, 2.0, 3.0, 5.0, 89.0, 89.0, 89.0],
                    [89.0, 89.0, 7.0, 11.0, 13.0, 89.0, 89.0, 89.0],
                ],
            ),
            id="pad-last-dimension",
        ),
        pytest.param(
            (1, 2),
            0,
            79.0,
            torch.tensor(
                [
                    [79.0, 79.0, 79.0],
                    [2.0, 3.0, 5.0],
                    [7.0, 11.0, 13.0],
                    [79.0, 79.0, 79.0],
                    [79.0, 79.0, 79.0],
                ],
            ),
            id="positive-dimension-normalization",
        ),
        pytest.param(
            (1, 2),
            -2,
            79.0,
            torch.tensor(
                [
                    [79.0, 79.0, 79.0],
                    [2.0, 3.0, 5.0],
                    [7.0, 11.0, 13.0],
                    [79.0, 79.0, 79.0],
                    [79.0, 79.0, 79.0],
                ],
            ),
            id="negative-dimension-normalization",
        ),
    ],
)
def test_pad_at_dim(
    pad_tuple: tuple[int, int],
    dimension: int,
    padding_value: float,
    expected_tensor: Tensor,
    tensor_two_by_three_prime: Tensor,
) -> None:
    padded_tensor: Tensor = pad_at_dim(
        tensor_two_by_three_prime,
        pad_tuple,
        dim=dimension,
        value=padding_value,
    )

    assert tuple(padded_tensor.shape) == tuple(expected_tensor.shape), (
        f"pad_at_dim returned shape {tuple(padded_tensor.shape)}, expected "
        f"{tuple(expected_tensor.shape)} for {pad_tuple=}, {dimension=}, and {padding_value=}."
    )
    assert torch.equal(padded_tensor, expected_tensor), (
        f"pad_at_dim returned {padded_tensor}, expected {expected_tensor} for {pad_tuple=}, {dimension=}, and {padding_value=}."
    )


@pytest.mark.parametrize(
    ("padding_function", "padding_size", "dimension", "padding_value", "expected_pad_tuple"),
    [
        pytest.param(pad_left_at_dim, 2, 1, 83.0, (2, 0), id="left-wrapper"),
        pytest.param(pad_right_at_dim, 2, 1, 83.0, (0, 2), id="right-wrapper"),
    ],
)
def test_pad_side_wrappers_match_pad_at_dim(
    padding_function: Callable[..., Tensor],
    padding_size: int,
    dimension: int,
    padding_value: float,
    expected_pad_tuple: tuple[int, int],
    tensor_two_by_three_prime: Tensor,
) -> None:
    wrapped_tensor: Tensor = padding_function(
        tensor_two_by_three_prime,
        pad=padding_size,
        dim=dimension,
        value=padding_value,
    )
    base_tensor: Tensor = pad_at_dim(
        tensor_two_by_three_prime,
        expected_pad_tuple,
        dim=dimension,
        value=padding_value,
    )

    assert torch.equal(wrapped_tensor, base_tensor), (
        f"{padding_function.__name__} returned {wrapped_tensor}, expected {base_tensor} when compared with pad_at_dim."
    )


@pytest.mark.parametrize(
    (
        "padding_function",
        "target_length",
        "dimension",
        "padding_value",
        "expected_tensor",
        "is_expanded",
    ),
    [
        pytest.param(
            pad_left_at_dim_to,
            5,
            1,
            73.0,
            torch.tensor(
                [
                    [73.0, 73.0, 2.0, 3.0, 5.0],
                    [73.0, 73.0, 7.0, 11.0, 13.0],
                ],
            ),
            True,
            id="left-expand",
        ),
        pytest.param(
            pad_right_at_dim_to,
            5,
            1,
            73.0,
            torch.tensor(
                [
                    [2.0, 3.0, 5.0, 73.0, 73.0],
                    [7.0, 11.0, 13.0, 73.0, 73.0],
                ],
            ),
            True,
            id="right-expand",
        ),
        pytest.param(
            pad_left_at_dim_to,
            3,
            1,
            79.0,
            None,
            False,
            id="left-no-expand",
        ),
        pytest.param(
            pad_right_at_dim_to,
            2,
            1,
            79.0,
            None,
            False,
            id="right-no-expand",
        ),
    ],
)
def test_pad_at_dim_to(
    padding_function: Callable[..., Tensor],
    target_length: int,
    dimension: int,
    padding_value: float,
    expected_tensor: Tensor | None,
    is_expanded: bool,
    tensor_two_by_three_prime: Tensor,
) -> None:
    expected = expected_tensor if expected_tensor is not None else tensor_two_by_three_prime
    padded_tensor: Tensor = padding_function(
        tensor_two_by_three_prime,
        length=target_length,
        dim=dimension,
        value=padding_value,
    )

    if is_expanded:
        assert padded_tensor is not tensor_two_by_three_prime, (
            f"{padding_function.__name__} returned original tensor identity for {target_length=}, expected a new padded tensor."
        )
    else:
        assert padded_tensor is tensor_two_by_three_prime, (
            f"{padding_function.__name__} returned a new tensor for {target_length=}, "
            "expected original tensor identity when no expansion is needed."
        )

    assert tuple(padded_tensor.shape) == tuple(expected.shape), (
        f"{padding_function.__name__} returned shape {tuple(padded_tensor.shape)}, expected "
        f"{tuple(expected.shape)} for {target_length=} and dim={dimension}."
    )
    assert torch.equal(padded_tensor, expected), (
        f"{padding_function.__name__} returned {padded_tensor}, expected {expected} for {target_length=} and {padding_value=}."
    )


def test_pad_sequence_types(
    tensor_sequence_with_variable_lengths: list[Tensor],
) -> None:
    stacked_without_lengths = assert_type(
        pad_sequence(
            tensor_sequence_with_variable_lengths,
            dim=1,
            value=107.0,
            left=False,
            dim_stack=0,
            return_stacked=True,
            return_lens=False,
            pad_lens=False,
        ),
        Optional[Tensor],
    )
    stacked_with_lengths = assert_type(
        pad_sequence(
            tensor_sequence_with_variable_lengths,
            dim=1,
            value=109.0,
            left=True,
            dim_stack=1,
            return_stacked=True,
            return_lens=True,
            pad_lens=False,
        ),
        Optional[tuple[Tensor, Tensor]],
    )
    list_without_lengths = assert_type(
        pad_sequence(
            tensor_sequence_with_variable_lengths,
            dim=1,
            value=113.0,
            left=False,
            dim_stack=0,
            return_stacked=False,
            return_lens=False,
            pad_lens=False,
        ),
        Optional[list[Tensor]],
    )
    list_with_lengths = assert_type(
        pad_sequence(
            tensor_sequence_with_variable_lengths,
            dim=1,
            value=127.0,
            left=True,
            dim_stack=0,
            return_stacked=False,
            return_lens=True,
            pad_lens=True,
        ),
        Optional[tuple[list[Tensor], Tensor]],
    )

    assert isinstance(stacked_without_lengths, Tensor), (
        f"assert_type returned {type(stacked_without_lengths).__name__}, expected Tensor for "
        "pad_sequence with return_stacked=True and return_lens=False."
    )
    assert isinstance(stacked_with_lengths, tuple), (
        f"assert_type returned {type(stacked_with_lengths).__name__}, expected tuple for "
        "pad_sequence with return_stacked=True and return_lens=True."
    )
    assert isinstance(list_without_lengths, list), (
        f"assert_type returned {type(list_without_lengths).__name__}, expected list for "
        "pad_sequence with return_stacked=False and return_lens=False."
    )
    assert isinstance(list_with_lengths, tuple), (
        f"assert_type returned {type(list_with_lengths).__name__}, expected tuple for "
        "pad_sequence with return_stacked=False and return_lens=True."
    )


@pytest.mark.parametrize(
    (
        "left",
        "dimension",
        "dim_stack",
        "return_stacked",
        "return_lens",
        "pad_lens",
        "padding_value",
        "use_empty_input",
        "expected_last_tensor",
        "expected_stacked_shape",
    ),
    [
        pytest.param(
            False,
            1,
            0,
            True,
            False,
            False,
            83.0,
            False,
            torch.tensor([[59.0, 61.0, 83.0, 83.0, 83.0], [67.0, 71.0, 83.0, 83.0, 83.0]]),
            (3, 2, 5),
            id="stacked-without-lengths",
        ),
        pytest.param(
            True,
            1,
            1,
            True,
            True,
            False,
            89.0,
            False,
            torch.tensor([[89.0, 89.0, 89.0, 59.0, 61.0], [89.0, 89.0, 89.0, 67.0, 71.0]]),
            (2, 3, 5),
            id="stacked-with-lengths",
        ),
        pytest.param(
            False,
            1,
            0,
            False,
            False,
            False,
            97.0,
            False,
            torch.tensor([[59.0, 61.0, 97.0, 97.0, 97.0], [67.0, 71.0, 97.0, 97.0, 97.0]]),
            None,
            id="list-without-lengths",
        ),
        pytest.param(
            True,
            1,
            0,
            False,
            True,
            True,
            101.0,
            False,
            torch.tensor([[101.0, 101.0, 101.0, 59.0, 61.0], [101.0, 101.0, 101.0, 67.0, 71.0]]),
            None,
            id="list-with-lengths",
        ),
        pytest.param(False, 1, 0, True, False, False, 73.0, True, None, None, id="empty-stacked-no-lengths"),
        pytest.param(False, 1, 0, True, True, False, 73.0, True, None, None, id="empty-stacked-with-lengths"),
        pytest.param(False, 1, 0, False, False, False, 73.0, True, None, None, id="empty-list-no-lengths"),
        pytest.param(
            False,
            1,
            0,
            False,
            True,
            True,
            73.0,
            True,
            None,
            None,
            id="empty-list-with-padding-lengths",
        ),
    ],
)
def test_pad_sequence(
    left: bool,
    dimension: int,
    dim_stack: int,
    return_stacked: bool,
    return_lens: bool,
    pad_lens: bool,
    padding_value: float,
    use_empty_input: bool,
    expected_last_tensor: Tensor | None,
    expected_stacked_shape: tuple[int, int, int] | None,
    tensor_sequence_with_variable_lengths: list[Tensor],
    tensor_sequence_lengths_prime: Tensor,
    tensor_sequence_padding_lengths_prime: Tensor,
) -> None:
    tensors: list[Tensor] = [] if use_empty_input else tensor_sequence_with_variable_lengths
    output_value: Tensor | list[Tensor] | tuple[Tensor | list[Tensor], Tensor] | None = pad_sequence(
        tensors,
        dim=dimension,
        value=padding_value,
        left=left,
        dim_stack=dim_stack,
        return_stacked=return_stacked,
        return_lens=return_lens,
        pad_lens=pad_lens,
    )
    parameter_description: str = (
        f"{left=}, {dimension=}, {dim_stack=}, {return_stacked=}, {return_lens=}, {pad_lens=}, {padding_value=}, and {use_empty_input=}"
    )

    if use_empty_input:
        assert output_value is None, f"pad_sequence returned {output_value}, expected None for empty tensors with {parameter_description}."
        return

    assert output_value is not None, (
        f"pad_sequence returned None for non-empty tensors, expected padded output for {parameter_description}."
    )

    payload: Tensor | list[Tensor]
    if return_lens:
        assert isinstance(output_value, tuple), (
            f"pad_sequence returned type {type(output_value).__name__}, expected tuple for {parameter_description}."
        )
        payload, lengths = output_value
        expected_lengths: Tensor = tensor_sequence_padding_lengths_prime if pad_lens else tensor_sequence_lengths_prime
        assert torch.equal(lengths, expected_lengths), (
            f"pad_sequence returned lengths {lengths}, expected {expected_lengths} for {parameter_description}."
        )
    else:
        assert not isinstance(output_value, tuple), (
            f"pad_sequence returned tuple output {output_value}, expected Tensor or list for {parameter_description}."
        )
        payload = output_value

    if return_stacked:
        assert isinstance(payload, Tensor), (
            f"pad_sequence returned payload type {type(payload).__name__}, expected Tensor for {parameter_description}."
        )
        assert tuple(payload.shape) == expected_stacked_shape, (
            f"pad_sequence returned stacked shape {tuple(payload.shape)}, expected {expected_stacked_shape} for {parameter_description}."
        )
        padded_last_tensor: Tensor = payload.select(dim_stack, 2)
    else:
        assert isinstance(payload, list), (
            f"pad_sequence returned payload type {type(payload).__name__}, expected list for {parameter_description}."
        )
        assert len(payload) == len(tensor_sequence_with_variable_lengths), (
            f"pad_sequence returned list length {len(payload)}, expected "
            f"{len(tensor_sequence_with_variable_lengths)} for {parameter_description}."
        )
        assert all(tuple(padded_tensor.shape) == (2, 5) for padded_tensor in payload), (
            "pad_sequence returned at least one tensor with incorrect shape, expected each padded "
            f"tensor to have shape (2, 5) for {parameter_description}."
        )
        padded_last_tensor = payload[2]

    assert torch.equal(padded_last_tensor, expected_last_tensor), (
        f"pad_sequence returned last tensor {padded_last_tensor}, expected {expected_last_tensor} for {parameter_description}."
    )


@pytest.mark.parametrize(
    ("left", "dim_sequence", "dim_cat", "padding_value", "use_empty_input", "expected_shape"),
    [
        pytest.param(False, 1, 0, 79.0, False, (6, 5), id="cat-leading-dimension"),
        pytest.param(True, 1, 1, 83.0, False, (2, 15), id="cat-feature-dimension"),
        pytest.param(False, 1, 0, 89.0, True, None, id="empty-right-cat"),
        pytest.param(True, 1, 1, 97.0, True, None, id="empty-left-cat"),
    ],
)
def test_pad_sequence_and_cat(
    left: bool,
    dim_sequence: int,
    dim_cat: int,
    padding_value: float,
    use_empty_input: bool,
    expected_shape: tuple[int, int] | None,
    tensor_sequence_with_variable_lengths: list[Tensor],
) -> None:
    tensors: list[Tensor] = [] if use_empty_input else tensor_sequence_with_variable_lengths
    output_tensor: Tensor | None = pad_sequence_and_cat(
        tensors,
        dim=dim_sequence,
        value=padding_value,
        left=left,
        dim_cat=dim_cat,
    )
    parameter_description: str = f"{left=}, {dim_sequence=}, {dim_cat=}, {padding_value=}, and {use_empty_input=}"

    if use_empty_input:
        assert output_tensor is None, f"pad_sequence_and_cat returned {output_tensor}, expected None for {parameter_description}."
        return

    manually_padded = pad_sequence(
        tensor_sequence_with_variable_lengths,
        dim=dim_sequence,
        value=padding_value,
        left=left,
        return_stacked=False,
        return_lens=False,
    )

    assert output_tensor is not None, (
        f"pad_sequence_and_cat returned None for non-empty tensors with {parameter_description}, expected concatenated Tensor."
    )
    assert isinstance(manually_padded, list), (
        f"pad_sequence returned type {type(manually_padded).__name__}, expected list for manual "
        f"concatenation comparison with {parameter_description}."
    )

    expected_tensor: Tensor = torch.cat(manually_padded, dim=dim_cat)

    assert tuple(output_tensor.shape) == expected_shape, (
        f"pad_sequence_and_cat returned shape {tuple(output_tensor.shape)}, expected {expected_shape} for {parameter_description}."
    )
    assert torch.equal(output_tensor, expected_tensor), (
        f"pad_sequence_and_cat returned {output_tensor}, expected {expected_tensor} for {parameter_description}."
    )
