from __future__ import annotations

import torch
from torch import Tensor

from torch_einops_utils import and_masks, lens_to_mask, or_masks, reduce_masks


def test_lens_to_mask(t: Tensor) -> None:
    base_length = int(t.shape[-1])
    lengths_vector = torch.tensor(
        [base_length, base_length + 2, base_length + 3, base_length + 5],
        dtype=torch.long,
    )
    inferred_mask = lens_to_mask(lengths_vector)

    inferred_max_length = int(lengths_vector.max().item())
    assert tuple(inferred_mask.shape) == (len(lengths_vector), inferred_max_length), (
        f"lens_to_mask returned shape {tuple(inferred_mask.shape)}, expected ({len(lengths_vector)}, {inferred_max_length})."
    )
    assert inferred_mask.dtype == torch.bool, f"lens_to_mask returned dtype {inferred_mask.dtype}, expected torch.bool."

    list_lengths_vector: list[int] = lengths_vector.tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    for tensor_index, length_value in enumerate(list_lengths_vector):
        row_mask = inferred_mask[tensor_index]

        assert int(row_mask.sum().item()) == int(length_value), (
            f"lens_to_mask row sum mismatch for {tensor_index=}; got {int(row_mask.sum().item())}, expected {int(length_value)}."
        )
        assert torch.all(row_mask[:length_value]), (
            f"lens_to_mask did not set leading True values correctly for {tensor_index=} and {length_value=}."
        )
        assert not torch.any(row_mask[length_value:]), (
            f"lens_to_mask did not set trailing False values correctly for {tensor_index=} and {length_value=}."
        )

    explicit_max_length = base_length + 23
    explicit_mask = lens_to_mask(lengths_vector, max_len=explicit_max_length)

    assert tuple(explicit_mask.shape) == (len(lengths_vector), explicit_max_length), (
        f"lens_to_mask returned shape {tuple(explicit_mask.shape)} with explicit max length, "
        f"expected ({len(lengths_vector)}, {explicit_max_length})."
    )

    list_lengths_vector: list[int] = lengths_vector.tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    for tensor_index, length_value in enumerate(list_lengths_vector):
        row_mask = explicit_mask[tensor_index]

        assert torch.all(row_mask[:length_value]), (
            f"lens_to_mask explicit max length did not preserve leading True values for {tensor_index=} and {length_value=}."
        )
        assert not torch.any(row_mask[length_value:]), (
            f"lens_to_mask explicit max length did not preserve trailing False values for {tensor_index=} and {length_value=}."
        )

    matrix_lengths = torch.tensor(
        [
            [base_length, base_length + 2],
            [base_length + 3, base_length + 5],
            [base_length + 7, base_length + 11],
            [base_length + 13, base_length + 17],
        ],
        dtype=torch.long,
    )
    matrix_mask = lens_to_mask(matrix_lengths, max_len=explicit_max_length)

    assert tuple(matrix_mask.shape) == (4, 2, explicit_max_length), (
        f"lens_to_mask returned matrix shape {tuple(matrix_mask.shape)}, expected (4, 2, {explicit_max_length})."
    )

    for row_index in range(matrix_lengths.shape[0]):
        for column_index in range(matrix_lengths.shape[1]):
            length_value = int(matrix_lengths[row_index, column_index].item())
            cell_mask = matrix_mask[row_index, column_index]

            assert int(cell_mask.sum().item()) == length_value, (
                f"lens_to_mask matrix cell sum mismatch for {row_index=}, {column_index=}; expected {length_value}."
            )
            assert torch.all(cell_mask[:length_value]), (
                f"lens_to_mask matrix cell leading True values are incorrect for {row_index=}, {column_index=}."
            )
            assert not torch.any(cell_mask[length_value:]), (
                f"lens_to_mask matrix cell trailing False values are incorrect for {row_index=}, {column_index=}."
            )


def test_reduce_masks(
    sequence_tensors: list[Tensor],
    empty_optional_tensor_sequence: list[Tensor | None],
) -> None:
    list_tensors = sequence_tensors

    lengths_vector = torch.tensor([tensor_value.shape[-1] for tensor_value in list_tensors], dtype=torch.long)
    max_mask_length = int(lengths_vector.max().item()) + 4

    list_lengths_vector: list[int] = lengths_vector.tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    mask_collection: list[Tensor] = [
        lens_to_mask(torch.tensor([int(length_value)]), max_len=max_mask_length).squeeze(0) for length_value in list_lengths_vector
    ]

    empty_and_result = reduce_masks(empty_optional_tensor_sequence, torch.logical_and)
    empty_or_result = reduce_masks(empty_optional_tensor_sequence, torch.logical_or)
    assert empty_and_result is None, "reduce_masks must return None for empty input with logical_and."
    assert empty_or_result is None, "reduce_masks must return None for empty input with logical_or."

    all_none_sequence: list[Tensor | None] = [None for _ in mask_collection]
    all_none_and_result = reduce_masks(all_none_sequence, torch.logical_and)
    all_none_or_result = reduce_masks(all_none_sequence, torch.logical_or)
    assert all_none_and_result is None, "reduce_masks must return None for all-None input with logical_and."
    assert all_none_or_result is None, "reduce_masks must return None for all-None input with logical_or."

    interspersed_sequence: list[Tensor | None] = []
    for mask_index, mask_value in enumerate(mask_collection):
        interspersed_sequence.append(mask_value)
        if mask_index % 2 == 0:
            interspersed_sequence.append(None)

    active_masks = [mask_value for mask_value in interspersed_sequence if mask_value is not None]
    assert len(active_masks) > 0, "reduce_masks test requires at least one active mask after interspersing None entries."

    expected_and_mask = active_masks[0].clone()
    expected_or_mask = active_masks[0].clone()
    for mask_value in active_masks[1:]:
        expected_and_mask = torch.logical_and(expected_and_mask, mask_value)
        expected_or_mask = torch.logical_or(expected_or_mask, mask_value)

    and_result = reduce_masks(interspersed_sequence, torch.logical_and)
    or_result = reduce_masks(interspersed_sequence, torch.logical_or)

    assert and_result is not None, "reduce_masks returned None for logical_and with active masks present."
    assert or_result is not None, "reduce_masks returned None for logical_or with active masks present."
    assert torch.equal(and_result, expected_and_mask), "reduce_masks logical_and result differs from manual reduction expectation."
    assert torch.equal(or_result, expected_or_mask), "reduce_masks logical_or result differs from manual reduction expectation."

    for mask_index, mask_value in enumerate(mask_collection):
        single_and_result = reduce_masks([None, mask_value, None], torch.logical_and)
        single_or_result = reduce_masks([None, mask_value, None], torch.logical_or)

        assert single_and_result is not None, f"reduce_masks returned None for single active logical_and mask at {mask_index=}."
        assert single_or_result is not None, f"reduce_masks returned None for single active logical_or mask at {mask_index=}."
        assert torch.equal(single_and_result, mask_value), f"reduce_masks logical_and changed a single active mask at {mask_index=}."
        assert torch.equal(single_or_result, mask_value), f"reduce_masks logical_or changed a single active mask at {mask_index=}."


def test_and_masks(
    sequence_tensors: list[Tensor],
    empty_optional_tensor_sequence: list[Tensor | None],
) -> None:
    list_tensors = sequence_tensors

    lengths_vector = torch.tensor([tensor_value.shape[-1] for tensor_value in list_tensors], dtype=torch.long)
    max_mask_length = int(lengths_vector.max().item()) + 4

    list_lengths_vector: list[int] = lengths_vector.tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    mask_collection: list[Tensor] = [
        lens_to_mask(torch.tensor([int(length_value)]), max_len=max_mask_length).squeeze(0) for length_value in list_lengths_vector
    ]

    empty_result = and_masks(empty_optional_tensor_sequence)
    assert empty_result is None, "and_masks must return None for empty input tensors."

    all_none_sequence: list[Tensor | None] = [None for _ in mask_collection]
    all_none_result = and_masks(all_none_sequence)
    assert all_none_result is None, "and_masks must return None for all-None input tensors."

    interspersed_sequence: list[Tensor | None] = []
    for mask_index, mask_value in enumerate(mask_collection):
        interspersed_sequence.append(mask_value)
        if mask_index % 3 == 0:
            interspersed_sequence.append(None)

    expected_result = reduce_masks(interspersed_sequence, torch.logical_and)
    result = and_masks(interspersed_sequence)

    assert expected_result is not None, "reduce_masks returned None while preparing and_masks expected output."
    assert result is not None, "and_masks returned None for active masks input."
    assert torch.equal(result, expected_result), "and_masks output differs from reduce_masks with logical_and."

    full_result = and_masks(mask_collection)
    assert full_result is not None, "and_masks returned None for full active mask collection."

    expected_true_count = int(lengths_vector.min().item())
    assert int(full_result.sum().item()) == expected_true_count, (
        f"and_masks true-count mismatch: got {int(full_result.sum().item())}, expected {expected_true_count}."
    )


def test_or_masks(
    sequence_tensors: list[Tensor],
    empty_optional_tensor_sequence: list[Tensor | None],
) -> None:
    list_tensors = sequence_tensors

    lengths_vector = torch.tensor([tensor_value.shape[-1] for tensor_value in list_tensors], dtype=torch.long)
    max_mask_length = int(lengths_vector.max().item()) + 4

    list_lengths_vector: list[int] = lengths_vector.tolist()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    mask_collection: list[Tensor] = [
        lens_to_mask(torch.tensor([int(length_value)]), max_len=max_mask_length).squeeze(0) for length_value in list_lengths_vector
    ]

    empty_result = or_masks(empty_optional_tensor_sequence)
    assert empty_result is None, "or_masks must return None for empty input tensors."

    all_none_sequence: list[Tensor | None] = [None for _ in mask_collection]
    all_none_result = or_masks(all_none_sequence)
    assert all_none_result is None, "or_masks must return None for all-None input tensors."

    interspersed_sequence: list[Tensor | None] = []
    for mask_index, mask_value in enumerate(mask_collection):
        interspersed_sequence.append(mask_value)
        if mask_index % 4 == 0:
            interspersed_sequence.append(None)

    expected_result = reduce_masks(interspersed_sequence, torch.logical_or)
    result = or_masks(interspersed_sequence)

    assert expected_result is not None, "reduce_masks returned None while preparing or_masks expected output."
    assert result is not None, "or_masks returned None for active masks input."
    assert torch.equal(result, expected_result), "or_masks output differs from reduce_masks with logical_or."

    full_result = or_masks(mask_collection)
    assert full_result is not None, "or_masks returned None for full active mask collection."

    expected_true_count = int(lengths_vector.max().item())
    assert int(full_result.sum().item()) == expected_true_count, (
        f"or_masks true-count mismatch: got {int(full_result.sum().item())}, expected {expected_true_count}."
    )
