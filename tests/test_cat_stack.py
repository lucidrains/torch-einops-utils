from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor

from torch_einops_utils import safe_cat, safe_stack


def test_safe_stack_returns_stacked_tensor(sequence_collection: list[Sequence[Tensor]]) -> None:
    for sequence in sequence_collection:
        unique_shapes = {t.shape for t in sequence}
        if len(unique_shapes) != 1:
            continue
        result = safe_stack(sequence)
        assert isinstance(result, Tensor), (
            f"safe_stack returned {type(result).__name__}, expected Tensor "
            f"for sequence of {len(sequence)} tensors each with shape {sequence[0].shape}."
        )
        assert result.shape[0] == len(sequence), (
            f"safe_stack returned shape {result.shape}, expected dim 0 to be {len(sequence)} "
            f"for {len(sequence)} input tensors with shape {sequence[0].shape}."
        )
        assert result.shape[1:] == sequence[0].shape, (
            f"safe_stack returned shape {result.shape}, expected trailing shape {sequence[0].shape} "
            f"matching input tensor shape for sequence of length {len(sequence)}."
        )


def test_safe_cat_returns_concatenated_tensor(sequence_collection: list[Sequence[Tensor]]) -> None:
    for sequence in sequence_collection:
        unique_ranks = {t.dim() for t in sequence}
        unique_trailing_shapes = {t.shape[1:] for t in sequence}
        if len(unique_ranks) != 1 or len(unique_trailing_shapes) != 1:
            continue
        result = safe_cat(sequence)
        expected_dim0 = sum(t.shape[0] for t in sequence)
        assert isinstance(result, Tensor), (
            f"safe_cat returned {type(result).__name__}, expected Tensor for sequence of {len(sequence)} tensors."
        )
        assert result.shape[0] == expected_dim0, (
            f"safe_cat returned shape {result.shape}, expected dim 0 to be {expected_dim0} "
            f"from concatenating {len(sequence)} tensors along dim 0."
        )
        assert result.shape[1:] == sequence[0].shape[1:], (
            f"safe_cat returned shape {result.shape}, expected trailing shape {sequence[0].shape[1:]} "
            f"matching input tensor trailing shape for sequence of length {len(sequence)}."
        )
