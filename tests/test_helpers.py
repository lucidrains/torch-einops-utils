from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from torch_einops_utils import (
    compact,
    default,
    divisible_by,
    exists,
    first,
    identity,
    safe
)

import pytest


@pytest.mark.parametrize(
    ("input_value", "fallback_value", "expected"),
    [
        pytest.param(None, 42, 42, id="none-returns-fallback-int"),
        pytest.param(None, "default_string", "default_string", id="none-returns-fallback-str"),
        pytest.param(7, 42, 7, id="exists-returns-value-int"),
        pytest.param("alpha", "default_string", "alpha", id="exists-returns-value-str"),
    ],
)
def test_default_returns_fallback_only_when_none(
    input_value: int | str | None,
    fallback_value: int | str,
    expected: int | str,
) -> None:
    result = default(input_value, fallback_value)
    assert result == expected, f"default returned {result}, expected {expected} for {input_value=} and {fallback_value=}."


@pytest.mark.parametrize(
    ("input_value", "expected"),
    [
        pytest.param(None, False, id="none"),
        pytest.param(0, True, id="zero"),
        pytest.param(False, True, id="boolean"),
        pytest.param("alpha", True, id="string"),
        pytest.param([], True, id="empty-list"),
    ],
)
def test_exists_identifies_non_none_values(
    input_value: int | str | list[int] | None,
    expected: bool,
) -> None:
    result = exists(input_value)
    assert result is expected, f"exists returned {result}, expected {expected} for {input_value=}."


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        pytest.param(12, 3, True, id="int-divisible"),
        pytest.param(13, 3, False, id="int-not-divisible"),
        pytest.param(12.0, 3.0, True, id="float-divisible"),
        pytest.param(13.0, 3.0, False, id="float-not-divisible"),
        pytest.param(12, 0, False, id="zero-division"),
    ],
)
def test_divisible_by_validates_modulus(
    numerator: float,
    denominator: float,
    expected: bool,
) -> None:
    result = divisible_by(numerator, denominator)
    assert result is expected, f"divisible_by returned {result}, expected {expected} for {numerator=} and {denominator=}."


@pytest.mark.parametrize(
    "input_value",
    [
        pytest.param(42, id="int"),
        pytest.param("alpha", id="string"),
        pytest.param([2, 3, 5], id="list"),
    ],
)
def test_identity_returns_exact_object(input_value: int | str | list[int]) -> None:
    result = identity(input_value, "ignored_arg", kwarg="ignored")
    assert result is input_value, f"identity returned different object reference for {input_value=}."


@pytest.mark.parametrize(
    ("sequence_value", "expected_first"),
    [
        pytest.param([2, 3, 5], 2, id="list-of-int"),
        pytest.param((7, 11, 13), 7, id="tuple-of-int"),
        pytest.param(["alpha", "beta", "gamma"], "alpha", id="list-of-str"),
    ],
)
def test_first_returns_zero_index(
    sequence_value: Sequence[int] | Sequence[str],
    expected_first: int | str,
) -> None:
    result = first(sequence_value)
    assert result == expected_first, f"first returned {result}, expected {expected_first} for {sequence_value=}."


@pytest.mark.parametrize(
    "expected",
    [pytest.param([2, 3, "alpha", 5], id="mixed-sequence")],
)
def test_compact_removes_none_from_mixed_sequence(
    mixed_sequence_with_nones: list[int | str | None],
    expected: list[int | str],
) -> None:
    result = compact(mixed_sequence_with_nones)

    assert result == expected, f"compact returned {result}, expected {expected} for {mixed_sequence_with_nones=}."


@pytest.mark.parametrize(
    "expected_count",
    [pytest.param(3, id="three-non-none-tensors")],
)
def test_compact_with_tensors(
    tensor_sequence_with_nones: list[Tensor | None],
    expected_count: int,
) -> None:
    result = compact(tensor_sequence_with_nones)

    assert len(result) == expected_count, (
        f"compact returned list of length {len(result)}, expected {expected_count} "
        f"non-None tensors from input of length {len(tensor_sequence_with_nones)}."
    )
    assert not any(v is None for v in result), "compact included None values in the result."


@pytest.mark.parametrize(
    ("tensors_list", "expected_output", "expect_none", "expect_identity"),
    [
        pytest.param([None, None], None, True, False, id="all-none"),
        pytest.param(
            [torch.tensor([2.0, 3.0]), None, torch.tensor([5.0, 7.0])],
            torch.tensor([2.0, 3.0, 5.0, 7.0]),
            False,
            False,
            id="multiple-tensors",
        ),
    ],
)
def test_safe_decorator_unwraps_tensors(
    tensors_list: list[Tensor | None],
    expected_output: Tensor | None,
    expect_none: bool,
    expect_identity: bool,
) -> None:
    @safe
    def dummy_func(tensors_arg: Sequence[Tensor | None]) -> Tensor | None:
        return torch.cat(compact(tensors_arg))

    result = dummy_func(tensors_list)

    if expect_none:
        assert result is None, f"safe-decorated function returned {result}, expected None for all-None input."
    elif expect_identity:
        # Find the first non-None tensor in the input list
        original_tensor = next(t for t in tensors_list if t is not None)
        assert result is original_tensor, (
            "safe-decorated function returned different tensor identity, expected identical reference for single active tensor."
        )
    else:
        assert result is not None, "safe-decorated function returned None for multiple active tensors."
        assert expected_output is not None
        assert torch.equal(result, expected_output), (
            f"safe-decorated function returned {result}, expected {expected_output} for multiple active tensors."
        )
