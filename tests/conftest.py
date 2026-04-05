from __future__ import annotations

import torch
from torch import Tensor

import pytest


@pytest.fixture
def tensor_two_by_three_prime() -> Tensor:
    return torch.tensor(
        [
            [2.0, 3.0, 5.0],
            [7.0, 11.0, 13.0],
        ],
    )


@pytest.fixture
def tensor_sequence_with_variable_lengths() -> list[Tensor]:
    return [
        torch.tensor(
            [
                [2.0, 3.0, 5.0],
                [7.0, 11.0, 13.0],
            ],
        ),
        torch.tensor(
            [
                [17.0, 19.0, 23.0, 29.0, 31.0],
                [37.0, 41.0, 43.0, 47.0, 53.0],
            ],
        ),
        torch.tensor(
            [
                [59.0, 61.0],
                [67.0, 71.0],
            ],
        ),
    ]


@pytest.fixture
def tensor_sequence_lengths_prime() -> Tensor:
    return torch.tensor([3, 5, 2])


@pytest.fixture
def tensor_sequence_padding_lengths_prime() -> Tensor:
    return torch.tensor([2, 0, 3])


@pytest.fixture
def mixed_sequence_with_nones() -> list[int | str | None]:
    return [2, None, 3, "alpha", None, 5]


@pytest.fixture
def tensor_sequence_with_nones() -> list[Tensor | None]:
    return [
        torch.tensor([2.0, 3.0]),
        None,
        torch.tensor([5.0, 7.0]),
        None,
        torch.tensor([11.0, 13.0]),
    ]
