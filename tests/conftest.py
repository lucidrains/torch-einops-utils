from __future__ import annotations

import datetime
import itertools
import random
from collections.abc import Iterator, Sequence
from more_itertools import extract
import torch
from torch import Tensor

import pytest

TENSOR_SPECS: list[tuple[str, Tensor]] = [
    ("rank-2-2x5-a", torch.tensor([[2.0, 3.0, 5.0, 73.0, 73.0], [7.0, 11.0, 13.0, 73.0, 73.0]])),
    ("rank-2-2x3", torch.tensor([[2.0, 3.0, 5.0], [7.0, 11.0, 13.0]])),
    ("rank-2-2x5-b", torch.tensor([[17.0, 19.0, 23.0, 29.0, 31.0], [37.0, 41.0, 43.0, 47.0, 53.0]])),
    ("rank-2-2x5-c", torch.tensor([[59.0, 61.0, 83.0, 83.0, 83.0], [67.0, 71.0, 83.0, 83.0, 83.0]])),
    ("rank-2-2x5-d", torch.tensor([[59.0, 61.0, 97.0, 97.0, 97.0], [67.0, 71.0, 97.0, 97.0, 97.0]])),
    ("rank-2-2x2", torch.tensor([[59.0, 61.0], [67.0, 71.0]])),
    ("rank-2-2x5-e", torch.tensor([[73.0, 73.0, 2.0, 3.0, 5.0], [73.0, 73.0, 7.0, 11.0, 13.0]])),
    ("rank-2-5x3", torch.tensor([[79.0, 79.0, 79.0], [2.0, 3.0, 5.0], [7.0, 11.0, 13.0], [79.0, 79.0, 79.0], [79.0, 79.0, 79.0]])),
    ("rank-2-2x8", torch.tensor([[89.0, 89.0, 2.0, 3.0, 5.0, 89.0, 89.0, 89.0], [89.0, 89.0, 7.0, 11.0, 13.0, 89.0, 89.0, 89.0]])),
    ("rank-2-2x5-f", torch.tensor([[89.0, 89.0, 89.0, 59.0, 61.0], [89.0, 89.0, 89.0, 67.0, 71.0]])),
    ("rank-2-2x5-g", torch.tensor([[101.0, 101.0, 101.0, 59.0, 61.0], [101.0, 101.0, 101.0, 67.0, 71.0]])),
    ("rank-1-len-3-a", torch.tensor([2, 0, 3])),
    ("rank-1-len-2-a", torch.tensor([2.0, 3.0])),
    ("rank-1-len-3-b", torch.tensor([3, 5, 2])),
    ("rank-1-len-2-b", torch.tensor([5.0, 7.0])),
    ("rank-1-len-2-c", torch.tensor([11.0, 13.0])),
]

LIST_T: list[Tensor] = [tensor_value for _tensor_id, tensor_value in TENSOR_SPECS]

T_PARAMS: list[object] = [pytest.param(tensor_value, id=tensor_id) for tensor_id, tensor_value in TENSOR_SPECS]


def _day_of_year() -> int:
    return int(datetime.datetime.now(tz=datetime.timezone.utc).timetuple().tm_yday)


def _daily_shuffled_tensors(parameter_index: int) -> list[Tensor]:
    random_seed = _day_of_year() * parameter_index
    random_generator = random.Random(random_seed)  # noqa: S311
    list_shuffled_tensors = list(LIST_T)
    random_generator.shuffle(list_shuffled_tensors)
    return list_shuffled_tensors


def _product_sequences_of_length(sequence_length: int) -> Iterator[tuple[Tensor, ...]]:
    cartesian_product: Iterator[tuple[Tensor, ...]] = itertools.product(LIST_T, *map(_daily_shuffled_tensors, range(1, sequence_length)))
    indices: list[int] = torch.arange(len(LIST_T)).tolist()  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    list_sequences: Iterator[tuple[Tensor, ...]] = extract(cartesian_product, indices)

    return list_sequences


@pytest.fixture
def list_t_len_1() -> list[list[Tensor]]:
    return [*map(list, _product_sequences_of_length(1))]


@pytest.fixture
def list_t_len_2() -> list[list[Tensor]]:
    return [*map(list, _product_sequences_of_length(2))]


@pytest.fixture
def list_t_len_3() -> list[list[Tensor]]:
    return [*map(list, _product_sequences_of_length(3))]


@pytest.fixture
def list_t_len_4() -> list[list[Tensor]]:
    return [*map(list, _product_sequences_of_length(4))]


@pytest.fixture
def list_t_len_5() -> list[list[Tensor]]:
    return [*map(list, _product_sequences_of_length(5))]


@pytest.fixture
def tuple_t_len_1() -> list[tuple[Tensor, ...]]:
    return list(_product_sequences_of_length(1))


@pytest.fixture
def tuple_t_len_2() -> list[tuple[Tensor, ...]]:
    return list(_product_sequences_of_length(2))


@pytest.fixture
def tuple_t_len_3() -> list[tuple[Tensor, ...]]:
    return list(_product_sequences_of_length(3))


@pytest.fixture
def tuple_t_len_4() -> list[tuple[Tensor, ...]]:
    return list(_product_sequences_of_length(4))


@pytest.fixture
def tuple_t_len_5() -> list[tuple[Tensor, ...]]:
    return list(_product_sequences_of_length(5))


SEQUENCE_FIXTURE_NAMES: tuple[str, ...] = (
    "list_t_len_1",
    "list_t_len_2",
    "list_t_len_3",
    "list_t_len_4",
    "list_t_len_5",
    "tuple_t_len_1",
    "tuple_t_len_2",
    "tuple_t_len_3",
    "tuple_t_len_4",
    "tuple_t_len_5",
)


@pytest.fixture(params=[pytest.param(fixture_name, id=fixture_name) for fixture_name in SEQUENCE_FIXTURE_NAMES])
def sequence_collection(request: pytest.FixtureRequest) -> list[Sequence[Tensor]]:
    return request.getfixturevalue(request.param)


@pytest.fixture
def sequence_tensors(sequence_collection: list[Sequence[Tensor]]) -> list[Tensor]:
    return [tensor_value for tensor_sequence in sequence_collection for tensor_value in tensor_sequence]


@pytest.fixture
def empty_tensor_sequence() -> list[Tensor]:
    return []


@pytest.fixture
def empty_optional_tensor_sequence() -> list[Tensor | None]:
    return []


@pytest.fixture(params=T_PARAMS)
def t(request: pytest.FixtureRequest) -> Tensor:
    return request.param


@pytest.fixture(
    params=[
        pytest.param((torch.tensor([2.0, 3.0, 5.0]), -2, 0), id="negative-left-only"),
        pytest.param((torch.tensor([7.0, 11.0, 13.0]), 0, -3), id="negative-right-only"),
        pytest.param((torch.tensor([[17.0, 19.0], [23.0, 29.0]]), -5, -7), id="both-negative"),
        pytest.param((torch.tensor([31.0, 37.0]), -1, 4), id="negative-left-positive-right"),
    ],
)
def tensor_malformed_padding(request: pytest.FixtureRequest) -> tuple[Tensor, int, int]:
    return request.param
