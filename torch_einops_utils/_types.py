from __future__ import annotations

from collections.abc import Hashable
from os import PathLike
from typing import ParamSpec, Protocol, TypeAlias, TypedDict, TypeVar

from torch.nn import Module

DVar = TypeVar("DVar")
KVar = TypeVar("KVar", bound=Hashable)
TVar = TypeVar("TVar")
RVar = TypeVar("RVar")
T_co = TypeVar("T_co", covariant=True)
TypeModule = TypeVar("TypeModule", bound=Module)

PSpec = ParamSpec("PSpec")

StrPath: TypeAlias = str | PathLike[str]


class DimAndValue(TypedDict, total=False):
    dim: int
    value: float


class IdentityCallable(Protocol):
    def __call__(self, value: TVar, /, *args: object, **kwargs: object) -> TVar: ...


class SupportsIntIndex(Protocol[T_co]):
    def __getitem__(self, index: int, /) -> T_co: ...
