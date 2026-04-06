from __future__ import annotations

from typing import ParamSpec, Protocol, TypedDict, TypeVar

DVar = TypeVar("DVar")
TVar = TypeVar("TVar")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

PSpec = ParamSpec("PSpec")


class DimAndValue(TypedDict, total=False):
    dim: int
    value: float


class SupportsGetItem(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co: ...


class SupportsMod(Protocol[T_contra, T_co]):
    def __mod__(self, other: T_contra, /) -> T_co: ...


class SupportsRMod(Protocol[T_contra, T_co]):
    def __rmod__(self, other: T_contra, /) -> T_co: ...
