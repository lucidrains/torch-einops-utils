from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from typing import (
    Any,
    Concatenate,
    ParamSpec,
    Protocol,
    TypeGuard,
    TypeVar,
    overload
)

from torch import Tensor

D = TypeVar("D")
T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

P = ParamSpec("P")


class _SupportsGetItem(Protocol[T_co]):
    def __getitem__(self, index: int) -> T_co: ...


class _SupportsMod(Protocol[T_contra, T_co]):
    def __mod__(self, other: T_contra, /) -> T_co: ...


class _SupportsRMod(Protocol[T_contra, T_co]):
    def __rmod__(self, other: T_contra, /) -> T_co: ...


def default(v: T | None, d: D) -> T | D:
    return v if exists(v) else d


@overload
def divisible_by(num: T_contra, den: _SupportsRMod[T_contra, T_co]) -> bool: ...
@overload
def divisible_by(num: _SupportsMod[T_contra, T_co], den: T_contra) -> bool: ...
# TODO Narrow this at least a little.
def divisible_by(num: Any, den: Any) -> bool:
    with contextlib.suppress(ZeroDivisionError):
        return (num % den) == 0
    return False


def exists(v: T | None) -> TypeGuard[T]:
    return v is not None


def identity(t: T, *args: Any, **kwargs: Any) -> T:  # noqa: ANN401, ARG001
    return t


def first(arr: _SupportsGetItem[T_co]) -> T_co:
    return arr[0]


def compact(arr: Iterable[T | None]) -> list[T]:
    return [*filter(exists, arr)]


def safe(
    fn: Callable[Concatenate[Sequence[Tensor | None], P], Tensor | None],
) -> Callable[Concatenate[Sequence[Tensor | None], P], Tensor | None]:
    @wraps(fn)
    def inner(tensors: Sequence[Tensor | None], *args: P.args, **kwargs: P.kwargs) -> Tensor | None:
        compacted: list[Tensor] = compact(tensors)
        if len(compacted) == 0:
            output: Tensor | None = None
        elif len(compacted) == 1:
            output = compacted[0]
        else:
            output = fn(compacted, *args, **kwargs)
        return output

    return inner
