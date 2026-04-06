from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence
from functools import wraps
from typing import Any, Concatenate, TypeGuard, overload

from torch import Tensor

from torch_einops_utils import (
    DVar,
    PSpec,
    SupportsGetItem,
    SupportsMod,
    SupportsRMod,
    T_co,
    T_contra,
    TVar
)


@overload
def default(v: None, d: DVar) -> DVar: ...
@overload
def default(v: TVar, d: object) -> TVar: ...
def default(v: TVar | None, d: DVar) -> TVar | DVar:
    return v if exists(v) else d


# TODO Narrow this at least a little.
@overload
def divisible_by(num: T_contra, den: SupportsRMod[T_contra, T_co]) -> bool: ...
@overload
def divisible_by(num: SupportsMod[T_contra, T_co], den: T_contra) -> bool: ...
def divisible_by(num: Any, den: Any) -> bool:
    with contextlib.suppress(ZeroDivisionError):
        return (num % den) == 0
    return False


def exists(v: TVar | None) -> TypeGuard[TVar]:
    return v is not None


def identity(t: TVar, *args: Any, **kwargs: Any) -> TVar:  # noqa: ANN401, ARG001
    return t


def first(arr: SupportsGetItem[T_co]) -> T_co:
    return arr[0]


def compact(arr: Iterable[TVar | None]) -> list[TVar]:
    return [*filter(exists, arr)]


# TODO Docstring needs to tell the user that `fn` must be able to handle len(compacted) == 1
def safe(
    fn: Callable[Concatenate[Sequence[Tensor], PSpec], Tensor | None],
) -> Callable[Concatenate[Sequence[Tensor | None], PSpec], Tensor | None]:
    @wraps(fn)
    def inner(tensors: Sequence[Tensor | None], *args: PSpec.args, **kwargs: PSpec.kwargs) -> Tensor | None:
        compacted: list[Tensor] = compact(tensors)
        if len(compacted) == 0:
            output: Tensor | None = None
        else:
            output = fn(compacted, *args, **kwargs)
        return output

    return inner
