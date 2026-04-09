from typing import ParamSpec, Protocol, TypedDict, TypeVar

PSpec = ParamSpec("PSpec")

RVar = TypeVar("RVar")
TVar = TypeVar("TVar")
T_co = TypeVar("T_co", covariant=True)

class DimAndValue(TypedDict, total=False):
    dim: int
    value: float

class IdentityCallable(Protocol):
    def __call__(self, value: TVar, /, *args: object, **kwargs: object) -> TVar: ...

class SupportsIntIndex(Protocol[T_co]):
    def __getitem__(self, index: int, /) -> T_co: ...
