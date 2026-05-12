from __future__ import annotations
from collections.abc import Callable
from typing import Generic, TypeVar
from typing_extensions import ParamSpec

from torch import nn
from torch_einops_utils.torch_einops_utils import compact, identity

PSpec = ParamSpec("PSpec")
RVar = TypeVar("RVar")

def Sequential(*modules: nn.Module | None) -> nn.Sequential:
    return nn.Sequential(*compact(modules))

class Identity(nn.Module):
    forward = staticmethod(identity)

class Lambda(nn.Module, Generic[PSpec, RVar]):
    def __init__(self, fn: Callable[PSpec, RVar]) -> None:
        super().__init__()
        self.fn: Callable[PSpec, RVar] = fn

    def forward(self, *args: PSpec.args, **kwargs: PSpec.kwargs) -> RVar:
        return self.fn(*args, **kwargs)
