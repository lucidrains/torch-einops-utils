from __future__ import annotations

from collections.abc import Callable
from itertools import chain
from functools import wraps
from typing import Concatenate, ParamSpec, TypeGuard, TypeVar

from torch import Tensor, device
from torch.nn import Module
from torch.types import Device

from torch_einops_utils import tree_map_tensor

TVar = TypeVar("TVar")
TorchNNModule = TypeVar("TorchNNModule", bound=Module)

PSpec = ParamSpec("PSpec")

# helpers

def exists(v: TVar | None) -> TypeGuard[TVar]:
    return v is not None

# infer the device for a module

def module_device(m: Module) -> device | None:

    first_param_or_buffer: Tensor | None = next(chain(m.parameters(), m.buffers()), None)

    if not exists(first_param_or_buffer):
        return None

    return first_param_or_buffer.device

# moving all inputs into a function onto a device

def move_inputs_to_device(device: Device) -> Callable[[Callable[PSpec, TVar]], Callable[PSpec, TVar]]:

    def decorator(fn: Callable[PSpec, TVar]) -> Callable[PSpec, TVar]:
        @wraps(fn)
        def inner(*args: PSpec.args, **kwargs: PSpec.kwargs) -> TVar:
            args, kwargs = tree_map_tensor(lambda t: t.to(device), (args, kwargs))

            return fn(*args, **kwargs)

        return inner

    return decorator

def move_inputs_to_module_device(fn: Callable[Concatenate[TorchNNModule, PSpec], TVar]) -> Callable[Concatenate[TorchNNModule, PSpec], TVar]:

    @wraps(fn)
    def inner(self: TorchNNModule, *args: PSpec.args, **kwargs: PSpec.kwargs) -> TVar:
        device: device | None = module_device(self)

        if exists(device):
            args, kwargs = tree_map_tensor(lambda t: t.to(device), (args, kwargs))

        return fn(self, *args, **kwargs)

    return inner
