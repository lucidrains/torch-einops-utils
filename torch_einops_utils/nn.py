from __future__ import annotations

from typing import Callable
from functools import partial

from torch.nn import Module
from torch.nn import Sequential as PyTorchSequential

from torch_einops_utils.torch_einops_utils import exists, tree_flatten_with_inverse

# helpers / functions

def count_parameters(model_or_class = None, *, requires_grad: bool | None = None):
    if not exists(model_or_class):
        return partial(count_parameters, requires_grad = requires_grad)

    def _count(model):
        return sum(p.numel() for p in model.parameters() if not exists(requires_grad) or p.requires_grad == requires_grad)

    if isinstance(model_or_class, type) and issubclass(model_or_class, Module):
        model_or_class.num_parameters = property(_count)
        return model_or_class

    return _count(model_or_class)

def Sequential(*modules):
    return PyTorchSequential(*filter(exists, modules))

# classes

class Identity(Module):
    def forward(self, t, *args, **kwargs):
        return t

class Lambda(Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, t, *args, **kwargs):
        return self.fn(t, *args, **kwargs)

class Residual(Module):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        out = self.fn(x, *args, **kwargs)

        (first, *rest), inverse = tree_flatten_with_inverse(out)
        return inverse((first + x, *rest))
