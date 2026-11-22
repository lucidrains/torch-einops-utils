from __future__ import annotations

from itertools import chain
from functools import wraps

import torch
from torch import Tensor
from torch.nn import Module

from torch_einops_utils.torch_einops_utils import tree_map_tensor

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# infer device and dtype from a tensor or module

def first_param_or_buffer(m: Module):
    return next(chain(m.parameters(), m.buffers()), None)

def module_device(m: Module):
    t = first_param_or_buffer(m)
    return t.device if exists(t) else None

def device_dtype(t: Tensor | Module, *, device = None, dtype = None):
    if isinstance(t, Module):
        t = first_param_or_buffer(t)
        assert exists(t), 'module has no parameters or buffers'

    return dict(
        device = default(device, t.device),
        dtype = default(dtype, t.dtype)
    )

def tree_map_tensor_to_device(tree, device):
    return tree_map_tensor(lambda t: t.to(device), tree)

# moving all inputs into a function onto a device

def move_inputs_to_device(device):

    def decorator(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            args, kwargs = tree_map_tensor(lambda t: t.to(device), (args, kwargs))

            return fn(*args, **kwargs)

        return inner

    return decorator

def move_inputs_to_module_device(fn):

    @wraps(fn)
    def inner(self, *args, **kwargs):
        device = module_device(self)

        if exists(device):
            args, kwargs = tree_map_tensor(lambda t: t.to(device), (args, kwargs))

        return fn(self, *args, **kwargs)

    return inner

# temporary cpu offload for ops without a kernel on some device

def fallback_from(device_type):
    device_type = getattr(device_type, 'type', device_type)

    def decorator(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            device = None

            def detect(t):
                nonlocal device

                if not exists(device) and t.device.type == device_type:
                    device = t.device

                return t

            tree_map_tensor(detect, (args, kwargs))

            if not exists(device):
                return fn(*args, **kwargs)

            args, kwargs = tree_map_tensor(lambda t: t.cpu(), (args, kwargs))

            return tree_map_tensor(lambda t: t.to(device), fn(*args, **kwargs))

        return inner

    return decorator
