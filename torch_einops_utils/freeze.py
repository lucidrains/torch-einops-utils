from __future__ import annotations

from contextlib import contextmanager
from functools import wraps

from torch.nn import Module

from torch_einops_utils.selectors import resolve_parameters

# freezing / unfreezing
#   selectors -> glob / regex / module type / module instance / predicate
#   no selectors -> whole module

def _set(module, selectors, value, strict):
    for _, param in resolve_parameters(module, *selectors, strict = strict):
        param.requires_grad_(value)

    return module

def freeze(module, *selectors, strict = True):
    return _set(module, selectors, False, strict)

def unfreeze(module, *selectors, strict = True):
    return _set(module, selectors, True, strict)

def freeze_except(module, *selectors, strict = True):
    if not selectors:
        return freeze(module)

    keep = {name for name, _ in resolve_parameters(module, *selectors, strict = strict)}

    for name, param in module.named_parameters():
        if name not in keep:
            param.requires_grad_(False)

    return module

# optimizer ready params

def trainable_parameters(module, *selectors, strict = True):
    params = resolve_parameters(module, *selectors, strict = strict)
    return [param for _, param in params if param.requires_grad]

# scope, restore on exit
#   eval -> switch to eval mode for the block, per module training flags are restored

def frozen(module_or_fn, *selectors, strict = True, eval = False):
    if not isinstance(module_or_fn, Module):
        @wraps(module_or_fn)
        def inner(self, *args, **kwargs):
            with frozen(self, *selectors, strict = strict, eval = eval):
                return module_or_fn(self, *args, **kwargs)

        return inner

    @contextmanager
    def scope():
        targets = resolve_parameters(module_or_fn, *selectors, strict = strict)
        original = [(param, param.requires_grad) for _, param in targets]
        training = {m: m.training for m in module_or_fn.modules()} if eval else {}

        try:
            for param, _ in original:
                param.requires_grad_(False)

            if eval:
                module_or_fn.eval()

            yield module_or_fn
        finally:
            for param, value in original:
                param.requires_grad_(value)

            for m, value in training.items():
                m.training = value

    return scope()
