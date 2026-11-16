from __future__ import annotations

from fnmatch import fnmatchcase
from re import Pattern

from torch.nn import Module, Parameter

# selectors
#   str -> glob over qualified names
#   Pattern -> regex search over qualified names
#   Module instance / type -> names under matching modules
#   Parameter -> that exact tensor
#   callable -> predicate(name, obj)
#   list / tuple / set -> flattened

def _flatten(selectors):
    for selector in selectors:
        if isinstance(selector, (list, tuple, set)):
            yield from _flatten(selector)
        else:
            yield selector

def _is_module_type(selector):
    return isinstance(selector, type) and issubclass(selector, Module)

def _modules(module, selector):
    if isinstance(selector, Module):
        return (selector,)

    if _is_module_type(selector):
        return tuple(m for m in module.modules() if isinstance(m, selector))

    return None

def _names_under(named, modules):
    # match the modules themselves or their parameters
    ids = {id(m) for m in modules} | {id(p) for m in modules for p in m.parameters()}
    return {name for name, obj in named if id(obj) in ids}

def _match(module, named, selector, kind):
    if isinstance(selector, str):
        return {name for name, _ in named if fnmatchcase(name, selector)}

    if isinstance(selector, Pattern):
        return {name for name, _ in named if selector.search(name)}

    modules = _modules(module, selector)

    if modules is not None:
        return _names_under(named, modules)

    if kind == 'parameters' and isinstance(selector, Parameter):
        return {name for name, param in named if param is selector}

    if isinstance(selector, type):
        raise TypeError(f'unsupported selector {type(selector)}')

    if callable(selector):
        return {name for name, obj in named if selector(name, obj)}

    raise TypeError(f'unsupported selector {type(selector)}')

def _resolve(module, *selectors, kind = 'parameters', strict = True, default_all = True):
    named = list(module.named_modules() if kind == 'modules' else module.named_parameters())
    selectors = tuple(_flatten(selectors))

    if not selectors:
        return named if default_all else []

    matched = set()

    for selector in selectors:
        names = _match(module, named, selector, kind)

        if strict and not names:
            raise ValueError(f'no {kind} matched {selector!r}')

        matched |= names

    return [(name, obj) for name, obj in named if name in matched]

def resolve_modules(module, *selectors, strict = True, default_all = True):
    return _resolve(module, *selectors, kind = 'modules', strict = strict, default_all = default_all)

def resolve_parameters(module, *selectors, strict = True, default_all = True):
    return _resolve(module, *selectors, kind = 'parameters', strict = strict, default_all = default_all)
