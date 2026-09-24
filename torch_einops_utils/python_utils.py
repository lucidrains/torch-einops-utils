from __future__ import annotations

from collections import namedtuple
from functools import wraps
from inspect import signature

# decorator for a function returning the primary output(s), or `(primary, *outputs)`
# `@maybe_return('memories', 'hiddens')` returns just the primary by default
# `primary` may be a single field name, a tuple of names that are always returned, or a callable deriving the name from the call kwargs
# `return_memories = True` returns `(tokens, memories)`, while `return_all = True` returns all outputs as a namedtuple
# outputs may be returned positionally, aligned with the field names, or as a single dict keyed by field name
# the decorated function may declare any `return_{field}` in its signature to only compute that output when it is requested

def maybe_return(*field_names, primary = 'tokens', flag = 'return_all'):
    def decorator(fn):
        sig = signature(fn)
        accepted_keys = {f'return_{field}' for field in field_names if f'return_{field}' in sig.parameters}

        is_dynamic_name = callable(primary)

        if is_dynamic_name:
            fixed_names = None
        else:
            fixed_names = (primary,) if isinstance(primary, str) else tuple(primary)

        output_types = {}

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return_all = kwargs.pop(flag, False)

            returns = {
                field: return_all or kwargs.pop(f'return_{field}', False)
                for field in field_names
            }

            # pass the flags through, so the decorated function can skip computing an output that was not requested

            fn_kwargs = {f'return_{field}': value for field, value in returns.items() if f'return_{field}' in accepted_keys}
            fn_kwargs.update(kwargs)

            out = fn(*args, **fn_kwargs)
            outputs = out if isinstance(out, tuple) else (out,)

            names = (primary(fn_kwargs),) if is_dynamic_name else fixed_names
            primary_out, rest = outputs[:len(names)], outputs[len(names):]

            requested = tuple(field for field in field_names if returns[field])

            if not requested:
                return primary_out[0] if len(names) == 1 else primary_out

            # a lone dict names its outputs, positional outputs align with the field names, missing fields are `None`

            fields = rest[0] if len(rest) == 1 and isinstance(rest[0], dict) else dict(zip(field_names, rest))
            key = (names, requested)

            if key not in output_types:
                output_types[key] = namedtuple('Output', (*names, *requested))

            return output_types[key](*primary_out, *(fields.get(field) for field in requested))

        return wrapper

    return decorator
