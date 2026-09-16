from __future__ import annotations

from collections import namedtuple
from functools import wraps
from inspect import signature

# decorator for a function returning (primary, *outputs)
# `@maybe_return('memories', 'hiddens')` returns just the primary by default
# `return_memories = True` returns `(tokens, memories)`, while `return_all = True` returns all outputs as a namedtuple
# the decorated function may declare any `return_{field}` in its signature to only compute that output when it is requested

def maybe_return(*field_names, primary = 'tokens', flag = 'return_all'):
    def decorator(fn):
        sig = signature(fn)
        accepted_keys = {f'return_{field}' for field in field_names if f'return_{field}' in sig.parameters}

        output_types = {}

        def get_output_type(fields):
            if fields not in output_types:
                output_types[fields] = namedtuple('Output', (primary, *fields))
            return output_types[fields]

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
            primary_out, *rest = out

            requested = tuple(field for field in field_names if returns[field])

            if not requested:
                return primary_out

            return get_output_type(requested)(primary_out, *(rest[field_names.index(field)] for field in requested))

        return wrapper

    return decorator
