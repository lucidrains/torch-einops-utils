from __future__ import annotations

from collections import namedtuple
from functools import wraps
from inspect import signature

# decorator for a function returning the primary output(s), or `(primary, *outputs)`
# `@maybe_return('memories', 'hiddens')` returns just the primary by default
# `primary` may be a single field name, a tuple of names that are always returned, or a callable deriving the name(s) from the bound call arguments
# `return_memories = True` returns `(tokens, memories)`, while `return_all = True` returns all outputs as a namedtuple
# outputs may be returned positionally, aligned with the field names, or as a single dict keyed by field name
# the decorated function may declare any `return_{field}` in its signature to only compute that output when it is requested

def maybe_return(*field_names, primary = 'tokens', flag = 'return_all'):
    def decorator(fn):
        sig = signature(fn)

        known_flags = {flag, *(f'return_{field}' for field in field_names)}
        undeclared_flags = known_flags - set(sig.parameters)
        accepted_flags = {f'return_{field}' for field in field_names if f'return_{field}' in sig.parameters}

        is_dynamic = callable(primary)
        fixed_names = None if is_dynamic else ((primary,) if isinstance(primary, str) else tuple(primary))

        def resolve_names(call_args):
            if not is_dynamic:
                return fixed_names

            # a dynamic primary may return a single name or a tuple, e.g. depending on the mode of the call

            names = primary(call_args)

            return (names,) if isinstance(names, str) else tuple(names)

        output_types = {}

        @wraps(fn)
        def wrapper(*args, **kwargs):
            # flags the body does not declare are read, then dropped

            extra = {name: kwargs.pop(name) for name in undeclared_flags if name in kwargs}

            # resolve against the full call, whether given positionally or by keyword

            call_args = dict(sig.bind_partial(*args, **kwargs).arguments)
            call_args.update(extra)

            return_all = call_args.get(flag, False)

            returns = {
                field: return_all or call_args.get(f'return_{field}', False)
                for field in field_names
            }

            # forward declared flags unless already bound, so the body can skip computing unrequested outputs

            shorthand = {
                f'return_{field}': returns[field]
                for field in field_names
                if f'return_{field}' in accepted_flags and f'return_{field}' not in call_args
            }

            names = resolve_names(call_args)

            out = fn(*args, **kwargs, **shorthand)
            outputs = out if isinstance(out, tuple) else (out,)

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
