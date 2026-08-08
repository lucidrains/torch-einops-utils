from __future__ import annotations

import re
import inspect
from functools import lru_cache, reduce, wraps

import torch
from torch import is_tensor

# constants

ANONYMOUS_AXES = ('1', '_')
NAME_RE = re.compile(r'[\w\-]+')

# exceptions

# raised when a tensor shape does not match the given pattern

class ShapeError(AssertionError):
    pass

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d() if callable(d) else d

def divisible_by(num, den):
    return (num % den) == 0 if den != 0 else num == 0

def is_anonymous_or_num(name):
    return name in ANONYMOUS_AXES or name.isdigit()

def prod(arr):
    return reduce(lambda acc, x: acc * x, arr, 1)

def fail(reason):
    return None, None, None, reason

# parsing

def validate_name(name, pattern):
    if not is_anonymous_or_num(name):
        assert NAME_RE.fullmatch(name), f'pattern "{pattern}" has invalid axis name "{name}"'

@lru_cache(maxsize=256)
def parse_pattern(pattern):
    assert isinstance(pattern, str), f'pattern must be a string, got {type(pattern).__name__}'

    tokens = []
    i, n = 0, len(pattern)

    while i < n:
        char = pattern[i]

        if char.isspace():
            i += 1
            continue

        if char == '(':
            j = pattern.find(')', i)
            assert j != -1, f'pattern "{pattern}" has an unclosed parenthesis'
            assert '(' not in pattern[i + 1:j], f'pattern "{pattern}" has nested parentheses, which are not supported'

            group_str = pattern[i + 1:j].strip()
            assert len(group_str) > 0, f'pattern "{pattern}" has an empty group'

            group_tokens = group_str.split()

            if any('...' in tok for tok in group_tokens):
                if len(group_tokens) == 1:
                    token_str = group_tokens[0]
                    parts = token_str.split('...')
                    prefix, suffix = parts[0], parts[-1]

                    name = None
                    if prefix != '':
                        name = prefix
                    elif suffix != '' and not suffix.isdigit():
                        name = suffix

                    length = int(suffix) if suffix.isdigit() else None

                    if exists(name):
                        validate_name(name, pattern)

                    tokens.append(('ellipsis', name, length))
                    i = j + 1
                    continue

            for name in group_tokens:
                validate_name(name, pattern)

            tokens.append(('group', tuple(group_tokens)))
            i = j + 1
            continue

        j = i
        while j < n and not pattern[j].isspace() and pattern[j] not in '()':
            j += 1

        token_str = pattern[i:j]
        i = j

        if '...' in token_str:
            parts = token_str.split('...')
            prefix, suffix = parts[0], parts[-1]

            name = None
            if prefix != '':
                name = prefix
            elif suffix != '' and not suffix.isdigit():
                name = suffix

            length = int(suffix) if suffix.isdigit() else None

            if exists(name):
                validate_name(name, pattern)

            tokens.append(('ellipsis', name, length))
            continue

        name = token_str
        validate_name(name, pattern)
        tokens.append(('name', name))

    assert len(tokens) > 0, f'pattern "{pattern}" is empty'
    assert sum(tok[0] == 'ellipsis' and tok[2] is None for tok in tokens) <= 1, f'pattern "{pattern}" has more than one variable-length ellipsis'

    names = []
    seen = set()

    for tok in tokens:
        tok_type = tok[0]
        axis_names = ()

        if tok_type in ('name', 'group'):
            axis_names = tok[1] if tok_type == 'group' else (tok[1],)
        elif tok_type == 'ellipsis' and exists(tok[1]):
            axis_names = (tok[1],)

        for name in axis_names:
            if is_anonymous_or_num(name):
                continue

            assert name not in seen, f'pattern "{pattern}" repeats axis "{name}"'
            seen.add(name)
            names.append(name)

    return tokens, names

# matching

def match(tokens, shape, assertions):
    n_var_ellipsis = sum(tok[0] == 'ellipsis' and tok[2] is None for tok in tokens)

    fixed_len_sum = 0
    for tok in tokens:
        if tok[0] in ('name', 'group'):
            fixed_len_sum += 1
        elif tok[0] == 'ellipsis' and tok[2] is not None:
            fixed_len_sum += tok[2]

    if n_var_ellipsis > 0:
        if fixed_len_sum > len(shape):
            return fail(f'expected at least {fixed_len_sum} dims, got {len(shape)}')
        var_len = len(shape) - fixed_len_sum
    else:
        if fixed_len_sum != len(shape):
            return fail(f'expected {fixed_len_sum} dims, got {len(shape)}')
        var_len = 0

    pairs = []
    curr = 0

    for tok in tokens:
        tok_type = tok[0]

        if tok_type in ('name', 'group'):
            pairs.append((tok, shape[curr], curr, curr + 1))
            curr += 1
        elif tok_type == 'ellipsis':
            length = tok[2] if tok[2] is not None else var_len
            sub_shape = tuple(shape[curr:curr + length])
            pairs.append((tok, sub_shape, curr, curr + length))
            curr += length

    dims, indices = dict(), dict()
    known = dict(assertions)
    ellipsis_shape = None

    for tok, dim_val, start, end in pairs:
        tok_type = tok[0]

        if tok_type == 'ellipsis':
            name = tok[1]
            if exists(name):
                if name in known:
                    expected = tuple(known[name]) if isinstance(known[name], (tuple, list)) else known[name]
                    if tuple(dim_val) != expected:
                        return fail(f'axis "{name}" at position {start}:{end} should be {known[name]}, got {list(dim_val)}')
                dims[name] = list(dim_val)
                indices[name] = slice(start, end)
            else:
                ellipsis_shape = list(dim_val)
                indices['...'] = slice(start, end)
            continue

        if tok_type == 'name':
            name = tok[1]
            dim = dim_val

            if name == '_':
                pass
            elif name.isdigit():
                target_size = int(name)
                if dim != target_size:
                    return fail(f'axis at position {start} should be of size {target_size}, got {dim}')
            elif name in known and known[name] != dim:
                return fail(f'axis "{name}" at position {start} should be {known[name]}, got {dim}')
            else:
                dims[name] = dim
                indices[name] = start

            continue

        if tok_type == 'group':
            dim = dim_val
            group_names = tok[1]
            group_known = dict()
            for name in group_names:
                if name in known:
                    group_known[name] = known[name]
                elif name.isdigit():
                    group_known[name] = int(name)

            unknown = [name for name in group_names if name not in group_known and name != '_']
            known_product = prod(group_known.values())
            group_repr = f'({" ".join(group_names)})'

            if len(unknown) == 0:
                if known_product != dim:
                    return fail(f'group "{group_repr}" at position {start} should have product {known_product}, got {dim}')
            else:
                if not divisible_by(dim, known_product):
                    return fail(f'group "{group_repr}" at position {start} should have product divisible by {known_product}, got {dim}')

                if len(unknown) == 1:
                    known[unknown[0]] = dim // known_product if known_product != 0 else 0

            for name, size in known.items():
                if name in group_names and not is_anonymous_or_num(name):
                    dims[name] = size
                    indices[name] = start

    return dims, indices, ellipsis_shape, None

# main function

# validate a tensor against an einops-style pattern, returning a ParsedShape with named accessors

def shape(
    t,
    pattern,
    *,
    throw_error = True,
    **assertions
):
    assert is_tensor(t), f'shape() expects a tensor, got {type(t).__name__}'

    tokens, names = parse_pattern(pattern)

    SymInt = getattr(torch, 'SymInt', int)

    for name, value in assertions.items():
        assert isinstance(value, (int, SymInt, tuple, list)), f'assertion for axis "{name}" must be an int, tuple, or list, got {type(value).__name__}'
        assert name in names, f'asserted axis "{name}" is not in pattern "{pattern}"'

    dims, indices, ellipsis_shape, error = match(tokens, t.shape, assertions)

    if exists(error):
        if throw_error:
            raise ShapeError(f'tensor of shape {tuple(t.shape)} does not match pattern "{pattern}": {error}')
        return None

    return ParsedShape(t.shape, pattern, dims, indices, ellipsis_shape, tokens = tokens)

def is_shape(
    t,
    pattern,
    **assertions
) -> bool:
    return exists(shape(t, pattern, throw_error = False, **assertions))

# parsed shape

# result of shape() with attribute accessors for each named axis

class ParsedShape:

    def __init__(
        self,
        shape,
        pattern,
        dims,
        indices,
        ellipsis_shape,
        tokens = ()
    ):
        self._shape = tuple(shape)
        self._pattern = pattern
        self._dims = dict(dims)
        self._indices = dict(indices)
        self._ellipsis = list(ellipsis_shape) if exists(ellipsis_shape) else None
        self._tokens = tuple(tokens)

    @property
    def pattern(self): return self._pattern
    @property
    def shape(self): return self._shape
    @property
    def ndim(self): return len(self._shape)
    @property
    def dims(self): return dict(self._dims)
    @property
    def names(self): return tuple(self._dims.keys())
    @property
    def ellipsis(self): return self._ellipsis
    @property
    def total(self): return prod(self._shape)

    def axis(self, name):
        return self._indices.get(name)

    def replace(self, **sizes):
        shape = list(self._shape)

        for name, size in sizes.items():
            index_or_slice = self._indices.get(name)
            if not exists(index_or_slice):
                raise KeyError(f'axis "{name}" is not in pattern "{self._pattern}"')
            if isinstance(index_or_slice, slice):
                shape[index_or_slice] = list(size)
            else:
                shape[index_or_slice] = size

        return tuple(shape)

    def matches(self, other):
        if isinstance(other, ParsedShape):
            other = other.dims
        elif not isinstance(other, dict):
            raise TypeError(f'matches() expects a ParsedShape or dict, got {type(other).__name__}')

        return all(self._dims[name] == size for name, size in other.items() if name in self._dims)

    def __getattr__(self, name):
        dims = self.__dict__.get('_dims')
        if exists(dims) and name in dims:
            return dims[name]
        raise AttributeError(f'ParsedShape has no axis named "{name}"')

    def __getitem__(self, name):
        if name == '...':
            if exists(self._ellipsis):
                return self._ellipsis
            raise KeyError(f'pattern "{self._pattern}" has no ellipsis')
        if name not in self._dims:
            raise KeyError(f'axis "{name}" is not in pattern "{self._pattern}"')
        return self._dims[name]

    def __iter__(self):
        for tok in self._tokens:
            tok_type = tok[0]

            if tok_type == 'ellipsis':
                name = tok[1]
                if exists(name) and name in self._dims:
                    yield self._dims[name]
                elif exists(self._ellipsis):
                    yield self._ellipsis
                continue

            if tok_type == 'name':
                name = tok[1]
                if name in self._dims:
                    yield self._dims[name]
                continue

            if tok_type == 'group':
                group_names = tok[1]
                for name in group_names:
                    if name in self._dims:
                        yield self._dims[name]
                continue

    def __len__(self):
        return sum(1 for _ in self)

    def __eq__(self, other):
        if isinstance(other, ParsedShape):
            return self.names == other.names and self._shape == other._shape
        if is_tensor(other):
            other = other.shape
        if isinstance(other, (tuple, list)):
            return self._shape == tuple(other)
        return NotImplemented

    def __repr__(self):
        return f'ParsedShape(pattern = {self._pattern!r}, shape = {self._shape!r}, dims = {self._dims!r})'

# decorator

# decorator to validate the shapes of tensor arguments against einops-style patterns

def assert_shape(spec, **assertions):
    is_dict = isinstance(spec, dict)
    assert is_dict or isinstance(spec, str), f'assert_shape() expects a pattern string or dict, got {type(spec).__name__}'

    def decorator(fn):
        signature = inspect.signature(fn)

        @wraps(fn)
        def inner(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            if is_dict:
                for name, pattern in spec.items():
                    arg = bound.arguments.get(name)
                    if is_tensor(arg):
                        shape(arg, pattern, **assertions)
            else:
                arg_name = next(
                    (
                        p.name for p in signature.parameters.values()
                        if p.name not in ('self', 'cls') and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                    ),
                    None
                )
                arg = bound.arguments.get(arg_name)
                if is_tensor(arg):
                    shape(arg, spec, **assertions)

            return fn(*args, **kwargs)

        return inner

    return decorator
