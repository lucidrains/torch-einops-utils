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

def _ellipsis_token(token_str, pattern):
    parts = token_str.split('...')
    prefix, suffix = parts[0], parts[-1]

    name = prefix if prefix != '' else (suffix if suffix != '' and not suffix.isdigit() else None)
    length = int(suffix) if suffix.isdigit() else None

    if exists(name):
        validate_name(name, pattern)

    return ('ellipsis', name, length)

def _tokenize(pattern):
    tokens = []
    i, n = 0, len(pattern)

    while i < n:
        if pattern[i].isspace():
            i += 1
            continue

        if pattern[i] == '(':
            j = pattern.find(')', i)
            assert j != -1, f'pattern "{pattern}" has an unclosed parenthesis'
            assert '(' not in pattern[i + 1:j], f'pattern "{pattern}" has nested parentheses, which are not supported'

            group_tokens = pattern[i + 1:j].strip().split()
            assert len(group_tokens) > 0, f'pattern "{pattern}" has an empty group'

            if len(group_tokens) == 1 and '...' in group_tokens[0]:
                tokens.append(_ellipsis_token(group_tokens[0], pattern))
            else:
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
            tokens.append(_ellipsis_token(token_str, pattern))
            continue

        validate_name(token_str, pattern)
        tokens.append(('name', token_str))

    assert len(tokens) > 0, f'pattern "{pattern}" is empty'
    assert sum(tok[0] == 'ellipsis' and not exists(tok[2]) for tok in tokens) <= 1, f'pattern "{pattern}" has more than one variable-length ellipsis'

    return tokens

def _axis_names(tok):
    if tok[0] == 'ellipsis':
        return (tok[1],) if exists(tok[1]) else ()
    return tok[1] if tok[0] == 'group' else (tok[1],)

def _collect_names(tokens, pattern):
    names = []
    seen = set()

    for tok in tokens:
        for name in _axis_names(tok):
            if is_anonymous_or_num(name):
                continue
            assert name not in seen, f'pattern "{pattern}" repeats axis "{name}"'
            seen.add(name)
            names.append(name)

    return names

@lru_cache(maxsize=256)
def parse_pattern(pattern):
    assert isinstance(pattern, str), f'pattern must be a string, got {type(pattern).__name__}'

    left, *rest = pattern.split('->')
    assert len(rest) <= 1, f'pattern "{pattern}" has more than one arrow "->"'

    tokens = _tokenize(left)
    names = _collect_names(tokens, pattern)

    if len(rest) == 0:
        return tokens, names, None

    right = rest[0]
    assert len(right.strip()) > 0, f'pattern "{pattern}" has nothing after the arrow "->"'

    selection = []
    seen = set()
    left_has_ellipsis = any(tok[0] == 'ellipsis' for tok in tokens)

    for tok in _tokenize(right):
        if tok[0] == 'group':
            raise AssertionError(f'pattern "{pattern}" cannot use groups after the arrow "->"')

        if tok[0] == 'ellipsis':
            assert tok[2] is None, f'pattern "{pattern}" only supports bare "..." after the arrow "->"'
            assert left_has_ellipsis, f'pattern "{pattern}" uses "..." after the arrow "->" but the left side has no ellipsis'
            key = '...'
        else:
            if is_anonymous_or_num(tok[1]):
                continue
            key = tok[1]

        assert key == '...' or key in names, f'pattern "{pattern}" selects axis "{key}" that is not on the left side of "->"'
        assert key not in seen, f'pattern "{pattern}" repeats axis "{key}" after the arrow "->"'
        seen.add(key)
        selection.append(tok)

    return tokens, names, selection

# matching

def match(tokens, shape, assertions):
    fixed_len_sum = sum(
        1 if tok[0] in ('name', 'group') else (tok[2] or 0)
        for tok in tokens
    )

    n_var_ellipsis = sum(tok[0] == 'ellipsis' and not exists(tok[2]) for tok in tokens)

    if n_var_ellipsis > 0:
        if fixed_len_sum > len(shape):
            return fail(f'expected at least {fixed_len_sum} dims, got {len(shape)}')
        var_len = len(shape) - fixed_len_sum
    else:
        if fixed_len_sum != len(shape):
            return fail(f'expected {fixed_len_sum} dims, got {len(shape)}')
        var_len = 0

    dims, indices = dict(), dict()
    known = dict(assertions)
    ellipsis_shape = None
    curr = 0

    for tok in tokens:
        tok_type, is_ellipsis = tok[0], tok[0] == 'ellipsis'
        length = default(tok[2], var_len) if is_ellipsis else 1
        dim_val = tuple(shape[curr:curr + length]) if is_ellipsis else shape[curr]
        start, end = curr, curr + length
        curr = end

        if is_ellipsis:
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

            if is_anonymous_or_num(name):
                if name.isdigit() and dim_val != int(name):
                    return fail(f'axis at position {start} should be of size {int(name)}, got {dim_val}')
            elif name in known and known[name] != dim_val:
                return fail(f'axis "{name}" at position {start} should be {known[name]}, got {dim_val}')
            else:
                dims[name] = dim_val
                indices[name] = start

            continue

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
            if known_product != dim_val:
                return fail(f'group "{group_repr}" at position {start} should have product {known_product}, got {dim_val}')
        else:
            if not divisible_by(dim_val, known_product):
                return fail(f'group "{group_repr}" at position {start} should have product divisible by {known_product}, got {dim_val}')

            if len(unknown) == 1:
                known[unknown[0]] = dim_val // known_product if known_product != 0 else 0

        for name, size in known.items():
            if name in group_names and not is_anonymous_or_num(name):
                dims[name] = size
                indices[name] = start

    return dims, indices, ellipsis_shape, None

# main

def shape(
    t,
    pattern,
    *,
    throw_error = True,
    **assertions
):
    assert is_tensor(t), f'shape() expects a tensor, got {type(t).__name__}'

    tokens, names, selection = parse_pattern(pattern)

    SymInt = getattr(torch, 'SymInt', int)

    for name, value in assertions.items():
        assert isinstance(value, (int, SymInt, tuple, list)), f'assertion for axis "{name}" must be an int, tuple, or list, got {type(value).__name__}'
        assert name in names, f'asserted axis "{name}" is not in pattern "{pattern}"'

    dims, indices, ellipsis_shape, error = match(tokens, t.shape, assertions)

    if exists(error):
        if throw_error:
            raise ShapeError(f'tensor of shape {tuple(t.shape)} does not match pattern "{pattern}": {error}')
        return None

    return ParsedShape(t.shape, pattern, dims, indices, ellipsis_shape, tokens = tokens, selection = selection)

def is_shape(
    t,
    pattern,
    **assertions
) -> bool:
    assert '->' not in pattern, f'is_shape() does not support arrow patterns, given "{pattern}"'
    return exists(shape(t, pattern, throw_error = False, **assertions))

# parsed shape

def _extract_selection(tokens, selection, dims, indices, ellipsis):
    left_ellipsis_name = next(
        (tok[1] for tok in tokens if tok[0] == 'ellipsis' and exists(tok[1])),
        None
    )

    sel_dims, sel_indices, sel_items = dict(), dict(), []
    pos = 0

    for tok in selection:
        if tok[0] == 'ellipsis':
            item = ellipsis if exists(ellipsis) else dims[left_ellipsis_name]
            sel_indices['...'] = slice(pos, pos + len(item))
        else:
            name = tok[1]
            item = dims[name]
            sel_dims[name] = item
            sel_indices[name] = pos if not isinstance(item, (tuple, list)) else slice(pos, pos + len(item))

        sel_items.append(item)
        pos += len(item) if isinstance(item, (tuple, list)) else 1

    return sel_dims, sel_indices, sel_items

class ParsedShape:

    def __init__(
        self,
        shape,
        pattern,
        dims,
        indices,
        ellipsis_shape,
        tokens = (),
        selection = None
    ):
        self._shape = tuple(shape)
        self._pattern = pattern
        self._ellipsis = list(ellipsis_shape) if exists(ellipsis_shape) else None
        self._tokens = tuple(tokens)
        self._selection = None

        if exists(selection):
            dims, indices, self._selection = _extract_selection(self._tokens, selection, dims, indices, self._ellipsis)
            self._shape = tuple(dim for item in self._selection for dim in (item if isinstance(item, (tuple, list)) else (item,)))

        self._dims = dict(dims)
        self._indices = dict(indices)

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
            shape[index_or_slice] = list(size) if isinstance(index_or_slice, slice) else size

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
            if not exists(self._ellipsis):
                raise KeyError(f'pattern "{self._pattern}" has no ellipsis')
            return self._ellipsis
        if name not in self._dims:
            raise KeyError(f'axis "{name}" is not in pattern "{self._pattern}"')
        return self._dims[name]

    def __iter__(self):
        if exists(self._selection):
            yield from self._shape
            return

        for tok in self._tokens:
            if tok[0] == 'ellipsis':
                name = tok[1]
                if exists(name) and name in self._dims:
                    yield self._dims[name]
                elif exists(self._ellipsis):
                    yield self._ellipsis

            elif tok[0] == 'name':
                if tok[1] in self._dims:
                    yield self._dims[tok[1]]

            else:
                for name in tok[1]:
                    if name in self._dims:
                        yield self._dims[name]

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
                    if is_tensor(bound.arguments.get(name)):
                        shape(bound.arguments[name], pattern, **assertions)
            else:
                arg_name = next(
                    (
                        p.name for p in signature.parameters.values()
                        if p.name not in ('self', 'cls') and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                    ),
                    None
                )
                if is_tensor(bound.arguments.get(arg_name)):
                    shape(bound.arguments[arg_name], spec, **assertions)

            return fn(*args, **kwargs)

        return inner

    return decorator
