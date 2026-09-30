from __future__ import annotations

import re
import inspect
from collections import namedtuple
from functools import wraps
from math import prod

import torch
from torch import is_tensor

# constants

ANONYMOUS_AXES = ('1', '_')
DELIMITERS = '()[]'
NAME_RE = re.compile(r'[\w\-]+')

SYM_INT = getattr(torch, 'SymInt', int)

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

# tokens
# a pattern is a sequence of factors, each of which consumes tensor dims
# 'name'     - one dim, bound to a name or a '1' / '_' literal
# 'select'   - consecutive dims, each bound to a name, marked for selection with [...]
# 'group'    - one dim, constrained to be the product of its names
# 'ellipsis' - `length` dims (variable length if None), optionally bound to a name

Token = namedtuple('Token', ('kind', 'names', 'length'), defaults = ((), None))

def _num_dims(tok):
    if tok.kind == 'ellipsis':
        return tok.length
    return len(tok.names) if tok.kind in ('name', 'select') else 1

def _is_variable(tok):
    return tok.kind == 'ellipsis' and not exists(tok.length)

# parsing

def validate_name(name, pattern):
    if not is_anonymous_or_num(name):
        assert NAME_RE.fullmatch(name), f'pattern "{pattern}" has invalid axis name "{name}"'

def _ellipsis_token(token_str, pattern):
    assert token_str.count('...') == 1, f'pattern "{pattern}" has invalid ellipsis "{token_str}"'

    prefix, suffix = token_str.split('...')
    name, length = None, None

    if suffix.isdigit():
        length = int(suffix)
        name = prefix or None
    elif suffix:
        assert not prefix, f'pattern "{pattern}" has invalid ellipsis "{token_str}"'
        name = suffix
    elif prefix:
        name = prefix

    if exists(name):
        validate_name(name, pattern)

    return Token('ellipsis', (name,) if exists(name) else (), length)

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
            assert not any(c in pattern[i + 1:j] for c in '[]'), f'pattern "{pattern}" cannot select axes inside a group'

            group_names = pattern[i + 1:j].strip().split()
            assert len(group_names) > 0, f'pattern "{pattern}" has an empty group'

            if len(group_names) == 1 and '...' in group_names[0]:
                tokens.append(_ellipsis_token(group_names[0], pattern))
            else:
                for name in group_names:
                    validate_name(name, pattern)
                tokens.append(Token('group', tuple(group_names)))

            i = j + 1
            continue

        if pattern[i] == '[':
            j = pattern.find(']', i)
            assert j != -1, f'pattern "{pattern}" has an unclosed bracket'
            assert not any(c in pattern[i + 1:j] for c in DELIMITERS), f'pattern "{pattern}" has invalid nested syntax inside brackets'

            selected = pattern[i + 1:j].split()
            assert len(selected) > 0, f'pattern "{pattern}" has an empty bracket'

            for name in selected:
                assert '...' not in name, f'pattern "{pattern}" cannot select an ellipsis'
                validate_name(name, pattern)
                assert not is_anonymous_or_num(name), f'pattern "{pattern}" can only select named axes, but got "{name}"'

            tokens.append(Token('select', tuple(selected)))
            i = j + 1
            continue

        if pattern[i] in ')]':
            closing = 'bracket' if pattern[i] == ']' else 'parenthesis'
            raise AssertionError(f'pattern "{pattern}" has an unmatched closing {closing}')

        j = i
        while j < n and not pattern[j].isspace() and pattern[j] not in DELIMITERS:
            j += 1

        token_str = pattern[i:j]
        i = j

        if '...' in token_str:
            tokens.append(_ellipsis_token(token_str, pattern))
            continue

        validate_name(token_str, pattern)
        tokens.append(Token('name', (token_str,)))

    assert len(tokens) > 0, f'pattern "{pattern}" is empty'
    assert sum(_is_variable(tok) for tok in tokens) <= 1, f'pattern "{pattern}" has more than one variable-length ellipsis'

    return tokens

def _collect_names(tokens, pattern):
    names = []
    seen = set()

    for tok in tokens:
        for name in tok.names:
            if is_anonymous_or_num(name):
                continue
            assert name not in seen, f'pattern "{pattern}" repeats axis "{name}"'
            seen.add(name)
            names.append(name)

    return names

_PATTERN_CACHE = dict()
_MAX_CACHE_SIZE = 512

def parse_pattern(pattern):
    cached = _PATTERN_CACHE.get(pattern)
    if cached is not None:
        return cached

    assert isinstance(pattern, str), f'pattern must be a string, got {type(pattern).__name__}'

    left, *rest = pattern.split('->')
    assert len(rest) <= 1, f'pattern "{pattern}" has more than one arrow "->"'

    tokens = _tokenize(left)
    names = _collect_names(tokens, pattern)

    selected = [name for tok in tokens if tok.kind == 'select' for name in tok.names]

    if len(rest) == 0:
        selection = [Token('name', (name,)) for name in selected] or None
        res = (tokens, names, selection)
    else:
        assert not selected, f'pattern "{pattern}" cannot combine brackets with "->"'

        right = rest[0]
        assert len(right.strip()) > 0, f'pattern "{pattern}" has nothing after the arrow "->"'

        selection = []
        seen = set()
        left_has_ellipsis = any(tok.kind == 'ellipsis' for tok in tokens)

        for tok in _tokenize(right):
            if tok.kind in ('group', 'select'):
                raise AssertionError(f'pattern "{pattern}" cannot use groups or brackets after the arrow "->"')

            if tok.kind == 'ellipsis':
                assert not exists(tok.length), f'pattern "{pattern}" only supports bare "..." after the arrow "->"'
                assert left_has_ellipsis, f'pattern "{pattern}" uses "..." after the arrow "->" but the left side has no ellipsis'
                key = '...'
            else:
                key = tok.names[0]
                assert not is_anonymous_or_num(key), f'pattern "{pattern}" can only select named axes after the arrow "->", but got "{key}"'

            assert key == '...' or key in names, f'pattern "{pattern}" selects axis "{key}" that is not on the left side of "->"'
            assert key not in seen, f'pattern "{pattern}" repeats axis "{key}" after the arrow "->"'
            seen.add(key)
            selection.append(tok)

        res = (tokens, names, selection)

    if len(_PATTERN_CACHE) >= _MAX_CACHE_SIZE:
        _PATTERN_CACHE.pop(next(iter(_PATTERN_CACHE)))

    _PATTERN_CACHE[pattern] = res
    return res

parse_pattern.cache_clear = _PATTERN_CACHE.clear

# matching

def match(tokens, shape, assertions):
    fixed_len_sum = sum(_num_dims(tok) or 0 for tok in tokens)

    if any(_is_variable(tok) for tok in tokens):
        if fixed_len_sum > len(shape):
            raise ShapeError(f'expected at least {fixed_len_sum} dims, got {len(shape)}')
        var_len = len(shape) - fixed_len_sum
    else:
        if fixed_len_sum != len(shape):
            raise ShapeError(f'expected {fixed_len_sum} dims, got {len(shape)}')
        var_len = 0

    dims, indices = dict(), dict()
    known = dict(assertions)
    ellipsis_shape = None
    curr = 0

    for tok in tokens:
        if tok.kind == 'ellipsis':
            length = default(tok.length, var_len)

            dim_val = tuple(shape[curr:curr + length])
            start, end = curr, curr + length
            curr = end

            name = tok.names[0] if tok.names else None

            if not exists(name):
                ellipsis_shape = list(dim_val)
                indices['...'] = slice(start, end)
                continue

            if name in known:
                expected = tuple(known[name]) if isinstance(known[name], (tuple, list)) else known[name]
                if tuple(dim_val) != expected:
                    raise ShapeError(f'axis "{name}" at position {start}:{end} should be {known[name]}, got {list(dim_val)}')

            dims[name] = list(dim_val)
            indices[name] = slice(start, end)
            continue

        if tok.kind == 'group':
            dim_val = shape[curr]
            start = curr
            curr += 1

            group_known = dict()
            for name in tok.names:
                if name in known:
                    group_known[name] = known[name]
                elif name.isdigit():
                    group_known[name] = int(name)

            unknown = [name for name in tok.names if name not in group_known and name != '_']
            known_product = prod(group_known.values())
            group_repr = f'({" ".join(tok.names)})'

            if len(unknown) == 0:
                if known_product != dim_val:
                    raise ShapeError(f'group "{group_repr}" at position {start} should have product {known_product}, got {dim_val}')
            else:
                if not divisible_by(dim_val, known_product):
                    raise ShapeError(f'group "{group_repr}" at position {start} should have product divisible by {known_product}, got {dim_val}')

                if len(unknown) == 1:
                    known[unknown[0]] = dim_val // known_product if known_product != 0 else 0

            for name, size in known.items():
                if name in tok.names and not is_anonymous_or_num(name):
                    dims[name] = size
                    indices[name] = start

            continue

        # name / select - one dim per name

        for name in tok.names:
            dim_val = shape[curr]
            start = curr
            curr += 1

            if is_anonymous_or_num(name):
                if name.isdigit() and dim_val != int(name):
                    raise ShapeError(f'axis at position {start} should be of size {int(name)}, got {dim_val}')
                continue

            if name in known and known[name] != dim_val:
                raise ShapeError(f'axis "{name}" at position {start} should be {known[name]}, got {dim_val}')

            dims[name] = dim_val
            indices[name] = start

    return dims, indices, ellipsis_shape

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

    for name, value in assertions.items():
        assert isinstance(value, (int, SYM_INT, tuple, list)), f'assertion for axis "{name}" must be an int, tuple, or list, got {type(value).__name__}'
        assert name in names, f'asserted axis "{name}" is not in pattern "{pattern}"'

    try:
        dims, indices, ellipsis_shape = match(tokens, t.shape, assertions)
    except ShapeError as err:
        if throw_error:
            raise ShapeError(f'tensor of shape {tuple(t.shape)} does not match pattern "{pattern}": {err}') from None
        return None

    return ParsedShape(t.shape, pattern, dims, indices, ellipsis_shape, tokens = tokens, selection = selection)

def is_shape(
    t,
    pattern,
    **assertions
) -> bool:
    assert '->' not in pattern, f'is_shape() does not support arrow patterns, given "{pattern}"'

    if not is_tensor(t):
        return False

    return exists(shape(t, pattern, throw_error = False, **assertions))

def size(
    t,
    pattern,
    **assertions
) -> int:
    return int(shape(t, pattern, **assertions))

# parsed shape

def _extract_selection(tokens, selection, dims, indices, ellipsis):
    left_ellipsis_name = next(
        (tok.names[0] for tok in tokens if tok.kind == 'ellipsis' and tok.names),
        None
    )

    sel_dims, sel_indices, sel_items = dict(), dict(), []
    pos = 0

    for tok in selection:
        if tok.kind == 'ellipsis':
            item = ellipsis if exists(ellipsis) else dims[left_ellipsis_name]
            sel_indices['...'] = slice(pos, pos + len(item))
        else:
            name = tok.names[0]
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
        self._all_dims = dict(dims)

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

    def keys(self):
        return self._dims.keys()

    def values(self):
        return self._dims.values()

    def items(self):
        return self._dims.items()

    def get(self, name, default = None):
        if name == '...':
            return self._ellipsis if exists(self._ellipsis) else default
        return self._dims.get(name, default)

    def replace(self, **sizes):
        shape = list(self._shape)

        for name in sizes:
            if name not in self._indices:
                raise KeyError(f'axis "{name}" is not in pattern "{self._pattern}"')

        def get_start(name):
            idx = self._indices[name]
            return idx.start if isinstance(idx, slice) else idx

        for name, size in sorted(sizes.items(), key = lambda item: get_start(item[0]), reverse = True):
            index_or_slice = self._indices[name]
            shape[index_or_slice] = list(size) if isinstance(size, (tuple, list)) else ([size] if isinstance(index_or_slice, slice) else size)

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

    def __contains__(self, name):
        if name == '...':
            return exists(self._ellipsis)
        return name in self._dims

    # unpacking - iteration yields the flat parsed shape by default
    # `unpack` is the single override point, backing both `iter` and `len`
    # e.g. a subclass may yield one value per pattern factor, with an ellipsis as a list

    def unpack(self):
        yield from self._shape

    def __iter__(self):
        return iter(self.unpack())

    def __len__(self):
        return sum(1 for _ in self)

    # int protocol - a parsed shape unpacking to a single dim can be used directly
    # e.g. `num_tokens = size(logits, '... [l]')`

    def __index__(self):
        if self.ndim != 1:
            raise TypeError(f'cannot interpret shape {self._shape} as an int - expected exactly one dim, got {self.ndim}')

        return self._shape[0]

    __int__ = __index__

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

# assert_shape

def _is_pair(spec):
    return (
        isinstance(spec, (tuple, list))
        and len(spec) == 2
        and is_tensor(spec[0])
        and isinstance(spec[1], str)
    )

def _validate_pairs(named_pairs, known):
    for name, t, pattern in named_pairs:
        _, names, _ = parse_pattern(pattern)

        try:
            parsed = shape(t, pattern, **{axis: size for axis, size in known.items() if axis in names})
        except ShapeError as err:
            label = f'argument "{name}": ' if exists(name) else ''
            raise ShapeError(f'{label}{err}') from None

        for axis, size in parsed._all_dims.items():
            known.setdefault(axis, size)

def assert_shape(spec, *patterns, **assertions):
    # direct invocation with tensor and pattern: assert_shape(t, 'b s d')

    if is_tensor(spec):
        assert len(patterns) == 1 and isinstance(patterns[0], str), \
            'assert_shape(tensor, pattern) expects a single pattern string'

        pairs = ((None, spec, patterns[0]),)

    # direct invocation with pair or list of pairs: assert_shape((t, 'b s d')) or assert_shape([(t, 'b s d'), ...])

    elif isinstance(spec, (tuple, list)):
        assert len(patterns) == 0, 'assert_shape() called directly expects a single (tensor, pattern) or list of pairs, got extra positional arguments'

        raw_pairs = (spec,) if _is_pair(spec) else spec
        assert isinstance(raw_pairs, (list, tuple)) and len(raw_pairs) > 0 and all(_is_pair(p) for p in raw_pairs), \
            f'assert_shape() called directly expects a (tensor, pattern) pair or list of pairs, got {type(spec).__name__}'

        pairs = tuple((None, t, pattern) for t, pattern in raw_pairs)

    else:
        pairs = None

    if exists(pairs):
        all_names = set()
        for _, _, pattern in pairs:
            _, names, _ = parse_pattern(pattern)
            all_names.update(names)

        for axis in assertions:
            assert axis in all_names, f'asserted axis "{axis}" is not in any pattern'

        _validate_pairs(pairs, dict(assertions))
        return

    # decorator invocation: @assert_shape('b s d') or @assert_shape({'x': 'b s d'})

    is_dict = isinstance(spec, dict)
    assert is_dict or isinstance(spec, str), f'assert_shape() expects a tensor, (tensor, pattern) pair, list of pairs, pattern string, or dict, got {type(spec).__name__}'

    def decorator(fn):
        signature = inspect.signature(fn)

        # normalize to a mapping of argument name -> pattern

        if is_dict:
            patterns_by_arg = spec
            has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())

            if not has_kwargs:
                for arg_name in spec:
                    assert arg_name in signature.parameters, f'argument "{arg_name}" not found in function signature'
        else:
            arg_name = next(
                (
                    p.name for p in signature.parameters.values()
                    if p.name not in ('self', 'cls') and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                ),
                None
            )
            patterns_by_arg = {arg_name: spec}

        all_names = set()
        for pattern in patterns_by_arg.values():
            _, names, _ = parse_pattern(pattern)
            all_names.update(names)

        for axis in assertions:
            assert axis in all_names, f'asserted axis "{axis}" is not in any pattern of {spec}'

        @wraps(fn)
        def inner(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()

            if not is_dict and not exists(arg_name):
                named_pairs = (
                    (None, arg, spec)
                    for arg in args[:1]
                    if is_tensor(arg)
                )
            else:
                named_pairs = (
                    (name, bound.arguments.get(name), pattern)
                    for name, pattern in patterns_by_arg.items()
                    if is_tensor(bound.arguments.get(name))
                )

            _validate_pairs(named_pairs, dict(assertions))

            return fn(*args, **kwargs)

        return inner

    return decorator
