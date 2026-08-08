import pytest
import torch
from torch import tensor

import torch_einops_utils
from torch_einops_utils.shape import (
    ParsedShape,
    ShapeError,
    shape,
    is_shape,
    assert_shape
)

# helpers

def exists(val):
    return val is not None

# basic parsing and accessors

def test_shape_basic():
    t = torch.randn(2, 3, 4)
    parsed = shape(t, 'b s d')

    assert isinstance(parsed, ParsedShape)
    assert parsed.b == 2
    assert parsed.s == 3
    assert parsed.d == 4

    assert parsed['b'] == 2
    assert parsed.names == ('b', 's', 'd')
    assert parsed.shape == (2, 3, 4)
    assert parsed.ndim == 3
    assert parsed.total == 24
    assert parsed.pattern == 'b s d'

    b, s, d = parsed
    assert (b, s, d) == (2, 3, 4)

    assert len(parsed) == 3

def test_shape_missing_axis():
    parsed = shape(torch.randn(2, 3), 'b s')

    with pytest.raises(AttributeError):
        parsed.d

    with pytest.raises(KeyError):
        parsed['d']

def test_shape_mismatch_throws_by_default():
    t = torch.randn(2, 3, 4)

    assert exists(shape(t, 'b s d'))

    with pytest.raises(ShapeError) as err:
        shape(t, 'b s d e')

    assert isinstance(err.value, AssertionError)
    assert 'b s d e' in str(err.value)
    assert '(2, 3, 4)' in str(err.value)

    with pytest.raises(ShapeError):
        shape(t, 'b s')

def test_shape_mismatch_returns_none_when_quiet():
    t = torch.randn(2, 3, 4)

    assert exists(shape(t, 'b s d', throw_error = False))
    assert not exists(shape(t, 'b s', throw_error = False))
    assert not exists(shape(t, 'b s d e', throw_error = False))

def test_is_shape():
    t = torch.randn(2, 3, 4)

    assert is_shape(t, 'b s d')
    assert not is_shape(t, 'b s')
    assert not is_shape(t, 'b s d e')
    assert is_shape(t, 'b s d', d = 4)
    assert not is_shape(t, 'b s d', d = 16)

def test_shape_assertions():
    t = torch.randn(2, 3, 512)

    parsed = shape(t, 'b s d', d = 512)
    assert parsed.d == 512

    with pytest.raises(ShapeError):
        shape(t, 'b s d', d = 128)

    with pytest.raises(ShapeError):
        shape(t, 'b s d', s = 4)

    assert not exists(shape(t, 'b s d', throw_error = False, d = 128))

def test_shape_assertion_not_in_pattern():
    t = torch.randn(2, 3)

    with pytest.raises(AssertionError):
        shape(t, 'b s', throw_error = True, d = 4)

# ellipsis

def test_shape_ellipsis():
    t = torch.randn(2, 3, 4, 5, 6)
    parsed = shape(t, 'b ... d')

    assert parsed.b == 2
    assert parsed.d == 6
    assert parsed.ellipsis == [3, 4, 5]
    assert parsed['...'] == [3, 4, 5]
    assert parsed.ndim == 5

    b, rest, d = parsed
    assert b == 2
    assert rest == [3, 4, 5]
    assert d == 6
    assert len(parsed) == 3

def test_shape_ellipsis_zero_dims():
    t = torch.randn(2, 3)
    parsed = shape(t, 'b ... d', throw_error = True)

    assert parsed.b == 2
    assert parsed.d == 3
    assert parsed.ellipsis == []

def test_shape_ellipsis_mismatch():
    t = torch.randn(2, 3, 4)

    with pytest.raises(ShapeError):
        shape(t, 'b ... s d e', throw_error = True)

    with pytest.raises(ShapeError):
        shape(t, 'b ...', throw_error = True, b = 99)

    parsed = shape(t, 'b ...')
    assert parsed.ellipsis == [3, 4]
    b, rest = parsed
    assert b == 2 and rest == [3, 4]

    parsed = shape(t, '... d')
    assert parsed.ellipsis == [2, 3]
    rest, d = parsed
    assert rest == [2, 3] and d == 4

def test_shape_named_ellipsis():
    t = torch.randn(2, 3, 10, 20, 30, 4, 5)

    parsed = shape(t, 'b t ... f...2')
    assert parsed.b == 2
    assert parsed.t == 3
    assert parsed.ellipsis == [10, 20, 30]
    assert parsed.f == [4, 5]
    assert parsed['f'] == [4, 5]
    assert parsed['...'] == [10, 20, 30]

    b, t_val, rest, f = parsed
    assert b == 2
    assert t_val == 3
    assert rest == [10, 20, 30]
    assert f == [4, 5]
    assert len(parsed) == 4

    parsed_assert = shape(t, 'b t ... f...2', f = (4, 5))
    assert parsed_assert.f == [4, 5]

    with pytest.raises(ShapeError):
        shape(t, 'b t ... f...2', f = (4, 6))

def test_shape_fixed_length_ellipsis_unnamed_and_named():
    t = torch.randn(2, 3, 4, 5)

    parsed = shape(t, 'b spatial...2 d')
    assert parsed.b == 2
    assert parsed.spatial == [3, 4]
    assert parsed.d == 5

    b, spatial, d = parsed
    assert b == 2 and spatial == [3, 4] and d == 5

    parsed2 = shape(t, 'b ...2 d')
    assert parsed2.b == 2
    assert parsed2.ellipsis == [3, 4]
    assert parsed2.d == 5

    b2, rest, d2 = parsed2
    assert b2 == 2 and rest == [3, 4] and d2 == 5

    assert parsed.replace(spatial = (100, 200)) == (2, 100, 200, 5)

def test_shape_axis():
    t = torch.randn(2, 3, 4, 5)

    parsed = shape(t, 'b ... s')
    assert parsed.axis('b') == 0
    assert parsed.axis('s') == 3
    assert not exists(parsed.axis('nope'))

    parsed = shape(t, 'b s ...')
    assert parsed.axis('s') == 1

    parsed = shape(torch.randn(2, 48), 'b (h w)', h = 8)
    assert parsed.axis('h') == 1

# groups

def test_shape_group_solves_unknown():
    t = torch.randn(2, 48, 3)
    parsed = shape(t, 'b (h w) d', h = 8)

    assert parsed.h == 8
    assert parsed.w == 6

def test_shape_group_all_known():
    t = torch.randn(2, 48, 3)
    parsed = shape(t, 'b (h w) d', h = 8, w = 6)

    assert parsed.h == 8
    assert parsed.w == 6

    with pytest.raises(ShapeError):
        shape(t, 'b (h w) d', h = 8, w = 7)

def test_shape_group_not_divisible():
    t = torch.randn(2, 50, 3)

    with pytest.raises(ShapeError):
        shape(t, 'b (h w) d', h = 8)

    assert not exists(shape(t, 'b (h w) d', throw_error = False, h = 8))

def test_shape_group_unknown_not_solvable():
    t = torch.randn(2, 48, 3)
    parsed = shape(t, 'b (h w) d')

    assert parsed.b == 2
    assert parsed.d == 3
    assert parsed.shape == (2, 48, 3)

    with pytest.raises(AttributeError):
        parsed.h

def test_shape_group_anonymous():
    t = torch.randn(2, 5)
    parsed = shape(t, 'b (1 d)')

    assert parsed.b == 2
    assert parsed.d == 5

def test_shape_parenthesized_ellipsis():
    t = torch.randn(2, 3, 4, 5, 6)

    parsed = shape(t, 'b ... (f...2) d')
    assert parsed.b == 2
    assert parsed.ellipsis == [3]
    assert parsed.f == [4, 5]
    assert parsed.d == 6

    parsed2 = shape(t, 'b (...) d')
    assert parsed2.ellipsis == [3, 4, 5]

# anonymous and wildcard dims

def test_shape_anonymous():
    t = torch.randn(1, 3, 4)

    parsed = shape(t, '1 s d')
    assert parsed.s == 3
    assert parsed.d == 4

    parsed = shape(t, '_ s d')
    assert parsed.s == 3

    with pytest.raises(ShapeError):
        shape(torch.randn(2, 3), '1 s')

    assert not exists(shape(torch.randn(2, 3), '1 s', throw_error = False))

def test_shape_wildcard_any_dim():
    t = torch.randn(5, 3, 4)
    parsed = shape(t, '_ s d')
    assert parsed.s == 3
    assert parsed.d == 4

def test_shape_numeric_literals():
    t = torch.randn(2, 16, 512)
    parsed = shape(t, 'b 16 d')
    assert parsed.b == 2
    assert parsed.d == 512

    with pytest.raises(ShapeError):
        shape(t, 'b 32 d')

    t_group = torch.randn(2, 32, 512)
    parsed_group = shape(t_group, 'b (h 2) d')
    assert parsed_group.h == 16
    assert parsed_group.d == 512

# zero dim and empty tensors

def test_shape_zero_dim_tensors():
    t_scalar = torch.tensor(3.14)
    parsed_scalar = shape(t_scalar, '...')
    assert parsed_scalar.shape == ()
    assert parsed_scalar.ellipsis == []
    assert parsed_scalar.total == 1

    t_empty = torch.empty(2, 0, 4)
    parsed_empty = shape(t_empty, 'b s d')
    assert parsed_empty.b == 2
    assert parsed_empty.s == 0
    assert parsed_empty.d == 4
    assert parsed_empty.total == 0

    parsed_solve_zero = shape(t_empty, 'b (s1 s2) d', s1 = 0)
    assert parsed_solve_zero.s1 == 0
    assert parsed_solve_zero.s2 == 0

# equality

def test_shape_equality():
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)
    z = torch.randn(2, 3, 5)

    assert shape(x, 'b s d') == shape(y, 'b s d')
    assert not (shape(x, 'b s d') == shape(z, 'b s d'))
    assert shape(x, 'b s d') != shape(y, 'b d s')
    assert shape(x, 'b s d') == (2, 3, 4)
    assert shape(x, 'b s d') == [2, 3, 4]
    assert shape(x, 'b s d') == torch.Size((2, 3, 4))
    assert shape(x, 'b s d') == y

    assert not (shape(x, 'b s d') == 3)

def test_shape_matches():
    x = torch.randn(2, 5, 8)
    mask = torch.randn(2, 5)

    parsed_x = shape(x, 'b s d')
    parsed_mask = shape(mask, 'b s')

    assert parsed_x.matches(parsed_mask)
    assert parsed_mask.matches(parsed_x)

    bad_mask = shape(torch.randn(2, 6), 'b s', s = 5, throw_error = False)
    assert not exists(bad_mask)

    assert parsed_x.matches({'b': 2, 's': 5})
    assert not parsed_x.matches({'b': 2, 's': 6})

    with pytest.raises(TypeError):
        parsed_x.matches((2, 5))

# replace

def test_shape_replace():
    parsed = shape(torch.randn(2, 3, 4), 'b s d')

    assert parsed.replace(s = 10) == (2, 10, 4)
    assert parsed.replace(b = 1, d = 64) == (1, 3, 64)

    with pytest.raises(KeyError):
        parsed.replace(nope = 5)

# invalid patterns and inputs

def test_shape_non_tensor_inputs():
    for bad in [None, 3, 'b s d']:
        with pytest.raises(AssertionError):
            shape(bad, 'b s d')

@pytest.mark.parametrize('pattern', [
    '',
    'b s)',
    '(b s',
    '()',
    'b (s (d))',
    'b ... s ... d',
    'b s b',
    'b . s',
    'b s ^'
])
def test_shape_invalid_patterns(pattern):
    with pytest.raises(AssertionError):
        shape(torch.randn(2), pattern)

def test_shape_non_string_pattern():
    with pytest.raises(AssertionError):
        shape(torch.randn(2), 3)

# decorator

def test_assert_shape_string():
    @assert_shape('b s d')
    def fn(x):
        return x.sum()

    out = fn(torch.randn(2, 3, 4))
    assert out.shape == ()

    with pytest.raises(ShapeError):
        fn(torch.randn(2, 3))

def test_assert_shape_dict():
    @assert_shape({'x': 'b s d', 'mask': 'b s'})
    def fn(x, mask = None):
        return x, mask

    fn(torch.randn(2, 3, 4), mask = torch.randn(2, 3))

    with pytest.raises(ShapeError):
        fn(torch.randn(2, 3, 4), mask = torch.randn(2, 4, 5))

    with pytest.raises(ShapeError):
        fn(torch.randn(2, 3))

def test_assert_shape_dict_skips_non_tensors():
    @assert_shape({'x': 'b s d', 'mask': 'b s'})
    def fn(x, mask = None):
        return x, mask

    fn(torch.randn(2, 3, 4), mask = None)

def test_assert_shape_positional_and_assertions():
    @assert_shape('b s d', d = 512)
    def fn(x):
        return x

    fn(torch.randn(2, 3, 512))

    with pytest.raises(ShapeError):
        fn(torch.randn(2, 3, 128))

def test_assert_shape_method():
    class Module:
        @assert_shape({'x': 'b s d'})
        def forward(self, x):
            return x

    module = Module()
    module.forward(torch.randn(2, 3, 4))

    with pytest.raises(ShapeError):
        module.forward(torch.randn(2, 3))

def test_assert_shape_varargs_kwargs():
    @assert_shape('b s d')
    def fn(*args, **kwargs):
        return args, kwargs

    @assert_shape('b s d')
    def fn_named(x, *args, **kwargs):
        return x

    fn_named(torch.randn(2, 3, 4))
    with pytest.raises(ShapeError):
        fn_named(torch.randn(2, 3))

def test_assert_shape_invalid_spec():
    with pytest.raises(AssertionError):
        @assert_shape(3)
        def fn(x):
            return x

def test_torch_compile():
    def compute(x):
        b, s, d = shape(x, 'b s d')
        return x * 2

    compiled_compute = torch.compile(compute)
    t = torch.randn(2, 3, 4)
    res = compiled_compute(t)
    assert res.shape == (2, 3, 4)
    assert torch.allclose(res, t * 2)

def test_not_exported_in_top_level_init():
    assert not hasattr(torch_einops_utils, 'assert_shape')
    assert not hasattr(torch_einops_utils, 'ParsedShape')
    assert not hasattr(torch_einops_utils, 'ShapeError')
